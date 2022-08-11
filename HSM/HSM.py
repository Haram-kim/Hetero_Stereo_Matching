"""
edit: Haram Kim
email: rlgkfka614@gmail.com
github: https://github.com/haram-kim
homepage: https://haram-kim.github.io/
"""

from signal import SIG_DFL
from termios import VMIN
import numpy as np
import cv2

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import time

import copy

from utils import *
from metric import *
from filter import *

from Camera import Camera, CameraDSEC
from FeatureTracker import FeatureTrackerKeyframeBased, FeatureTracker

import cuda_source as cuda_source

import math
import scipy.io

class HSM:
    def __init__(self, params):
        self.cam0 = params['cam0']
        self.cam1 = params['cam1']
        self.loader = params['loader']
        self.generate_kernel()

        self.feature_tracker = FeatureTracker(params)

        # DISPARITY PARAMETERS
        self.min_disparity = -params['disparity_range']
        self.max_disparity = 0
        self.ncc_gaussian_std = params['ncc_gaussian_std']
        self.ncc_AEI_gaussian_std = 3
        self.focal = (self.cam0.K_rect[0][0] + self.cam0.K_rect[1][1])/2
        self.f_b = self.cam0.baseline * self.focal
        self.disp_range = np.abs(np.linspace(self.min_disparity, self.max_disparity, self.max_disparity-self.min_disparity+1))
        self.depth_range = (self.f_b / (self.disp_range + 1e-9)).astype(np.float32)
        self.pi_v_gap = params['msd_gap']
        self.kernel_radius = params['kernel_radius']
        self.filter_radius = np.min([self.ncc_AEI_gaussian_std, self.kernel_radius])
        # self.filter_radius = 1

        self.disparity = np.zeros(Camera.resolution, dtype=np.float32)
        self.disparity_init = np.zeros(Camera.resolution, dtype=np.float32)
        self.disparity_AEI = np.zeros(Camera.resolution, dtype=np.float32)

        ### CUDA PARAMETERS
        kernel_radius = self.kernel_radius
        self.cuda_EBS = (1024, 1, 1) # gpu event block size
        self.cuda_EGS = (8096, 1)# gpu event grid size | max = 65535
        self.cuda_SBS = (32, 16, 1) # gpu event block size_x
        self.cuda_SGS = (int(np.ceil(Camera.width/self.cuda_SBS[0])),
                        int(np.ceil(Camera.height/self.cuda_SBS[1])))
        
        cuda_code = cuda_source.template.substitute(WIDTH = Camera.width,
                           HEIGHT = Camera.height,
                           BLOCKDIM_X = self.cuda_SBS[0],
                           BLOCKDIM_Y = self.cuda_SBS[1],
                           MIN_DISP = self.min_disparity,
                           MAX_DISP = self.max_disparity,
                           RAD = self.kernel_radius,
                           FILTER_RAD = self.filter_radius,
                           FX = Camera.K_rect[0,0],
                           FY = Camera.K_rect[1,1],
                           CX = Camera.K_rect[0,2],
                           CY = Camera.K_rect[1,2],
                           FxB = self.f_b)
        self.cuda_module = SourceModule(cuda_code)
        self.event_projection = self.cuda_module.get_function('event_projection') 
        self.clear_event_image = self.cuda_module.get_function('clear_event_image')   
        self.clear_AEI = self.cuda_module.get_function('clear_AEI')   
        self.compute_AEI = self.cuda_module.get_function('compute_AEI')     
        self.stereo_ncc = self.cuda_module.get_function('stereo_ncc')
        self.stereo_ncc_AEI = self.cuda_module.get_function('stereo_ncc_AEI')
        self.stereo_postproc = self.cuda_module.get_function('stereo_postproc')
        self.stereo_postproc_AEI = self.cuda_module.get_function('stereo_postproc_AEI')
        self.densify_sparse_disp = self.cuda_module.get_function('densify_sparse_disp')
        

        gaussian1d = np.zeros((2*kernel_radius+1, 1))
        for i in range(2 * kernel_radius + 1):
            x = i - kernel_radius
            gaussian1d[i] = np.exp(-(x * x) / (4*kernel_radius))
        self.gaussian2d = gaussuian_filter(kernel_radius, self.ncc_gaussian_std)
        self.gaussian2d_AEI = gaussuian_filter(kernel_radius, self.ncc_AEI_gaussian_std)
        self.gaussian2d_gpu = cuda.In(self.gaussian2d.astype(np.float32)/np.max(self.gaussian2d.astype(np.float32)))
        self.gaussian2d_AEI_gpu = cuda.In(self.gaussian2d_AEI.astype(np.float32))
        
        self.tex2D_left = self.cuda_module.get_texref("tex2D_left")
        self.tex2D_right = self.cuda_module.get_texref("tex2D_right")
        self.tex2D_left_edge = self.cuda_module.get_texref("tex2D_left_edge")

        self.zero_int32 = np.zeros(Camera.resolution, dtype=np.int32)
        self.zero_float32 = np.zeros(Camera.resolution, dtype=np.float32)
        self.zero_volume_float32 = np.zeros((self.max_disparity - self.min_disparity + 1,) + Camera.resolution, dtype=np.float32)

        self.disparity_AEI_gpu = cuda.mem_alloc(self.zero_float32.nbytes)
        self.disparity_temp_gpu = cuda.mem_alloc(self.zero_float32.nbytes)        
        self.disparity_gpu = cuda.mem_alloc(self.zero_float32.nbytes) 
        self.cost_gpu = cuda.mem_alloc(self.zero_volume_float32.nbytes)
        self.cost_AEI_gpu = cuda.mem_alloc(self.zero_volume_float32.nbytes)
        self.AEI_gpu = cuda.mem_alloc(self.zero_volume_float32.nbytes)  
    
        # ITERATOR
        self.iter = -1
        # CAMERA POSE
        self.T_hist = {}
        self.image_diag_len = np.linalg.norm(Camera.resolution)
        # EVAL DATA
        self.eval_data = dict()  
        self.eval_data['gt'] = dict()
        self.log_data = dict()
        
        # FLAGS
        self.ts_first = 0
        self.ts_event_first = 0
        self.is_first_process = True
        self.is_dIdt_init = False
        self.is_init = False
        self.is_time_estimation_mode = params['time_estimation']
        self.show_disparity_inlier = params['show_disparity_inlier']
        self.show_disparity = params['show_disparity']
        self.semi_dense = params['semi_dense']


    def __del__(self):
        self.disparity_AEI_gpu.free()
        self.disparity_temp_gpu.free()
        self.disparity_gpu.free()
        self.cost_gpu.free()
        self.cost_AEI_gpu.free()
        self.AEI_gpu.free()

    def process(self, image, image_ts, event):
        self.iter += 1
        image_ts = image_ts - self.ts_first + self.ts_event_first
        self.preprocess(image, image_ts, event)
        if self.is_dIdt_init:
            if not self.is_init:
                self.compute_disparity_ncc_init()
                self.T_hist[self.iter] = np.eye(4)        
                self.is_init = True
            else:
                try:
                    self.find_pose_from_pts3d()
                except:
                    self.is_init = False
                    return
                
                # self.compute_disparity_ncc_init()
                if self.is_time_estimation_mode:          
                    self.compute_disparity_ncc_time()
                else:
                    self.compute_disparity_ncc()

                self.visualize()

    def preprocess(self, image, image_ts, event):
        if self.is_first_process:
            self.dt = 0
            self.ts_first = image_ts
            self.ts_event_first = event[-1, 2]
            self.image_prev = np.zeros(Camera.resolution)
            self.is_first_process = False
            self.ts = np.max(event[:, 2])
            image_ts = np.max(event[:, 2])
        else:
            self.dt = image_ts - self.ts
            self.image_prev = self.image
            self.is_dIdt_init = True
            self.ts = image_ts

        self.rectify(image, image_ts, event)
        self.compute_gradient()
        self.extract_features()
        self.compute_event_image()  

    def rectify(self, image, image_ts, event):
        event[:, 2] = image_ts - event[:, 2]

        event = event[::1,:]
        self.image = self.cam0.rectify(image)
        self.event_raw = event.astype(np.float32)
        self.event_raw_cuda = cuda.In(self.event_raw)        
        self.event = self.cam1.rectify_events(self.event_raw)
        self.event_cuda = cuda.In(self.event)

    def compute_gradient(self):
        self.dIdt = self.image.astype(np.float32) - self.image_prev.astype(np.float32)
        self.image = np.array(self.image).astype(np.float32)
        dIdu = cv2.filter2D(self.image, -1, self.kernel_du)
        dIdv = cv2.filter2D(self.image, -1, self.kernel_dv)
        self.dIdx = np.sqrt(np.array(dIdu**2 + dIdv**2))

        self.dIdt_sigmoid = (1.0/(1 + np.exp(-cv2.GaussianBlur(self.dIdt.astype(np.float32)/255, (5, 5), 1)))-0.5)
        self.dIdx_sigmoid = (1.0/(1 + np.exp(-cv2.GaussianBlur(self.dIdx.astype(np.float32)/255, (5, 5), 1)))-0.5)
        cuda.matrix_to_texref(self.dIdx_sigmoid, self.tex2D_left_edge, order="C")

    def extract_features(self):
        self.feature_tracker.detect_feature(self.image)

    def compute_event_image(self):
        result_gpu = cuda.mem_alloc(self.zero_int32.nbytes)
        self.clear_event_image(result_gpu, block=self.cuda_SBS, grid=self.cuda_SGS)
        self.event_projection(result_gpu, self.event_raw_cuda, np.uint32(self.event.shape[0]), block=self.cuda_EBS, grid=self.cuda_EGS)
        image = cuda.from_device_like(result_gpu, self.zero_int32)
        
        self.event_image = self.cam1.rectify(image)
        self.event_sigmoid = 1./(1 + np.exp(-self.event_image.astype(np.float32)*0.5))-0.5
        result_gpu.free()
        return
        
    def compute_disparity_ncc_init(self):
        cuda.matrix_to_texref(self.dIdt_sigmoid, self.tex2D_left, order="C")
        cuda.matrix_to_texref(self.event_sigmoid, self.tex2D_right, order="C")
        self.stereo_ncc(self.disparity_temp_gpu, self.cost_gpu, block=self.cuda_SBS, grid=self.cuda_SGS)
        self.stereo_postproc(self.disparity_gpu, self.cost_gpu, self.gaussian2d_gpu, block=(self.cuda_SBS), grid=self.cuda_SGS)
        cuda.memcpy_dtoh(self.disparity, self.disparity_gpu)
        
        self.depth = self.f_b / (self.disparity + 1e-9)
        self.feature_tracker.map_init(self.depth)        
        self.disparity_init = copy.deepcopy(self.disparity)

    def compute_disparity_ncc(self):
        xi = InvSE3(self.T_hist[self.iter]).astype(np.float32)/self.dt
        depth_msd, AEI_idx = self.compute_msd(xi)
        depth_msd_gpu = cuda.In(depth_msd.astype(np.float32))
        self.AEI_idx_gpu = cuda.In(AEI_idx.astype(np.int32))
        
        # compute aligned event images (AEI)
        # set memory
        cuda.matrix_to_texref(self.dIdx_sigmoid, self.tex2D_left_edge, order="C")
        cuda.matrix_to_texref(self.dIdt_sigmoid, self.tex2D_left, order="C")
        cuda.matrix_to_texref(self.event_sigmoid, self.tex2D_right, order="C")

        # CUDA functions
        self.clear_AEI(self.AEI_gpu, block=self.cuda_SBS, grid=self.cuda_SGS)
        self.compute_AEI(self.AEI_gpu, self.event_cuda, depth_msd_gpu, 
            np.int32(self.event.shape[0]), np.int32(depth_msd.shape[0]),
            cuda.In(xi), block=self.cuda_EBS, grid=self.cuda_EGS)
        self.stereo_ncc_AEI(self.disparity_AEI_gpu, self.cost_AEI_gpu, self.AEI_gpu, self.AEI_idx_gpu, block=self.cuda_SBS, grid=self.cuda_SGS)  
        self.stereo_ncc(self.disparity_temp_gpu, self.cost_gpu, block=self.cuda_SBS, grid=self.cuda_SGS)
        self.stereo_postproc_AEI(self.disparity_gpu, self.cost_gpu, self.cost_AEI_gpu, 
            self.gaussian2d_AEI_gpu, block=(self.cuda_SBS), grid=self.cuda_SGS)
        # copy cuda memory        
        cuda.memcpy_dtoh(self.disparity, self.disparity_gpu)
        cuda.memcpy_dtoh(self.disparity_init, self.disparity_temp_gpu)
        cuda.memcpy_dtoh(self.disparity_AEI, self.disparity_AEI_gpu)
        self.depth = self.f_b / (self.disparity + 1e-9)
        self.feature_tracker.map_update(self.depth, self.T_hist[self.iter])
        return
            
    def find_pose_from_pts3d(self):
        T = self.feature_tracker.compute_camera_pose()
        self.T_hist[self.iter] = T
        return

    def compute_msd(self, xi):        
        v_dt = xi[:3]*self.dt + 0.5*np.matmul(hat(xi[3:]), xi[:3])*self.dt**2
        v_xy = np.linalg.norm(v_dt[:2])
        v_z = np.abs(v_dt[2])
        pi_v = (self.focal * v_xy + self.image_diag_len * v_z) / self.depth_range
        round_pi_v = np.ceil((pi_v)/self.pi_v_gap)*self.pi_v_gap
        _, pi_v_idx, AEI_idx = np.unique(round_pi_v,return_index=True,return_inverse=True)
        depth_msd = self.depth_range[pi_v_idx]
        return depth_msd, AEI_idx

    def visualize(self):
        if self.show_disparity:
            if self.semi_dense:
                disp_color = cv2.applyColorMap((self.dIdx_sigmoid**2 > 1e-4)*(self.disparity/(self.max_disparity - self.min_disparity)*255).astype(np.uint8), cv2.COLORMAP_JET)
                cv2.imshow("disparity", disp_color)
            else:
                disp_color = cv2.applyColorMap((self.disparity/(self.max_disparity - self.min_disparity)*255).astype(np.uint8), cv2.COLORMAP_JET)
                cv2.imshow("disparity", disp_color)          
        cv2.waitKey(1)

    def generate_kernel(self):
        self.kernel_du = np.array([[0.0, 0, 0], [-1, 0, 1], [0, 0, 0]])
        self.kernel_dv = np.transpose(self.kernel_du)
        return 

    def evaluate(self, idx):        
        success, _, image_disparity, _ = self.loader.get_debug_data(idx)

        if not success:
            return

        gt_disparity = image_disparity
        result_gpu = cuda.mem_alloc(self.zero_float32.nbytes)
        sparse_gpu = cuda.mem_alloc(self.zero_float32.nbytes)        
        cuda.memcpy_htod(result_gpu, self.zero_int32)
        cuda.memcpy_htod(sparse_gpu, gt_disparity)
        self.densify_sparse_disp(result_gpu, sparse_gpu, self.gaussian2d_AEI_gpu, block=(self.cuda_SBS), grid=self.cuda_SGS)
        gt_disparity_dense = cuda.from_device_like(result_gpu, self.zero_float32)
        disparity_gt_edge = (self.dIdx_sigmoid**2 > 4e-4)  * (gt_disparity>0) * gt_disparity_dense
        gt_disparity =  (self.dIdx_sigmoid**2 > 1e-4) * gt_disparity_dense
        self.disparity_gt = gt_disparity_dense

        if self.show_disparity_inlier:
            err_thres = 10
            disp_inlier = (self.disparity > 0) * (np.abs(gt_disparity_dense - self.disparity)< err_thres) * (gt_disparity>0)
            disp_color = cv2.applyColorMap(disp_inlier * (self.disparity/(self.max_disparity - self.min_disparity)*255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imshow("disparity_inlier", disp_color)
            cv2.waitKey(1)

        self.metric('init', self.disparity_init, disparity_gt_edge, 3)
        self.metric('proposed', self.disparity, disparity_gt_edge, 3)
        return

    def compute_disparity_ncc_time(self):
        repeat = 10
        self.repeat = repeat
        st_all = time.time()
        for i in range(0, repeat):
            st = time.time()
            self.clear_AEI(self.AEI_gpu, block=self.cuda_SBS, grid=self.cuda_SGS)
            self.logger('clear_AEI', time.time() - st)

        for i in range(0, repeat):
            st = time.time()
            xi = InvSE3(self.T_hist[self.iter]).astype(np.float32)/self.dt
            depth_msd, AEI_idx = self.compute_msd(xi)
            depth_msd_gpu = cuda.In(depth_msd.astype(np.float32))
            AEI_idx_gpu = cuda.In(AEI_idx.astype(np.int32))
            self.logger('compute MSD', time.time() - st)

        for i in range(0, repeat):
            st = time.time()
            cuda.matrix_to_texref(self.dIdx_sigmoid, self.tex2D_left_edge, order="C")
            cuda.matrix_to_texref(self.dIdt_sigmoid, self.tex2D_left, order="C")
            cuda.matrix_to_texref(self.event_sigmoid, self.tex2D_right, order="C")
            self.logger('copy texture', time.time() - st)

        for i in range(0, repeat):
            st = time.time()
            self.compute_AEI(self.AEI_gpu, self.event_cuda, depth_msd_gpu, np.int32(self.event.shape[0]),
                np.int32(depth_msd.shape[0]), cuda.In(xi), block=self.cuda_EBS, grid=self.cuda_EGS)       
            self.logger('compute AEI', time.time() - st)   

        for i in range(0, repeat):
            st = time.time()
            self.stereo_ncc(self.disparity_temp_gpu, self.cost_gpu, block=self.cuda_SBS, grid=self.cuda_SGS)
            cuda.memcpy_dtoh(self.disparity_init, self.disparity_temp_gpu)
            self.logger('stereo NCC', time.time() - st)

        for i in range(0, repeat):
            st = time.time()
            self.stereo_ncc_AEI(self.disparity_AEI_gpu, self.cost_AEI_gpu, self.AEI_gpu, AEI_idx_gpu, block=self.cuda_SBS, grid=self.cuda_SGS)
            cuda.memcpy_dtoh(self.disparity_AEI, self.disparity_AEI_gpu)
            self.logger('stereo NCC AEI', time.time() - st)

        for i in range(0, repeat):
            st = time.time()
            self.stereo_postproc_AEI(self.disparity_gpu, self.cost_gpu, self.cost_AEI_gpu, self.gaussian2d_AEI_gpu, block=self.cuda_SBS, grid=self.cuda_SGS)
            cuda.memcpy_dtoh(self.disparity, self.disparity_gpu)
            self.logger('stereo postprocessing', time.time() - st)

        for i in range(0, repeat):
            st = time.time()
            cuda.memcpy_dtoh(self.disparity, self.disparity_gpu)
            self.logger('copy device to host', time.time() - st)

        self.logger('total time', time.time() - st_all)
        self.logger('the number of event', self.event.shape[0])

        self.depth = self.f_b / (self.disparity + 1e-9)
        self.feature_tracker.map_update(self.depth, self.T_hist[self.iter])       

    def metric(self, method, disparity, disparity_gt, err_thres):
        depth = self.f_b/(disparity + 1e-9)
        depth_gt = self.f_b/(disparity_gt + 1e-9)
        inlier_idx = np.where((np.abs(disparity_gt - disparity) < err_thres) * (disparity_gt > 0))
        all_idx = np.where((disparity_gt > 0) * (disparity_gt > 0))
        gt_idx = np.where(disparity_gt > 0)

        if not method in self.eval_data.keys():
            self.eval_data[method] = dict()

        if len(self.eval_data[method].keys()) == 0:
            self.eval_data[method]['disparity'] = disparity[inlier_idx]
            self.eval_data[method]['disparity_all'] = disparity[all_idx]
            self.eval_data[method]['depth'] = depth[inlier_idx]         
            self.eval_data[method]['gt'] = dict()
        else:
            self.eval_data[method]['disparity'] = np.append(self.eval_data[method]['disparity'], disparity[inlier_idx])
            self.eval_data[method]['disparity_all'] = np.append(self.eval_data[method]['disparity_all'], disparity[all_idx])
            self.eval_data[method]['depth'] = np.append(self.eval_data[method]['depth'], depth[inlier_idx])
        
        if len(self.eval_data[method]['gt'].keys()) == 0:
            self.eval_data[method]['gt']['disparity'] = disparity_gt[inlier_idx]
            self.eval_data[method]['gt']['disparity_all'] = disparity_gt[all_idx]
            self.eval_data[method]['gt']['depth'] = depth_gt[inlier_idx]
            self.eval_data[method]['gt']['depth_valid_num'] = depth_gt[gt_idx].shape[0]

        else:
            self.eval_data[method]['gt']['disparity'] = np.append(self.eval_data[method]['gt']['disparity'], disparity_gt[inlier_idx])
            self.eval_data[method]['gt']['disparity_all'] = np.append(self.eval_data[method]['gt']['disparity_all'], disparity_gt[all_idx])
            self.eval_data[method]['gt']['depth'] = np.append(self.eval_data[method]['gt']['depth'], depth_gt[inlier_idx])
            self.eval_data[method]['gt']['depth_valid_num'] += depth_gt[gt_idx].shape[0]

    def metric_print(self, method):
        try:
            d_data = self.eval_data[method]['disparity']
            d_data_all = self.eval_data[method]['disparity_all'] 
            z_data = self.eval_data[method]['depth']
            d_gt = self.eval_data[method]['gt']['disparity']
            d_gt_all = self.eval_data[method]['gt']['disparity_all'] 
            z_gt = self.eval_data[method]['gt']['depth']
            valid_gt = self.eval_data[method]['gt']['depth_valid_num']

            error = dict()
            error['RMSE'] = RMSE(d_data, d_gt)
            error['MAE'] = MAE(d_data, d_gt)
            error['DTdelta'] = DTdelta(d_data, d_gt, valid_gt)
            error['RTdelta'] = RTdelta(d_data_all, d_gt_all)
            error['depth_RMSE'] = RMSE(z_data, z_gt)
            error['depth_ARD'] = ARD(z_data, z_gt)
            error['depth_ATdelta'] = ATdelta(z_data, z_gt, valid_gt)        

            print('Method: {0}\nEval data num: {2}\nErros: {1}\n'.format(method, error, len(self.eval_data[method]['gt']['depth'])))
        except:
            print('Method {0} print error occured'.format(method))

    def logger(self, method, time):
        if not method in self.log_data.keys():
            self.log_data[method] = np.zeros((10000, 1))
        self.log_data[method][self.iter] += time

    def time_print(self):
        if self.is_time_estimation_mode:
            avg = dict()
            for key in self.log_data.keys():
                log = self.log_data[key][np.where(self.log_data[key]!=0)[0]]
                avg[key] = np.mean(log)
                if not (key == 'the number of event'):
                    avg[key] *= (1e3 / self.repeat)

            print(avg)
