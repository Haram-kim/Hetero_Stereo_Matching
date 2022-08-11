"""
edit: Haram Kim
email: rlgkfka614@gmail.com
github: https://github.com/haram-kim
homepage: https://haram-kim.github.io/
"""
import cv2
import glob
import h5py
import hdf5plugin
import yaml
import numpy as np
from numpy import linalg as la
from tqdm import tqdm
from tqdm import trange

from Camera import Camera, CameraDSEC

class DataLoaderDSEC:
    is_initialized = False
    # events.hdf5: 'events', 'ms_to_idx', 't_offset'
    def __init__(self, dir_path, data_seq):
        
        self.calib_path = ''.join([dir_path, data_seq, '/', data_seq, '_calibration/cam_to_cam.yaml'])
        self.disparity_image_paths = glob.glob(''.join([dir_path, data_seq, '/', data_seq, '_disparity_image/*.png']))
        self.disparity_event_paths = glob.glob(''.join([dir_path, data_seq, '/', data_seq, '_disparity_event/*.png']))
        self.disparity_timestamp_path = ''.join([dir_path, data_seq, '/', data_seq, '_disparity_timestamps.txt'])
        self.events_left_path = ''.join([dir_path, data_seq, '/', data_seq, '_events_left/events.h5'])
        self.events_right_path = ''.join([dir_path, data_seq, '/', data_seq, '_events_right/events.h5'])
        self.images_left_paths = glob.glob(''.join([dir_path, data_seq, '/', data_seq, '_images_rectified_left/*.png']))
        self.images_right_paths = glob.glob(''.join([dir_path, data_seq, '/', data_seq, '_images_rectified_right/*.png']))
        self.image_timestamp_path = ''.join([dir_path, data_seq, '/', data_seq, '_image_timestamps.txt'])
        self.processed_data_path = ''.join([dir_path, data_seq, '_processed_data.hdf5'])
        self.E2VID_data_path = glob.glob(''.join([dir_path, data_seq, '/E2VID/*.png']))
        self.data_seq = data_seq
        self.proc_data = {}

        try:
            self.proc_data = h5py.File(self.processed_data_path, "r")
            self.is_data_loaded = True            
            print('Data loader initiated.')
        except:
            self.proc_data = h5py.File(self.processed_data_path, "w")
            self.is_data_loaded = False

        self.load_data()
        print("Completed loading [{0}] images and events from dataset [{1}].".format(self.image_num, self.data_seq))         

    
    def __del__(self):
        self.proc_data.close()

    def load_data(self):
        if not self.is_data_loaded:
            print("Start converting dataset into hdf file for faster loading later.")
            print("This process will only run for the first time.")
            self.load_images()
            self.load_events()            
            print("Complete saving dataset as hdf file.")
            self.proc_data = h5py.File(self.processed_data_path, "r")
            self.is_data_loaded = True
        self.images = self.proc_data['images']
        self.image_ts = self.proc_data['image_ts']
        self.events = self.proc_data['events'] 
        self.image_num = len(self.image_ts)        

    def load_images(self):
        if not (len(self.images_left_paths) == len(self.images_right_paths)):
            raise Exception("Image file length is not equal")

        image_ts = np.loadtxt(self.image_timestamp_path, dtype = 'int')/1e6
        self.image_num = image_num = len(image_ts)

        cam0_images = np.zeros(((image_num,)+ cv2.imread(self.images_left_paths[0]).shape), dtype=np.uint8) 
        print("Loading images")
        for i in trange(image_num):
            cam0_images[i, :, :, :] = cv2.imread(self.images_left_paths[i]).astype(np.uint8)
        self.image_ts = image_ts
        self.proc_data.create_dataset('images', data=cam0_images[:,:,:,:])
        self.proc_data.create_dataset('image_ts', data=self.image_ts)

    def load_events(self):
        cam1_events = h5py.File(self.events_right_path)['events']
        # # frame: left cam0
        # # event right cam1         
        events = cam1_events
        image_ts = np.loadtxt(self.image_timestamp_path, dtype = 'int')/1e6
        event_chunk_st = [None] * image_ts.shape[0]

        t_idx = 0
        image_ts = image_ts - image_ts[0]
        ts = image_ts[t_idx]
        print("Indexing events")
        for e_idx in trange(0,len(events['t']),5000):
            if(events['t'][e_idx]/1e6 >= ts):
                event_chunk_st[t_idx] = e_idx
                t_idx += 1            
                if (t_idx >= len(image_ts)):
                    break
                ts = image_ts[t_idx]

        event_x = np.split(events['x'], event_chunk_st[:-1], axis = 0)
        event_y = np.split(events['y'], event_chunk_st[:-1], axis = 0)
        event_t = np.split(events['t'], event_chunk_st[:-1], axis = 0)
        event_p = np.split(events['p'], event_chunk_st[:-1], axis = 0)

        if not event_chunk_st[0]:
            event_x[0] = np.array([0])
            event_y[0] = np.array([0])
            event_t[0] = np.array([0])
            event_p[0] = np.array([0])
            
        event_group = self.proc_data.create_group('events')
        print("Loading events")
        for idx in trange(len(event_t)):
            events = np.stack((event_x[idx], event_y[idx], event_t[idx], event_p[idx]), axis=1).astype(np.float32)
            events[:,2] /= 1e6
            event_group.create_dataset('{0}'.format(idx), data=events)


    def get_data(self, idx):
        image = cv2.cvtColor(self.images[idx], cv2.COLOR_RGB2GRAY)
        image_ts = self.image_ts[idx]
        event = np.asarray(self.events['{0}'.format(idx)])
        return image, image_ts, event
        
    def get_debug_data(self, idx):
        # load right image
        try:
            cam1_image = cv2.cvtColor(cv2.imread(self.images_right_paths[idx]).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            cam1_image = self.cam0.rectify(cam1_image)
        except:
            cam1_image = np.zeros_like(self.cam0.resolution)
        # load E2VID image
        try:
            cam1_image_E2VID = cv2.cvtColor(cv2.imread(self.E2VID_data_path[idx]).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            cam1_image_E2VID = self.cam1.rectify(cam1_image_E2VID)
        except:
            cam1_image_E2VID = np.zeros_like(self.cam0.resolution)
        # load disparity
        try:
            image_disparity = cv2.cvtColor(cv2.imread(self.disparity_image_paths[np.int(idx/2)]),
                                            cv2.COLOR_RGB2GRAY).astype(np.uint16)  
            image_disparity = self.cam0.rectify(image_disparity, cv2.INTER_NEAREST)*(
                (self.cam0.K_rect[0][0]+self.cam0.K_rect[1][1])/(self.cam0.K[0][0]+self.cam0.K[1][1])
                *self.cam0.baseline/self.cam0.baseline_gt)
            success = True
        except:
            image_disparity = np.zeros_like(self.cam0.resolution)
            success = False
             
        return success, cam1_image, image_disparity, cam1_image_E2VID

    def load_calib_data(self):
        calib_data = []
        with open(self.calib_path, "r") as stream:
            try:
                calib_data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        rect_mat = np.eye(3)
        T_32 = np.asarray(calib_data['extrinsics']['T_32'])
        T_21 = np.asarray(calib_data['extrinsics']['T_21'])
        T_rect1 = np.eye(4)
        T_rect1[:3,:3] = np.asarray(calib_data['extrinsics']['R_rect1'])
        T_rect31 = np.matmul(np.matmul(np.linalg.inv(T_32), np.linalg.inv(T_21)), T_rect1)
        rect_mat = T_rect31[:3, :3]
        baseline = la.norm(T_rect31[:3, 3])

        rect_mat = np.eye(3)
        T_32 = np.asarray(calib_data['extrinsics']['T_32'])
        T_rect2 = np.eye(4)
        T_rect2[:3,:3] = np.asarray(calib_data['extrinsics']['R_rect2'])
        T_rect32 = np.matmul(np.linalg.inv(T_32), T_rect2)
        rect_mat = T_rect32[:3, :3]
        baseline = la.norm(T_rect31[:3, 3])
        baseline_gt = la.norm(T_21[:3, 3])

        # rect_mat = np.eye(3)

        CameraDSEC.set_baseline(baseline, baseline_gt)
        CameraDSEC.set_cam_rect(calib_data['intrinsics']['camRect3'])
        self.cam0 = CameraDSEC(calib_data['intrinsics']['camRect1'])
        self.cam1 = CameraDSEC(calib_data['intrinsics']['cam3'], rect_mat)        

        return self.cam0, self.cam1

    def load_params(self):
        cam0, cam1 = self.load_calib_data()

        params = {}
        params["cam0"] = cam0
        params["cam1"] = cam1

        params['loader'] = self

        with open('config/config.yaml', "r") as stream:
            try:
                config_data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        params['feature num'] = config_data['feature num']
        params['disparity_range'] = config_data['disparity_range']

        params['kernel_radius'] = config_data['kernel_radius']
        params['ncc_gaussian_std'] = config_data['ncc_gaussian_std']

        params['track_win_size'] = config_data['track_win_size']
        params['extract_win_size'] = config_data['extract_win_size']
        params['track_err_thres'] = config_data['track_err_thres']

        params['time_estimation'] = config_data['time_estimation']
        params['show_disparity_inlier'] = config_data['show_disparity_inlier']
        params['show_disparity'] = config_data['show_disparity']
        params['semi_dense'] = config_data['semi_dense']

        params['msd_gap'] = config_data['msd_gap']

        return params