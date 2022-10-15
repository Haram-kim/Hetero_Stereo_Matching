"""
edit: Haram Kim
email: rlgkfka614@gmail.com
github: https://github.com/haram-kim
homepage: https://haram-kim.github.io/
"""

import numpy as np
import numpy.linalg as la
import copy
import cv2

from utils import *
from Camera import Camera, CameraDSEC

class Map:
    def __init__(self, id):
        self.id = id   
        self.pts = np.zeros((1,3))
        self.confidence = 0
        self.is_reconstructed = False

    def set_pts(self, pts):
        self.pts = pts
        self.is_reconstructed = True

class Feature:
    id = 0
    def __init__(self):
        self.clear()

    def set_uv(self, uv, id):
        self.uv = uv
        self.id = id
        self.is_extracted = True
        self.is_tracked = True

    def new_uv(self, uv):
        self.clear()
        self.uv = uv
        self.id = Feature.id        
        Feature.id += 1
        self.is_extracted = True
        self.is_tracked = False

    def clear(self):
        self.id = -1
        self.uv = np.zeros((1, 2), dtype=np.float32)
        
        self.is_tracked = False
        self.is_reconstructed = False
        self.is_extracted = False

class FeatureTracker:
    def __init__(self, params):
        self.cam0 = params['cam0']
        self.cam1 = params['cam1']
        self.feature_num = params['feature num']
        self.track_win_size = params['track_win_size']
        self.extract_win_size = params['extract_win_size']
        self.track_err_thres = params['track_err_thres']
        self.track_params = dict(winSize=self.track_win_size,
                            criteria=(cv2.TERM_CRITERIA_EPS |
                            cv2.TERM_CRITERIA_COUNT, 30, 0.03))

        self.features = [Feature() for i in range(self.feature_num)]
        self.feature_idx = []
        self.tracked_feature_idx = []
        self.empty_feature_idx = [i for i in range(self.feature_num)]          

        self.map_pts = {}
        self.image = np.zeros(Camera.resolution)

        self.iter = 0
        
        self.is_first_frame = True
        self.is_recon_init = False
        self.is_init = False

    def detect_feature(self, image):
        self.iter += 1   
        self.previmage = self.image
        self.prevfeatures = copy.deepcopy(self.features)
        self.prevfeature_idx = self.feature_idx

        image = image.astype(np.uint8)
        self.image = image
        feature_mask = np.ones(Camera.resolution) # TODO change to rect area
        if self.is_first_frame:            
            self.is_first_frame = False
        else:
            tracked_points = self.track_feature(image)

        # extract new features            
        for _, t_idx in enumerate(self.tracked_feature_idx):
            point = self.features[t_idx].uv.astype(np.uint8)
            cv2.circle(feature_mask, point, 5, 0, -1)
        if len(self.empty_feature_idx) > 1:
            new_points = cv2.goodFeaturesToTrack(image, len(self.empty_feature_idx), 0.01, 5, mask=feature_mask.astype(np.uint8), blockSize=self.extract_win_size, k=0.03).squeeze()

            for pts_idx, new_idx in enumerate(self.empty_feature_idx):
                if(pts_idx >= new_points.shape[0]):
                    self.features[new_idx].clear()
                else:
                    self.features[new_idx].new_uv(new_points[pts_idx, :])
                    id = self.features[new_idx].id
                    self.map_pts[id] = Map(id)

        self.feature_idx = [i for i in range(self.feature_num)
                if self.features[i].is_extracted]
        # update parameters

    def track_feature(self, image):
        points = np.array([self.prevfeatures[idx].uv for i, idx in enumerate(self.prevfeature_idx)])
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.previmage,
                                               image, points.astype(np.float32),
                                               None, **self.track_params)
                                    
        for i, idx in enumerate(self.prevfeature_idx):
            if (st[i] & (err[i] < self.track_err_thres) 
                & (p1[i, 0] < Camera.resolution[1]) & (p1[i, 0] >= 0)
                & (p1[i, 1] < Camera.resolution[0]) & (p1[i, 1] >= 0)):
                self.features[idx].set_uv(p1[i, :], self.prevfeatures[idx].id)
                if p1[i, 1] == 0:
                    print(1)
            else:
                self.features[idx].clear()

        self.tracked_feature_idx = [i for i in range(self.feature_num)
                                if self.features[i].is_tracked]
        self.empty_feature_idx = [i for i in range(self.feature_num)
                                if not self.features[i].is_tracked]
        tracked_points = [self.features[i].uv for i in range(self.feature_num)
                                if self.features[i].is_tracked]
        return tracked_points

    def drawFeatureTracks(self, image):
        draw_image = cv2.cvtColor(((self.previmage/2 + image/2)).astype(np.uint8), cv2.COLOR_GRAY2RGB)

        for _, t_idx in enumerate(self.tracked_feature_idx):
            a,b = self.features[t_idx].uv.astype(int).ravel()
            c,d = self.prevfeatures[t_idx].uv.astype(int).ravel()
            cv2.line(draw_image, (a,b),(c,d), (0,255,0), 1)
            cv2.circle(draw_image,(a,b),1, (0,0,255),-1)
            cv2.circle(draw_image,(c,d),1, (255,0,0),-1)

        return draw_image

    def drawMapPointReproj(self, T):
        map_pts, image_pts = self.get_map_points()
        map_pts_c = np.matmul(np.concatenate([map_pts, np.ones((len(map_pts), 1))], 1), T.T)
        map_pts_proj_c = np.matmul(np.divide(map_pts_c[:, :3], map_pts_c[:, 2].reshape(-1,1)), self.cam0.K_rect.T)

        draw_image = cv2.cvtColor(self.image.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        for i in range(len(map_pts)):
            a = map_pts_proj_c[i, 0].astype(np.int32)
            b = map_pts_proj_c[i, 1].astype(np.int32)
            c = image_pts[i, 0].astype(np.int32)
            d = image_pts[i, 1].astype(np.int32)
            cv2.line(draw_image, (a,b),(c,d), (0,255,0), 1)
            cv2.circle(draw_image,(a,b),1, (0,255,255),-1)
            cv2.circle(draw_image,(c,d),1, (255,0,0),-1)

    def drawMapPoint(self, T):
        T = np.eye(4)
        map_pts, image_pts = self.get_map_points()
        map_pts_c = np.matmul(np.concatenate([map_pts, np.ones((len(map_pts), 1))], 1), T.T)
        map_pts_proj_c = np.matmul(np.divide(map_pts_c[:, :3], map_pts_c[:, 2].reshape(-1,1)), self.cam0.K_rect.T)

        draw_image = cv2.cvtColor(np.zeros_like(self.image, dtype=np.uint8), cv2.COLOR_GRAY2RGB)

        for i in range(len(map_pts)):
            a = map_pts_proj_c[i, 0].astype(np.int32)
            b = map_pts_proj_c[i, 1].astype(np.int32)
            cv2.circle(draw_image,(a,b),1, (0,0,255),-1)        
        cv2.imshow("map points", draw_image)
        cv2.waitKey(1)        

    def map_init(self, depth):
        for _, t_idx in enumerate(self.feature_idx):
            u, v = self.features[t_idx].uv.astype(int).ravel()
            id = self.features[t_idx].id
            if (depth[v,u] > 1e-1) & (depth[v,u] < 1e2):
                pts = np.matmul(self.cam0.K_rect_inv, np.array([u, v, 1]))*depth[v,u]
                self.map_pts[id].set_pts(pts)
                self.features[t_idx].is_reconstructed = True
            else:
                self.features[t_idx].is_reconstructed = False
        self.is_recon_init = True

    def map_update(self, depth, T):
        self.map_init(depth)
        
    def compute_camera_pose(self):
        map_pts, image_pts = self.get_map_points()        
        success, r, t, inlier = cv2.solvePnPRansac(map_pts, image_pts, self.cam0.K_rect, np.zeros((5,1)))
        r = r.squeeze()
        t = t.squeeze()
        T = np.eye(4)
        T[:3, :3] = SO3(r)
        T[:3, 3] = t
        self.drawMapPointReproj(T)
        return T

    def get_map_points(self):
        # get map points
        map_id = [self.prevfeatures[idx].id for i, idx in enumerate(self.prevfeature_idx)
                    if self.prevfeatures[idx].is_reconstructed & self.features[idx].is_tracked]
        map_pts = np.zeros((len(map_id), 3))
        for i, id in enumerate(map_id):
            map_pts[i, :] = self.map_pts[id].pts
        # get image points
        image_idx = [idx for i, idx in enumerate(self.prevfeature_idx)
                    if self.prevfeatures[idx].is_reconstructed & self.features[idx].is_tracked]
        image_pts = np.zeros((len(image_idx), 2))
        for i, idx in enumerate(image_idx):
            image_pts[i, :] = self.features[idx].uv

        return map_pts, image_pts
