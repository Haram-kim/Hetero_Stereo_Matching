"""
edit: Haram Kim
email: rlgkfka614@gmail.com
github: https://github.com/haram-kim
homepage: https://haram-kim.github.io/
"""

import numpy as np
import warnings

warnings.filterwarnings("ignore")
def logRMSE(data, gt):
    mask = (data > 0) & (gt > 0) & np.isfinite(data) & np.isfinite(gt)
    data = data[mask]
    gt = gt[mask]
    log_RMSE = np.sqrt(np.sum((np.log(data)-np.log(gt))**2)/np.log(10)**2/data.shape[0])
    return log_RMSE

def log10err(data, gt):
    mask = (data > 0) & (gt > 0) & np.isfinite(data) & np.isfinite(gt)
    data = data[mask]
    gt = gt[mask]
    log10err = np.sum(np.abs(np.log(data)-np.log(gt)))/np.log(10)/data.shape[0]
    return log10err

def ARD(data, gt):
    mask = (data >= 0) & (gt >= 0) & np.isfinite(data) & np.isfinite(gt)
    data = data[mask]
    gt = gt[mask]
    ARD = np.sum(np.abs(data - gt)/gt)/data.shape[0]
    return ARD

def ATdelta(data, gt, gt_valid_num):
    mask = (data > 0) & (gt > 0) & np.isfinite(data) & np.isfinite(gt)
    data = data[mask]
    gt = gt[mask]
    delta = np.array([1.05, 1.05**2, 1.05**3])
    ATdelta = np.zeros_like(delta)
    if data.shape[0] == 0:
        return ATdelta
    else:
        for i, d in enumerate(delta):
            ATdelta[i] = np.sum(np.max([data/gt, gt/data], axis=0) < d) / gt_valid_num
        return ATdelta

def DTdelta(data, gt, gt_valid_num):
    mask = (data > 0) & (gt > 0) & np.isfinite(data) & np.isfinite(gt)
    data = data[mask]
    gt = gt[mask]
    delta = np.array([1, 2, 3])
    DTdelta = np.zeros_like(delta, dtype = np.float32)
    for i, d in enumerate(delta):
        DTdelta[i] = np.sum(np.abs([data - gt]) < d) / gt_valid_num
    return DTdelta

def RTdelta(data, gt):
    mask = (data > 0) & (gt > 0) & np.isfinite(data) & np.isfinite(gt)
    data = data[mask]
    gt = gt[mask]
    delta = np.array([1, 2, 3])
    RTdelta = np.zeros_like(delta, dtype = np.float32)
    for i, d in enumerate(delta):
        RTdelta[i] = np.sum(np.abs([data - gt]) < d) / data.shape[0]
    return RTdelta

def MAE(data, gt):
    mask = (data >= 0) & (gt >= 0) & np.isfinite(data) & np.isfinite(gt)
    data = data[mask]
    gt = gt[mask]
    MAE = np.sum(np.abs(data - gt))/data.shape[0]
    return MAE

def RMSE(data, gt):
    mask = (data >= 0) & (gt >= 0) & np.isfinite(data) & np.isfinite(gt)
    data = data[mask]
    gt = gt[mask]
    RMSE = np.sqrt(np.sum(np.abs(data - gt)**2)/data.shape[0])
    return RMSE