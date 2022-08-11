Hetero Stereo Matching
=======================

The source code is released under the **MIT License**.

# Project Page
You can see the high-quality video at:  
[https://haram-kim.github.io/Hetero_Stereo_Matching/](https://haram-kim.github.io/Hetero_Stereo_Matching/)

# Prerequisites
## Python
We tested our code on [Python](https://www.python.org/) version 3.8.0   
[[Download]](https://www.python.org/downloads/release/python-380/)

## OpenCV
We use [OpenCV](http://opencv.org) to visualize and manipulate images.

## CUDA
We use [CUDA](https://developer.nvidia.com/cuda-toolkit) for parallel computation.  
[[Download]](https://developer.nvidia.com/cuda-downloads) [[Installation guide]](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

### pyCUDA
We use python wrapper [pyCUDA](https://documen.tician.de/pycuda/) for CUDA.

## Others
You can install dependent libararies by running :
```
pip install opencv pycuda pyyaml scipy tqdm h5py hdf5plugin
```

# Installation & Running guide

The code works best on Ubuntu 20.04  

0. Download DSEC data set  
DSEC [[Download]](https://dsec.ifi.uzh.ch/dsec-datasets/download/)  

    0-1. (Optional) If you want to download only the preprocessed data, please download:  
['interlaken_00_c_processed_data'](https://larr.snu.ac.kr/haramkim/DSEC/interlaken_00_c_processed_data.hdf5) (8.71GB)  
['interlaken_00_d_processed_data'](https://larr.snu.ac.kr/haramkim/DSEC/interlaken_00_d_processed_data.hdf5) (34.4GB)  
['interlaken_00_e_processed_data'](https://larr.snu.ac.kr/haramkim/DSEC/interlaken_00_e_processed_data.hdf5) (26.9GB)  
['interlaken_00_f_processed_data'](https://larr.snu.ac.kr/haramkim/DSEC/interlaken_00_f_processed_data.hdf5) (14.6GB)  
['interlaken_00_g_processed_data'](https://larr.snu.ac.kr/haramkim/DSEC/interlaken_00_g_processed_data.hdf5) (14.2GB)  
(Still, '[DATA_SEQUENCE]_calibration' must be download.)  

Directory structure :
```
/PATH_TO_DSEC
├────interlaken_00_c
│    ├────interlaken_00_c_calibration
│    ├────interlaken_00_c_disparity_event
│    ├────interlaken_00_c_disparity_image
│    ├────interlaken_00_c_events_left
│    ├────interlaken_00_c_events_right
│    ├────interlaken_00_c_images_rectified_left
│    ├────interlaken_00_c_images_rectified_right
│    ├────interlaken_00_c_disparity_timestamps.txt
│    └────interlaken_00_c_image_timestamps.txt
│
├────interlaken_00_c_processed_data.hdf5 (Please locate the preprocessed data to this.)

...


└────interlaken_00_g
     ├────interlaken_00_c_calibration
     
     ...
     
     └────interlaken_00_c_image_timestamps.txt
```

1. Clone this repository:
```
$ git clone https://github.com/Haram-kim/Hetero_Stereo_Matching.git
```

2. Run the code
```
$ cd HSM
$ python main.py [PATH_TO_DSEC] [DATA_SEQUENCE]
Example)  $ python main.py /c/DSEC/ interlaken_00_c
```


## Configuration settings

### config/config.yaml

#### Feature tracker
feature num - the number of features  
track_err_thres - KLT traker error threshold  
track_win_size - KLT tracker window size  
extract_win_size - Feature extractor window size  
```
feature num: 1000  
track_err_thres: 10
track_win_size:
- 21
- 21
extract_win_size: 12
```

#### Disparity estimator
disparity_range - disparity range (0 - max_disp)  
kernel_radius - patch radius  
ncc_gaussian_std - standard deviation of Gaussian filter  
msd_gap - Maximum Shift Distance interval  
```
disparity_range: 100
kernel_radius: 12
ncc_gaussian_std: 2
msd_gap: 10
```

#### Time estimation mode
If you want to estimate the computation time, you can run the time_estimation mode.  
```
time_estimation: True
```

#### Visualize
'show disparity inlier' requires ground truth disparity.  
If you want to see all estimated disparity of the proposed method, please change the flag 'show_disparity' False to True  
If you want to see estimated disparity on edges of the proposed method, please change the flag 'semi_dense'  False to True  
```
show_disparity_inlier: True
show_disparity: True
semi_dense: True
```


