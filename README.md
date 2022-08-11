Hetero Stereo Matching
=======================

The source code is released under the **MIT License**.
The code works best on Ubuntu 20.04

# Prerequisites
## OpenCV
We use [OpenCV](http://opencv.org) to visualize and manipulate images.

## CUDA
We use [CUDA](https://developer.nvidia.com/cuda-toolkit) for parallel computation.
[Download](https://developer.nvidia.com/cuda-downloads) [Installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

### pyCUDA for python
We use python wrapper [pyCUDA](https://documen.tician.de/pycuda/) for CUDA.

## Others
You can install dependent libararies by running :
```
pip install opencv pycuda pyyaml scipy tqdm h5py hdf5plugin
```

# Installation % Running guide

0. Download DSEC data set
[DSEC](https://dsec.ifi.uzh.ch/dsec-datasets/download/)

The directory structure should be :
```
/DSEC
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
python main.py $data path$ $data sequence name$
```
Example) python main.py /c/DSEC/ interlaken_00_c


# Configuration settings

## config/config.yaml

### Feature tracker
```
feature num: 1000  
track_err_thres: 10
track_win_size:
- 21
- 21
extract_win_size: 12
```

### Disparity estimator
```
disparity_range: 100
kernel_radius: 12
ncc_gaussian_std: 2
msd_gap: 10
```

### Time estimation mode
```
time_estimation: True
```

### Visualize
#### show disparity inlier requires ground truth disparity
```
show_disparity_inlier: True
show_disparity: True
semi_dense: True
```


