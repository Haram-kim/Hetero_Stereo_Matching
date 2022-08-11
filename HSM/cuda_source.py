import pycuda.driver as cuda
import pycuda.tools
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import string
import numpy as np

CUDA_SOURCE_FILE_PATH = "cuda/source.cu"

cuda_source_file = open(CUDA_SOURCE_FILE_PATH, "r")
template = string.Template(cuda_source_file.read())
cuda_source_file.close()
