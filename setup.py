# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Package metadata
NAME = "samurai-real-time"
VERSION = "1.0"
DESCRIPTION = "Real-time adaptation of SAMURAI: Segment Anything in Images and Videos"
URL = "https://github.com/InesSousa11/samurai-real-time"
AUTHOR = "Inês Sousa"

# Read the contents of README file
with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()

# Required dependencies
REQUIRED_PACKAGES = [
    "decord>=0.6.0",
    "huggingface_hub>=0.34.3",
    "hydra-core>=1.3.2",
    "imageio>=2.37.0",
    "iopath>=0.1.10",
    "loguru>=0.7.3",
    "matplotlib>=3.10.3",
    "numpy>=1.24.0",    
    "omegaconf>=2.3.0",
    "opencv-python>=4.7.0",
    "pillow>=9.4.0",
    "pycocotools>=2.0.10",
    "scipy>=1.10.0",
    "setuptools>=65.5.0",
    "torch>=2.3.1",
    "torchvision>=0.18.1",
    "tqdm>=4.66.1",
    "ultralytics>=8.3.0"
]

def get_extensions():
    srcs = ["sam2/csrc/connected_components.cu"]
    compile_args = {
        "cxx": [],
        "nvcc": [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ],
    }
    ext_modules = [CUDAExtension("sam2._C", srcs, extra_compile_args=compile_args)]
    return ext_modules


# Setup configuration
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    author=AUTHOR,
    packages=find_packages(exclude="notebooks"),
    include_package_data=True,
    install_requires=REQUIRED_PACKAGES,
    python_requires=">=3.10.0",
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
