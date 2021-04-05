# Current version: v0.0.1

#FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04
FROM nvcr.io/nvidia/tensorrt:20.12-py3

# TensorRT: 7.2.2 (required for ONNX To TensorRT)
# Ubuntu 20.04
# Note: Container image 20.12-py3 contains Python 3.8.
# NVIDIA CUDA 11.1.1 including cuBLAS 11.3.0.
# NVIDIA cuDNN 8.0.5
# NVIDIA NCCL 2.8.3 (optimized for NVLink™ )
# Note: Although NCCL is packaged in the container, it does not affect TensorRT nor inferencing in any way.
# MLNX_OFED 5.1
# OpenMPI 4.0.5
# Nsight Compute 2020.2.1.8
# Nsight Systems 2020.3.4.32
# Source: https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/rel_20-12.html#rel_20-12

ENV CUDA_HOME=/usr/local/cuda
ENV CUDACXX=/usr/local/cuda/bin/nvcc
ENV CUDNN_HOME=/usr/local/cuda

##################################
# Basics                         #
##################################

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    libsqlite3-dev \
    libbz2-dev \
    liblzma-dev \
    zlib1g-dev \
    wget && \
    rm -rf /var/lib/apt/lists/*


##################################
# GCC                            #
##################################

# Only env vars are required for this image
ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++


###################################
# CMake                           #
###################################

ARG CMAKE_VERS=3.19.1
RUN apt-get update && \
    mkdir -p build && cd build && \
    wget https://github.com/Kitware/CMake/releases/download/v$CMAKE_VERS/cmake-$CMAKE_VERS.tar.gz && \
    tar -xvzf cmake-$CMAKE_VERS.tar.gz && \
    cd cmake-$CMAKE_VERS && \
    ./bootstrap && \
    make && \
    make install

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:/code/cmake-$CMAKE_VERS-Linux-x86_64/bin:${PATH}


#####################################
# Minor additions and packages      #
#####################################

RUN apt-get update && \
    pip install -U pip && \
    pip install -U setuptools

RUN apt-get update && \
    apt-get install -y language-pack-en && \
    locale-gen en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8

RUN apt-get update && \
    apt-get install -y protobuf-compiler libprotoc-dev && \
    pip install -U pytest pytest-cov protobuf numpy==1.19.5

#####################################
# Compile ONNX Runtime and Install  #
#####################################

ENV ONNX_ML=1
RUN pip install -U onnx

ARG ONNXRUNTIME_VERS=1.7.1
RUN apt-get update && \
    git clone https://github.com/microsoft/onnxruntime.git --branch v$ONNXRUNTIME_VERS --single-branch && \
    /bin/sh onnxruntime/dockerfiles/scripts/install_common_deps.sh && \
    cd onnxruntime/ && \
    pip install -r requirements-dev.txt && \
    /bin/sh ./build.sh \
    --config Release \
    --update \
    --build \
    --build_wheel \
    --parallel \
    --use_cuda \
    --cuda_home $CUDA_HOME \
    --cudnn_home /usr/lib/x86_64-linux-gnu/ \
    --use_tensorrt \
    --tensorrt_home /opt/tensorrt/ \
    --build_dir /workspace/onnxruntime/python

RUN pip install onnxruntime-tools git+https://github.com/microsoft/ort-customops.git

RUN pip uninstall -y onnxruntime && \
    pip install -U /workspace/onnxruntime/python/Release/dist/*gpu*.whl

RUN echo "ONNXRuntime Execution Providers: " && python -c "import onnxruntime as ort; print(ort.get_available_providers())"

WORKDIR /home
