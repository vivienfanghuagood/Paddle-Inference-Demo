#!/bin/bash
set +x
set -e

work_path=$(dirname $(readlink -f $0))

# 2. compile
rm -rf build
mkdir -p build
cd build

DEMO_NAME=cpp_infer

WITH_MKL=ON
WITH_GPU=ON
USE_TENSORRT=OFF
WITH_ONNXRUNTIME=OFF

# LIB_DIR=/root/paddlejob/workspace/output/fhq_0324/nccl_multithread/Paddle/build/paddle_inference_install_dir
LIB_DIR=/root/paddlejob/workspace/output/fhq_0324/nccl_multithread/bf16/Paddle/build/paddle_inference_install_dir
CUDNN_LIB=/usr/lib/x86_64-linux-gnu/
CUDA_LIB=/usr/local/cuda/lib64
CUSTOM_OPERATOR_FILES="save_with_output.cc;token_penalty_multi_scores.cu;topp_sampling.cu"


cmake .. -DPADDLE_LIB=${LIB_DIR} \
  -DWITH_MKL=${WITH_MKL} \
  -DDEMO_NAME=${DEMO_NAME} \
  -DWITH_GPU=${WITH_GPU} \
  -DWITH_STATIC_LIB=OFF \
  -DUSE_TENSORRT=${USE_TENSORRT} \
  -DCUDNN_LIB=${CUDNN_LIB} \
  -DCUDA_LIB=${CUDA_LIB} \
  -DCUSTOM_OPERATOR_FILES=${CUSTOM_OPERATOR_FILES} \
  -DWITH_ONNXRUNTIME=${WITH_ONNXRUNTIME}

make -j
