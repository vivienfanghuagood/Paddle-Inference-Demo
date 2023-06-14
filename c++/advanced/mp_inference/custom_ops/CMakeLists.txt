FILE(GLOB CUSTOM_OPERATOR_FILES *.cu *.cc)
# set(CUSTOM_OPERATOR_FILES *.cu)

project(pd_infer_custom_op)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr)
# nv_library(pd_infer_custom_op SRCS ${CUSTOM_OPERATOR_FILES} SHARED DEPS paddle_inference)
cuda_add_library(pd_infer_custom_op ${CUSTOM_OPERATOR_FILES} SHARED)