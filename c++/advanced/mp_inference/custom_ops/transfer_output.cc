#include <iostream>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

#include <stdlib.h>
#include <stdio.h>
#include <dlfcn.h>  // dladdr
#include <sys/time.h>
#include <sys/stat.h>
#include "paddle/extension.h"

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "token_transfer.hpp"

void PrintVec(std::vector<int64_t> & vec) {
  // return;
  std::cout << "000 vec_size: " << vec.size();
  for(int i {0}; i < vec.size(); i++) {
    std::cout << " " << vec[i];
  }
  std::cout << std::endl;
}

void PrintVec(int64_t *arr) {
  std::cout << "READ vec_size: " << arr[0];
  for(int i {1}; i < arr[0] + 1; i++) {
    std::cout << " " << arr[i];
  }
  std::cout << std::endl;
}

void PrintVec(int64_t bs, int64_t *arr) {
  std::cout << "WRITE vec_size: " << bs;
  for(int i {0}; i < bs; i++) {
    std::cout << " " << arr[i];
  }
  std::cout << std::endl;
}

std::vector<paddle::Tensor> TransferOutput(const paddle::Tensor& x,
                                           int64_t rank_id) {
    using namespace paddle::inference::transfer;

    auto x_cpu = x.copy_to(paddle::CPUPlace(), false);
    VLOG(3) << "rank_id: " << rank_id;
    if (rank_id != 0) {
      return {x_cpu};
    }
    std::vector<int64_t> x_shape = x_cpu.shape();
    int64_t token_num = x_cpu.numel();
    // Currently only support int64_t
    assert(x_cpu.type() == paddle::DataType::INT64);
    TokenTransfer::Instance().PushBatchToken(token_num, x_cpu.data<int64_t>());
    VLOG(3) << "rank_id: " << rank_id << ". place_id " << int(x.place().GetDeviceId());
    // For Test
    // int64_t get_token_test[20];
    // TokenTransfer::Instance().GetBatchToken(get_token_test);
    // PrintVec(get_token_test);
    
    return {x_cpu};                                                     
}

std::vector<std::vector<int64_t>> TransferOutputInferShape(const std::vector<int64_t>& x_shape){
    return {x_shape};
}

std::vector<paddle::DataType> TransferOutputInferDtype(const paddle::DataType& x_dtype) {
    return {x_dtype};
}

PD_BUILD_OP(transfer_output)
    .Inputs({"x"})
    .Attrs({"rank_id: int64_t"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(TransferOutput))
    .SetInferShapeFn(PD_INFER_SHAPE(TransferOutputInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(TransferOutputInferDtype));
