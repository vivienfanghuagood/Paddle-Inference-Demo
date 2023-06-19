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

#include "token_transfer.hpp"

std::vector<paddle::Tensor> TransferOutput(const paddle::Tensor& x,
                                           int64_t rank_id) {
    using namespace paddle::inference::transfer;

    auto x_cpu = x.copy_to(paddle::CPUPlace(), false);
    if (rank_id != 0) {
      return {x_cpu};
    }
    std::vector<int64_t> x_shape = x_cpu.shape();
    int64_t token_num = x_cpu.numel();
    // Currently only support int64_t
    assert(x_cpu.type() == paddle::DataType::INT64);
    TokenTransfer::Instance().PushBatchToken(token_num, x_cpu.data<int64_t>());
    
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
