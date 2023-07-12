// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <atomic>
#include <chrono>
#include <ctime>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>
#include <map>
#include <sstream>
#include <fstream>
#include <string>
#include <thread>
#include <pthread.h>
#include <thread>
#include <sstream>
#include <stdexcept>

#include<sys/stat.h>
#include<unistd.h>
#include<fcntl.h>
#include<sys/mman.h>

#include <cuda_runtime.h>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include <sched.h> 
#include <stdlib.h>
// #include "paddle/include/experimental/phi/common/bfloat16.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/include/paddle_inference_api.h"
#include "paddle/include/paddle_tensor.h"
// #include "paddle/include/experimental/phi/common/float16.h"

#include <stdlib.h>
#include <dlfcn.h>

#include "custom_ops/token_transfer.hpp"
#include "cnpy.h"
#include "inference_helper.hpp"

using paddle_infer::Config;
using paddle_infer::CreatePredictor;
using paddle_infer::Predictor;
using bfloat16 = phi::dtype::bfloat16;
using paddle_infer::PlaceType;
// using phi::dtype::float16;

DEFINE_bool(use_multi_thread_inference, true, "whether use multi thread inference");
DEFINE_string(dump_input, "./dump_input_who.npz", "Directory of the dumped input npz file");
DEFINE_string(model_file, "", "Directory of the inference model.");
DEFINE_string(params_file, "", "Directory of the inference model.");
DEFINE_string(model_dir, "", "Directory of the inference model.");
DEFINE_int32(max_batch, 1, "max_batch");
DEFINE_int32(max_seq_len, 4096, "max_sequence_length");
DEFINE_int32(layer_num, 80, "number layers");
DEFINE_int32(ring_id, 0, "ring id");
DEFINE_int32(dist_nrank, 1, "dist_nrank");
DEFINE_int32(benchmark_time, 1, "the times of inference[used for benchmark only]");

std::atomic<int> finish_flag = {0};
// temp test
void PrintVec(int64_t *arr) {
  VLOG(0) << "READ vec_size: " << arr[0];
  std::string ss{};
  for(int i {1}; i < arr[0] + 1; i++) {
    ss += std::to_string(arr[i]) + ", ";
    // VLOG(0) << " " << arr[i];
  }
  VLOG(0) << ss;
  // std::cout << std::endl;
}

struct inference_attr {
    int rank;
    int device_id;
    int32_t thread_idx;
    std::string model_file;
    std::string params_file;
    int* ready_count;
    cudaStream_t cuda_stream{nullptr};
    cnpy::npz_t input_npz;
    std::unordered_map<std::string, void*> shared_buffer_map;
};

std::shared_ptr<Predictor> InitPredictor(std::string model_path, std::string params_path, int device_id=0, cudaStream_t stream=nullptr) {
  Config config;

  config.SetModel(model_path, params_path);
  // int device_id = rank >= 0? rank : 0;
  config.EnableUseGpu(50, device_id);
  // config.EnableMemoryOptim();
  config.SwitchIrOptim(true);
  // config.EnableProfile();
  config.SetCpuMathLibraryNumThreads(1);
  if (device_id == 0){
    VLOG(0) << "[multi_thread] " << "device_id 0 ";
    config.SetUseMultiThreadInference(true);
    config.SetMultiThreadRingId(FLAGS_ring_id);
    std::vector<int> dev_ids = {0,1,2,3,4,5,6,7};
    auto select_dev_ids = std::vector<int>(dev_ids.begin(), dev_ids.begin() + FLAGS_dist_nrank);
    config.SetMultiThreadInferenceAllRanks(select_dev_ids);
  }
  if (stream != nullptr) {
    VLOG(3) << "[multi_thread] " << "Set external stream: " << stream;
    config.SetExecStream(stream);
    CHECK_EQ(config.external_stream_enabled(), true);
  }
  config.pass_builder()->DeletePass("fc_fuse_pass");
  // VLOG(3) << "Thread "<< rank <<"  Used passes: " << config.pass_builder()->DebugString();
  VLOG(3) << config.Summary();
  return CreatePredictor(config);
}

void run_predict(Predictor *predictor, cnpy::npz_t& input_npz, int rank=-1, inference_attr* args=nullptr) {
  // cnpy::npz_t input_npz = cnpy::npz_load(FLAGS_dump_input);
  auto input_names = predictor->GetInputNames();
  auto input_types = predictor->GetInputTypes();
  for (auto name : input_names) {
    auto input_tmp = predictor->GetInputHandle(name);
    auto input_tmp_shape = input_tmp->shape();
    cnpy::NpyArray arr_npy = input_npz[name];

    std::vector<int> shape;
    int numel {1};
    for(size_t j = 0; j < arr_npy.shape.size(); ++j){
      numel *= arr_npy.shape[j];
      shape.emplace_back(arr_npy.shape[j]);
      VLOG(0) <<  "[check shape]: " <<  arr_npy.shape[j];
    }
    //numel = numel * FLAGS_max_batch / shape[0];
    //shape[0] = FLAGS_max_batch;
    
    VLOG(0) <<  "[check input] input names:" <<  name;

    if (name.find("cache_kv") != std::string::npos) {
       VLOG(3) <<  "[check input] load cache" <<  name;
      // 2 * FLAGS_max_batch * FLAGS_max_seq_len * 64 / FLAGS_dist_nrank * 128
      auto cache_buffer =  static_cast<bfloat16*>(args->shared_buffer_map[name]);
      std::vector<int> spec_shape {2, FLAGS_max_batch, 64 / FLAGS_dist_nrank, FLAGS_max_seq_len, 128};
      input_tmp->ShareExternalData<bfloat16>(cache_buffer, spec_shape, PlaceType::kGPU);
      continue;
    } 
    else if (name.find("position_ids") != std::string::npos) {
      int64_t* position_id_buffer = nullptr;
      cudaMalloc(&position_id_buffer, numel * sizeof(int64_t));
      cudaMemcpy(position_id_buffer, arr_npy.data<int64_t>(), numel * sizeof(int64_t), cudaMemcpyHostToDevice);
      input_tmp->ShareExternalData<int64_t>(position_id_buffer, shape, PlaceType::kGPU);
      continue;
    } 
    else if (name.find("mask") != std::string::npos) {
      bfloat16* mask_buffer = nullptr;
      cudaMalloc(&mask_buffer, numel * sizeof(bfloat16));
      cudaMemcpy(mask_buffer, arr_npy.data<bfloat16>(), numel * sizeof(bfloat16), cudaMemcpyHostToDevice);
      input_tmp->ShareExternalData<bfloat16>(mask_buffer, shape, PlaceType::kGPU);
      continue;
    }
    else if (name.find("pre_id") != std::string::npos) {
      long* pre_id_buffer = nullptr;
      long pre_count = FLAGS_max_batch * FLAGS_max_seq_len;
      cudaMalloc((void **) &pre_id_buffer, pre_count * sizeof(long));

      long* pre_id_buffer_cpu = new long[pre_count];
      for (int jj = 0; jj < pre_count; ++jj){
        pre_id_buffer_cpu[jj] = -1;
      }
      cudaMemcpy(pre_id_buffer, pre_id_buffer_cpu, pre_count, cudaMemcpyHostToDevice);
      delete pre_id_buffer_cpu;

      std::vector<int> spec_shape {FLAGS_max_batch, FLAGS_max_seq_len};
      // input_tmp->Reshape(shape);
      input_tmp->ShareExternalData<long>(pre_id_buffer, spec_shape, PlaceType::kGPU);
      continue;
    } 
    

    VLOG(0) <<  "[check dtype] : " << input_types[name];
    input_tmp->Reshape(shape);

    if (input_types[name] == paddle_infer::DataType::FLOAT32){
      float* data = arr_npy.data<float>();
      input_tmp->CopyFromCpu(data);
    } else if (input_types[name] == paddle_infer::DataType::BFLOAT16){
      bfloat16* data = arr_npy.data<bfloat16>();
      VLOG(0) <<  "[sample out] : " << data[0] << ", " << data[1] << ", " << data[2] << ", " << data[3];
      input_tmp->CopyFromCpu(data);
    } else if (input_types[name] == paddle_infer::DataType::INT64){
      
      long* data = arr_npy.data<long>();
      input_tmp->CopyFromCpu(data);
    } else if (input_types[name] == paddle_infer::DataType::INT32){
      int* data = arr_npy.data<int>();
      input_tmp->CopyFromCpu(data);
    } else if (input_types[name] == paddle_infer::DataType::BOOL) {
      bool *data = arr_npy.data<bool>();
      input_tmp->CopyFromCpu(data);
    } else {
      std::stringstream error_info;
      error_info << "Unsupported data type " << input_types[name] << " when get input dtype";
      throw(std::invalid_argument(error_info.str()));
    }
  }

  for (int i = 0; i < FLAGS_benchmark_time; i++) {
    // predictor->ExpRunWithExternalStream(comm_0.stream());
    // if(i < 2){
    //   VLOG(1) << "start set value: ";
    //   setenv("NCCL_LAUNCH_MODE", "GROUP", 1);
    // }
    // else{
    //   VLOG(1) << "start set value: ";
    //   setenv("NCCL_LAUNCH_MODE", "PARALLEL", 1);
    // }

    // auto* envs = getenv("NCCL_LAUNCH_MODE");

    // VLOG(1) << "set value: " << envs;
    // char get_v = get_mmap_value();
    // 
    auto start = std::chrono::system_clock::now();
    CHECK(predictor->Run());
    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    VLOG(0) << "[BENCHMARK] ellapse " << elapsed.count() << "ms \n";
  }
  VLOG(3) << "check run predictor finished";

  if (rank == 0) {
    using namespace paddle::inference::transfer;
    VLOG(0) << "start print res";

    int64_t get_token_test[20];
    while (TokenTransfer::Instance().GetBatchToken(get_token_test)) {
      PrintVec(get_token_test);
    }
    // TokenTransfer::Instance().GetBatchToken(get_token_test);
    // PrintVec(get_token_test);
    VLOG(0) << "end print res";
  }
  // if(rank == 0){
  //   auto output_names = predictor->GetOutputNames();
  //   auto output_t = predictor->GetOutputHandle(output_names[3]);
  //   std::vector<int> output_shape = output_t->shape();
  //   size_t shape_product = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
  //   std::vector<int>out_data(shape_product);
  //   output_t->CopyToCpu(out_data.data());
  //   std::cout << "print out\n";
  //   for(size_t i=0; i< shape_product; ++i){
  //       std::cout << out_data[i] << ", ";
  //   }
  //   std::cout << std::endl;
  // }

  return;
}

void *thread_fn(void *args) {
  

    struct inference_attr *targs = (struct inference_attr *)args;

    cpu_set_t cpuset; 
    int cpu = targs->device_id;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu , &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);

    // int rank = targs->rank;
    int device_id = targs->device_id;
    int32_t thread_idx = targs->thread_idx;
    std::string model_file = targs->model_file;
    std::string params_file = targs->params_file;
    cudaStream_t cuda_stream = targs->cuda_stream;

    VLOG(3) << "Thread:" << thread_idx << "  model_file: " << model_file ;
    VLOG(3) << "Thread:" << thread_idx << "  params_file: " << params_file ;

    auto predictor = InitPredictor(model_file, params_file, device_id, cuda_stream);
    finish_flag++;
    while(finish_flag.load() < 8){
        std::this_thread::sleep_for (std::chrono::seconds(1));
    }

    if (predictor == nullptr){
        LOG(ERROR) << "Thread:" << thread_idx << "init predictor failed";
    }

    VLOG(3) << "Thread:" << thread_idx << "  start run";

    run_predict(predictor.get(), targs->input_npz, device_id, targs);

    return nullptr;
}

int main(int argc, char *argv[]) {

  // warm_up();

  dlopen("./build/custom_ops/libpd_infer_custom_op.so", RTLD_NOW);
  google::ParseCommandLineFlags(&argc, &argv, true);
  std::vector<int> dev_ids{0,1,2,3,4,5,6,7};

  pthread_t workers[FLAGS_dist_nrank];
  inference_attr attrs[FLAGS_dist_nrank];
  cnpy::npz_t input_npz = cnpy::npz_load(FLAGS_dump_input);
  int64_t cache_nums = 2 * FLAGS_max_batch * FLAGS_max_seq_len * 64 / FLAGS_dist_nrank * 128;
  VLOG(5) << "cache_nums: " << cache_nums;
  bfloat16* cache_kv_buffer_cpu = new bfloat16[cache_nums];
  for(int jj=0; jj < cache_nums; ++jj){
    cache_kv_buffer_cpu[jj] = 0;
  }

  for (int i = 0; i < FLAGS_dist_nrank; ++i) {
    VLOG(5) << "device: " << i;
    cudaSetDevice(i);
    attrs[i].rank = i;
    attrs[i].device_id = dev_ids[i];
    attrs[i].model_file = FLAGS_model_dir  + "/rank_" + std::to_string(i) + "/model.pdmodel";
    attrs[i].params_file = FLAGS_model_dir  + "/rank_" + std::to_string(i) + "/model.pdiparams";
    for(int layer_id = 0; layer_id < FLAGS_layer_num; ++layer_id){
      // VLOG(4) << "layer:" << layer_id;
      bfloat16* cache_kv_buffer = nullptr;
      cudaMalloc((void **) &cache_kv_buffer, cache_nums * sizeof(bfloat16));
      cudaMemcpy(cache_kv_buffer, cache_kv_buffer_cpu, cache_nums, cudaMemcpyHostToDevice);
      
      std::string cache_key = "cache_kvs_" + std::to_string(layer_id);
      attrs[i].shared_buffer_map[cache_key] = static_cast<void*>(cache_kv_buffer);
    }

    // Set external stream here
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    VLOG(3) << "[multi_thread] " << "stream 0 " << stream;
    attrs[i].cuda_stream = stream;

    // set input data
    attrs[i].input_npz = input_npz;
  }
  delete cache_kv_buffer_cpu;
  
  for (int i = 0; i < FLAGS_dist_nrank; ++i) {
    pthread_create(&workers[i], nullptr, thread_fn, (void*)(&attrs[i]));
  }
  for (int i = 0; i < FLAGS_dist_nrank; ++i) {
    pthread_join(workers[i], nullptr);
  }

  return 0;

}
