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
#include <pthread.h>
#include <thread>

#include <cuda_runtime.h>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "paddle/include/paddle_inference_api.h"
#include "paddle/include/paddle_tensor.h"
#include "paddle/include/experimental/phi/common/float16.h"

#include "cnpy.h"

using paddle_infer::Config;
using paddle_infer::CreatePredictor;
using paddle_infer::Predictor;
using phi::dtype::float16;

DEFINE_bool(use_multi_thread_inference, true, "whether use multi thread inference");
DEFINE_string(dump_input, "./dump_input.npz", "Directory of the dumped input npz file");
DEFINE_string(model_file, "", "Directory of the inference model.");
DEFINE_string(params_file, "", "Directory of the inference model.");
DEFINE_string(model_dir, "", "Directory of the inference model.");
DEFINE_int32(ring_id, 0, "ring id");
DEFINE_int32(dist_nrank, 1, "dist_nrank");
DEFINE_int32(benchmark_time, 1, "the times of inference[used for benchmark only]");


std::shared_ptr<Predictor> InitPredictor(std::string model_path, std::string params_path, int device_id=0, cudaStream_t stream=nullptr) {
  Config config;

  config.SetModel(model_path, params_path);
  // int device_id = rank >= 0? rank : 0;
  config.EnableUseGpu(100, device_id);
  config.EnableMemoryOptim();
  config.SwitchIrOptim(true);
  config.SetCpuMathLibraryNumThreads(1);
  if(device_id == 0){
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
  // LOG(INFO) << "Thread "<< rank <<"  Used passes: " << config.pass_builder()->DebugString();
  LOG(INFO) << config.Summary();
  return CreatePredictor(config);
}

void run_predict(Predictor *predictor, int rank=-1) {
  LOG(INFO) << "start load dump input";
  cnpy::npz_t input_npz = cnpy::npz_load(FLAGS_dump_input);
  auto input_names = predictor->GetInputNames();
  // LOG(INFO) << "***** input names: ******";
  auto input_types = predictor->GetInputTypes();
  for (auto name : input_names) {
    auto input_tmp = predictor->GetInputHandle(name);
    auto input_tmp_shape = input_tmp->shape();
    cnpy::NpyArray arr_npy = input_npz[name];
    std::vector<int> shape;
    
    LOG(INFO) <<  "[check input] input names:" <<  name;
    for(size_t j=0; j < arr_npy.shape.size(); ++j){
      shape.emplace_back(arr_npy.shape[j]);
      LOG(INFO) <<  "[check shape]: " <<  arr_npy.shape[j];
    }
    LOG(INFO) <<  "[check dtype] : " << input_types[name];
    input_tmp->Reshape(shape);

    if(input_types[name] == paddle_infer::DataType::FLOAT32){
      float* data = arr_npy.data<float>();
      input_tmp->CopyFromCpu(data);
    }
    else if(input_types[name] == paddle_infer::DataType::INT64){
      long* data = arr_npy.data<long>();
      input_tmp->CopyFromCpu(data);
    }
  }

  for (int i = 0; i < FLAGS_benchmark_time; i++) {
    // predictor->ExpRunWithExternalStream(comm_0.stream());
    auto start = std::chrono::system_clock::now();
    CHECK(predictor->Run());
    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    LOG(INFO) << "[BENCHMARK] ellapse " << elapsed.count() << "ms \n";
  }
  LOG(INFO) << "check run predictor finished";

  return;
}


struct thread_attr {
    int rank;
    int device_id;
    int32_t thread_idx;
    std::string model_file;
    std::string params_file;
    int* ready_count;
    cudaStream_t cuda_stream{nullptr};
};

void *thread_fn(void *args) {
    struct thread_attr *targs = (struct thread_attr *)args;
    // int rank = targs->rank;
    int device_id = targs->device_id;
    int32_t thread_idx = targs->thread_idx;
    std::string model_file = targs->model_file;
    std::string params_file = targs->params_file;
    cudaStream_t cuda_stream = targs->cuda_stream;

    LOG(INFO) << "Thread:" << thread_idx << "  model_file: " << model_file ;
    LOG(INFO) << "Thread:" << thread_idx << "  params_file: " << params_file ;

    auto predictor = InitPredictor(model_file, params_file, device_id, cuda_stream);

    if (predictor == nullptr){
        LOG(ERROR) << "Thread:" << thread_idx << "init predictor failed";
    }

    LOG(INFO) << "Thread:" << thread_idx << "  start run";

    run_predict(predictor.get(), device_id);

    auto output_names = predictor.get()->GetOutputNames();
    auto output_t = predictor.get()->GetOutputHandle(output_names[0]);
    std::vector<int> output_shape = output_t->shape();
    size_t shape_product = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
    std::vector<int64_t>out_data(shape_product);
    output_t->CopyToCpu(out_data.data());
    if(device_id == 0){
      std::cout << "print out\n";
      for(size_t i=0; i< shape_product; ++i){
          std::cout << out_data[i] << ", ";
      }
      std::cout << std::endl;
    }
    

    return nullptr;
}


int main(int argc, char *argv[]) {

  google::ParseCommandLineFlags(&argc, &argv, true);

  std::vector<int> dev_ids{0,1,2,3,4,5,6,7};
  // auto select_dev_ids = std::vector<int>(dev_ids.begin(), dev_ids.begin() + FLAGS_dist_nrank);

  // paddle::platform::NCCLCommContext::Instance().CreateAllNCCLComms(select_dev_ids, FLAGS_ring_id);
  pthread_t workers[FLAGS_dist_nrank];
  thread_attr attrs[FLAGS_dist_nrank];

  for (int i = 0; i < FLAGS_dist_nrank; ++i) {

    attrs[i].rank = i;
    attrs[i].device_id = dev_ids[i];
    attrs[i].model_file = FLAGS_model_dir  + "/rank_" + std::to_string(i) + "/model.pdmodel";
    attrs[i].params_file = FLAGS_model_dir  + "/rank_" + std::to_string(i) + "/model.pdiparams";

    // Set external stream
    cudaSetDevice(i);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    VLOG(3) << "[multi_thread] " << "stream 0 " << stream;
    attrs[i].cuda_stream = stream;

  }
  
  for (int i = 0; i < FLAGS_dist_nrank; ++i) {
    pthread_create(&workers[i], nullptr, thread_fn, (void*)(&attrs[i]));
  }
  for (int i = 0; i < FLAGS_dist_nrank; ++i) {
    pthread_join(workers[i], nullptr);
  }

  return 0;

}
