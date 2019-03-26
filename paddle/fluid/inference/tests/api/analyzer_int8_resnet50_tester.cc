/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <fstream>
#include <iostream>
#include "paddle/fluid/inference/tests/api/tester_helper.h"

namespace paddle {
namespace inference {
namespace analysis {

void SetConfig(AnalysisConfig *cfg) {
  cfg->SetModel(FLAGS_infer_model);
  cfg->SetProgFile("__model__");
  cfg->DisableGpu();
  cfg->SwitchIrOptim();
  cfg->SwitchSpecifyInputNames();
  cfg->SetCpuMathLibraryNumThreads(FLAGS_paddle_num_threads);

  cfg->EnableMKLDNN();
}

template <typename T>
PaddleTensor LoadTensorFromStream(std::ifstream &file, std::vector<int> shape,
                                  std::string name) {
  PaddleTensor tensor;
  tensor.name = name;
  tensor.shape = std::move(shape);
  tensor.dtype = GetPaddleDType<T>();
  size_t numel = std::accumulate(begin(tensor.shape), end(tensor.shape), 1,
                                 std::multiplies<T>());
  tensor.data.Resize(numel * sizeof(T));

  std::copy_n(std::istream_iterator<T>(file), numel,
              static_cast<T *>(tensor.data.data()));

  return tensor;
}

void SetInput(std::vector<std::vector<PaddleTensor>> *inputs) {
  PADDLE_ENFORCE_EQ(FLAGS_test_all_data, 0, "Only have single batch of data.");
  std::string line;
  std::ifstream file(FLAGS_infer_data);

  int batch_size = 100;
  PaddleTensor input =
      LoadTensorFromStream<float>(file, {batch_size, 3, 224, 224}, "input");

  PaddleTensor labels =
      LoadTensorFromStream<int64_t>(file, {batch_size, 1}, "label");

  std::vector<PaddleTensor> input_slots{{input, labels}};
  inputs->emplace_back(input_slots);
}

TEST(Analyzer_int8_resnet50, quantization) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  AnalysisConfig q_cfg;
  SetConfig(&q_cfg);

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  auto warmup_data =
      std::make_shared<std::vector<PaddleTensor>>(input_slots_all[0]);

  q_cfg.EnableQuantizer();
  q_cfg.quantizer_config()->SetWarmupData(warmup_data);
  q_cfg.quantizer_config()->SetWarmupBatchSize(
      warmup_data->front().shape.front());

  CompareQuantizedAndAnalysis(
      reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
      reinterpret_cast<const PaddlePredictor::Config *>(&q_cfg),
      input_slots_all);
}

TEST(Analyzer_int8_resnet50, profile) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  std::vector<std::vector<PaddleTensor>> input_slots_all_warmup;
  SetInput(&input_slots_all_warmup);
  auto warmup_data =
      std::make_shared<std::vector<PaddleTensor>>(input_slots_all_warmup[0]);

  cfg.EnableQuantizer();
  cfg.quantizer_config()->SetWarmupData(warmup_data);
  cfg.quantizer_config()->SetWarmupBatchSize(
      warmup_data->front().shape.front());

  std::vector<PaddleTensor> outputs;

  std::vector<std::vector<PaddleTensor>> input_slots_all_test;
  std::vector<std::string> feed_names = {"input", "label"};
  std::vector<PaddleDType> feed_dtypes = {PaddleDType::FLOAT32,
                                          PaddleDType::INT64};
  SetFakeImageInput(&input_slots_all_test, FLAGS_infer_model, false,
                    "__model__", "", &feed_names, &feed_dtypes, 0);
  TestPrediction(reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
                 input_slots_all_test, &outputs, FLAGS_num_threads);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
