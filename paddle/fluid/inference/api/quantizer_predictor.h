// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include <algorithm>
#include <map>
#include <string>
#include <vector>
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/inference/analysis/analyzer.h"
#include "paddle/fluid/inference/analysis/quantizer.h"
#include "paddle/fluid/inference/api/analysis_predictor.h"
#include "paddle/fluid/inference/api/api_impl.h"
#include "paddle/fluid/inference/api/details/reset_tensor_array.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/string/printf.h"
#ifdef PADDLE_WITH_TESTING
#include <gtest/gtest.h>
#include <gtest/gtest_prod.h>
#endif
namespace paddle {

using inference::analysis::Argument;
using inference::analysis::Analyzer;
using framework::proto::ProgramDesc;
using framework::NaiveExecutor;

/** \brief This predictor enables quantization and is based on the analyzer
 * predictor.
 *
 * It will do what AnalysisPredictor does and additionally Quantize the graph.
 *
 */
class QuantizerPredictor : public AnalysisPredictor {
 public:
  explicit QuantizerPredictor(const QuantizerConfig &config)
      : config_(config) {}
  ~QuantizerPredictor();

  // bool Run(const std::vector<PaddleTensor> &inputs,
  //          std::vector<PaddleTensor> *output_data,
  //          int batch_size = -1) override;

  void PrepareArgument() override;

  std::unique_ptr<ZeroCopyTensor> GetInputTensor(
      const std::string &name) override;
  std::unique_ptr<ZeroCopyTensor> GetOutputTensor(
      const std::string &name) override;

  bool ZeroCopyRun() override;

  std::unique_ptr<PaddlePredictor> Clone() override;

  std::string GetSeriazlizedProgram() const override;

  bool Quantize();

 protected:
  // TODO(sfraczek): I may need to keep some of them and remove other
  QuantizerConfig config_;
  std::shared_ptr<inference::analysis::Quantizer> quantizer_;
};

}  // namespace paddle
