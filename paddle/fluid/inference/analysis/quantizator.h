// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

/*
 * This file defines IRPassManager, it helps control the passes in IR. Inference
 * phrase will load the model program and parameters from disk, that is quite
 * different from the training phase.
 * This manager will control the Passes and make the passes in IR work smoothly
 * for inference.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/api/paddle_quantize_config.h"

namespace paddle {
namespace inference {
namespace analysis {

using contrib::QuantizeConfig;
using framework::NaiveExecutor;
using framework::Scope;

class Quantizator final {
 public:
  explicit Quantizator(std::unique_ptr<framework::NaiveExecutor>& executor,
                       std::shared_ptr<framework::Scope>& scope,
                       std::shared_ptr<framework::ProgramDesc>& infer_program,
                       const std::shared_ptr<QuantizeConfig>& config)
      : executor_(executor),
        scope_(scope),
        infer_program_(infer_program),
        config_(config) {}

  bool Quantize();

 private:
  bool RunWarmup();
  bool GatherData();
  bool CalculateScales(std::string op_name, std::string conn_name,
                       std::string var_name, LoDTensor& lod_tensor);
  bool RunQuantizePass();
  bool RunOptimizePass();
  bool SaveModel();

 private:
  std::unique_ptr<framework::NaiveExecutor>& executor_;
  std::shared_ptr<framework::Scope>& scope_;
  std::shared_ptr<framework::ProgramDesc>& infer_program_;
  const std::shared_ptr<contrib::QuantizeConfig>& config_;

  // variable name -> data
  std::map<std::string, framework::LoDTensor> scales;
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
