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

#include "paddle/fluid/inference/analysis/quantizator.h"
#include <algorithm>

namespace paddle {
namespace inference {
namespace analysis {

bool Quantizator::RunWarmup() {
  // std::unique_ptr<std::map<std::string, PaddleTensor>> & quant_vars) {
  VLOG(3) << "Predictor: run a quantization warmup iteration";
  PADDLE_ENFORCE_NOT_NULL(config_.GetQuantWarmupData(),
                          "Warmup data cannot be NULL in the config.");
  framework::Scope *scope = sub_scope_ ? sub_scope_ : scope_.get();

  if (!SetFeed(*config_.quant_warmup_data_, scope)) {
    LOG(ERROR) << "fail to set feed for warmup iteration";
    return false;
  }

  // Run the inference program
  executor_->Run();

  return true;
}

bool Quantizator::GatherData() {
  // op_name, var_name
  std::map<std::string, std::map<std::string, LoDTensor>> data;
  for (auto *op : inference_program_->Block(0).AllOps()) {
    if (op->HasAttr("quantize") && op->Attr<bool>("quantize")) {
      std::vector<std::string> input_var_names = op->InputVars();
      std::vector<std::string> output_var_names = op->OutputVars();
      for (auto &var_name : input_var_names) {
        LoDTensor lod_tensor = framework::GetVariableTensor(var_name);
        CalculateScales(...);

        //...
      }
    }
  }
}

void Quantizator::CalculateScales(std::string op_name, std::string conn_name,
                                  std::string var_name, LoDTensor &lod_tensor) {
  // adds pairs variable name -> LoDTensor with scale to the scales map

  using contrib::QuantizeAlgorithm;
  using framework::CPUPlace;

  LoDTensor scale_tensor;
  scale_tensor.Resize(1);
  // auto *tensor_ptr = scale_tensor.mutable_data<float>(CPUPlace);
  auto eigen_tensor = EigenVector<float>::From(scale_tensor);

  auto &rule = config_->rules_[op_name][conn_name];
  switch (rule) {
    case QuantizeAlgorithm::none:
      return;
    case QuantizeAlgorithm::minmax:
      max_value = eigen_tensor.lpNorm<Eigen::Infinity>();
      break;
    case QuantizeAlgorithm::KL:
      break;
    default:
      throw std::runtime_error("Unknown QuantizeAlgorithm for quantization.");
  }
  scales[var_name] = std::move(scale_tensor);
}

bool Quantizator::RunQuantizePass() {
  // push the scales to the quantize pass
  auto quantize_pass =
      framework::ir::PassRegistry::Instance().Get("quantize_pass");
  quantize_pass->Set<std::map<std::string, LoDTensor *>>("quant_var_names",
                                                         scales_);
  //
  return true;
}

bool Quantizator::RunOptimizePass() {
  //
  return true;
}

bool Quantizator::SaveModel() {
  //
  return true;
}

bool Quantizator::Quantize() {
  // warmup iteration
  if (!RunWarmup()) return false;
  // gather data from variables
  if (!GatherData()) return false;
  // calculate scales
  if (!CalculateScales()) return false;
  // run quantization pass
  // run optimization passes
  // save quantized model if required
  if (!RunQuantizePass()) return false;
  if (!RunOptimizePass()) return false;
  if (!SaveModel()) return false;

  return true;
}

/*
    auto gather_pass = framework::ir::PassRegistry::Instance().Get(
        "quant_gather_var_names_pass");
    const auto &qvars_names =
        gather_pass->Get<std::unordered_set<std::string>>("quant_var_names");
    std::cout << "Length of the names set: " << qvars_names.size() << std::endl;
    // a vector for variables to be quantized
    // std::unique_ptr<std::map<std::string, PaddleTensor>> q_vars(
    // new std::vector<PaddleTensor>());
    // run 1 iteration of inference
    // RunQuantWarmup(q_vars);
    // store the data in the int8_scale_pass
    // auto pass =
    // framework::ir::PassRegistry::Instance().Get("quant_scale_pass");
    // pass->Set("quant_vars_data", std::move(q_vars));
*/

// ...
}  // namespace analysis
}  // namespace inference
}  // namespace paddle
