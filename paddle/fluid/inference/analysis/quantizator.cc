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
#include <map>
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/type_defs.h"

namespace paddle {
namespace inference {
namespace analysis {

bool Quantizator::RunWarmup() {
  // std::unique_ptr<std::map<std::string, PaddleTensor>> & quant_vars) {
  VLOG(3) << "Predictor: run a quantization warmup iteration";
  auto warmup_data = config_->GetQuantWarmupData();

  PADDLE_ENFORCE_NOT_NULL(warmup_data,
                          "Warmup data cannot be NULL in the config.");

  std::vector<PaddleTensor> output_slots;

  // Run the inference program
  predictor_run_(*warmup_data.get(), &output_slots,
                 config_->GetWarmupBatchSize());

  return true;
}

bool Quantizator::GatherData() {
  /*
   *   std::map<std::string, std::map<std::string, LoDTensor>> gathered_data;
   *   for (auto *op : infer_program_->Block(0).AllOps()) {
   *     if (op->HasAttr("quantize") && op->Attr<bool>("quantize")) {
   *       const VariableNameMap &connections = op->Inputs();
   *       const VariableNameMap &connections_out = op->Outputs();
   *       connections.insert(connections.end(), connections_out.begin(),
   *                          connections_out.end());
   *
   *       for (auto &conn_name : connections) {
   *         Variable *var = scope_.FindVar(var_name);
   *         PADDLE_ENFORCE(var, "%s is not in the scope", var_name);
   *         PADDLE_ENFORCE(var->IsType<LoDTensor>(),
   *                        "Only support lod tensor now.");
   *         LoDTensor *var_tensor = var->GetMutable<LoDTensor>();
   *
   *         CalculateScales(...);
   *
   *         //...
   *       }
   *     }
   *   }
   */
  return true;
}

void Quantizator::CalculateScales(std::string op_name, std::string conn_name,
                                  std::string var_name, LoDTensor &lod_tensor) {
  // adds pairs variable name -> LoDTensor with scale to the scales map
}

bool Quantizator::RunQuantizePass() {
  // push the scales to the quantize pass
  auto quantize_pass =
      framework::ir::PassRegistry::Instance().Get("quantize_pass");
  quantize_pass->Set<std::map<std::string, LoDTensor *>>("quant_var_names",
                                                         &scales_);
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
  // gather data from variables and calculate scales for them
  if (!GatherData()) return false;
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
