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
#include <map>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/platform/place.h"
#include <tuple>

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

void Quantizator::CalculateScales(const std::string& op_name,
                                  const std::string& conn_name,
                                  const std::string& var_name,
                                  LoDTensor& lod_tensor, float int_max_value) {
  // adds pairs variable name -> LoDTensor with scale to the scales map

  using contrib::QuantizeAlgorithm;
  using platform::CPUPlace;
  using framework::EigenVector;

  LoDTensor scale_tensor;
  scale_tensor.Resize({1});
  if (lod_tensor.numel() == 0)
    throw std::runtime_error(
        "Quantizator: LoDTensor of variable for quantization should not be "
        "empty.");
  // TODO: fix me
  // auto eigen_data_vector = EigenVector<float>::From(lod_tensor);

  auto rule = config_->GetRule(op_name, conn_name);
  switch (rule) {
    case QuantizeAlgorithm::none:
      return;
    case QuantizeAlgorithm::minmax: {
      // TODO: fix me
      // auto tensor_max_value = eigen_data_vector.lpNorm<Eigen::Infinity>();
      auto tensor_max_value = 1;
      auto quantization_factor = int_max_value / tensor_max_value;
      scale_tensor.mutable_data<float>(CPUPlace())[0] = quantization_factor;
      break;
    }
    case QuantizeAlgorithm::KL:
      Quantizator::GetOptimalScalingFactor(eigen_data_vector);
      throw std::runtime_error(
          "Quantizator: QuantizeAlgorithm KL is not yet implemented.");
      break;
    default:
      throw std::runtime_error(
          "Quantizator: Unexpected QuantizeAlgorithm specified.");
  }
  scales_[var_name] = std::move(scale_tensor);
}

// Using the KL-divergence method get the most precise scaling factor.
void Quantizator::GetOptimalScalingFactor(EigenVector activation_blob,
                                          int num_quantized_bins = 255) {
  float max_val = activation_blob.maxCoeff();
  float min_val = activation_blob.minCoeff();
  std::vector<int> hist;
  float bin_width;
  int starting_iter;
  int ending_iter;
  if (min_val >= 0) {
    std::tie(hist, bin_width) =
        Histogram(activation_blob, 2048, min_val, max_val);
    ending_iter = 2047;
    starting_iter = static_cast<int>(ending_iter * 0.7);
  } else {
    th = max(abs(max_val), abs(min_val));
    std::tie(hist, bin_width) = Histogram(activation_blob, 2048, -th, th);
    starting_iter = 0;
    ending_iter = 2047;
    if (abs(max_val) > abs(min_val)) {
      while (starting_iter < ending_iter) {
        if (hist[starting_iter] == 0) {
          starting_iter += 1;
          continue;
        } else {
          break;
        }
      }
      starting_iter += static_cast<int>((ending_iter - starting_iter) * 0.6);
    } else {
      while (ending_iter > 0) {
        if (hist[ending_iter] == 0) {
          ending_iter -= 1;
          continue;
        } else {
          break;
        }
      }
      starting_iter = static_Cast<int>(0.6 * ending_iter);
    }
  }
  auto P_sum = activation_blob.size();
}

// Returns histogram and bin width
std::tuple<std::vector<int>, float> Quantizator::Histogram(
    EigenVector<float> activation_blob, int num_bins = 2048, float min_val,
    float max_val) {
  auto bin_width = (max_val - min_val) / num_bins;
  std::vector<int> hist(num_bins);
  for (auto *val : activation_blob) {
    int bin = static_cast<int>(floor((val - min_val) / bin_width));
    hist[bin] = val;
  }
  return std::make_tuple(std::move(hist), std::move(bin_width));
}

void Quantizator::KLDivergence(refDistP, candidDistQ) {
  // naive implementation for 8 bits
  // collect historgram of activations
  // generate many quantized distributions with different saturation thresholds
  // pick threshold which minimizes KL divergence(reference_distribution,
  // quantized_distribution)
    outliersCount = sum(
}

bool Quantizator::RunQuantizePass() {
  // push the scales to the quantize pass
  auto quantize_pass =
      framework::ir::PassRegistry::Instance().Get("quantize_pass");
  quantize_pass->Set<std::map<std::string, LoDTensor>>("quant_var_names",
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
