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

#include <glog/logging.h>
#include "paddle/fluid/inference/api/paddle_quantizer_config.h"
#include "paddle_pass_builder.h"

namespace paddle {

QuantizerConfig::QuantizerConfig() {
  rules_["conv2d"]["Input"] = ScaleAlgo::MAX;
  rules_["conv2d"]["Filter"] = ScaleAlgo::KL;
  // do not calculate scale for biases
  rules_["conv2d"]["Bias"] = ScaleAlgo::NONE;
  rules_["conv2d"]["Output"] = ScaleAlgo::MAX;

  rules_["pool2d"]["X"] = ScaleAlgo::MAX;
  rules_["pool2d"]["Out"] = ScaleAlgo::MAX;
}

QuantizerConfig::QuantizerConfig(const QuantizerConfig& other)
    : AnalysisConfig(static_cast<AnalysisConfig>(other)) {
#define CP_MEMBER(member__) member__ = other.member__;
  // Quantization related.
  CP_MEMBER(use_quantizer_);
  CP_MEMBER(rules_);
  CP_MEMBER(enabled_op_types_);
  CP_MEMBER(warmup_data_);
  CP_MEMBER(warmup_bs);
#undef CP_MEMBER

  Update();
}

void QuantizerConfig::EnableQuantizer() {
  use_quantizer_ = true;

  Update();
}

ScaleAlgo QuantizerConfig::scale_algo(const std::string& op_type_name,
                                      const std::string& conn_name) const {
  if (rules_.find(op_type_name) != rules_.end()) {
    auto op_rule = rules_.at(op_type_name);
    if (op_rule.find(conn_name) != op_rule.end()) return op_rule.at(conn_name);
  }
  return ScaleAlgo::MAX;
}

// TODO(Superjomn) refactor this, buggy.
void QuantizerConfig::Update() {
  AnalysisConfig::Update();
  // Quantization passes must come after all other optimization passes
  if (use_quantizer_) {
    if (!enable_ir_optim_) {
      LOG(ERROR)
          << "EnableQuantizer() only works when IR optimization is enabled.";
    }
    pass_builder_->EnableQuantizer();
  }
}

std::string QuantizerConfig::SerializeInfoCache() {
  std::string str = AnalysisConfig::SerializeInfoCache();
  str += use_quantizer_;
  // TODO(wojtuss): handle QuantizerConfig
  return str;
}

}  // namespace paddle
