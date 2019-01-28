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

#include "paddle/fluid/framework/ir/quant_gather_var_names_pass.h"
#include <functional>
#include <string>
#include <vector>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/operators/math/cpu_vec.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

std::unique_ptr<ir::Graph> QuantGatherVarNamesPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  VLOG(3) << "Gathers names of variables to be quantized.";
  PADDLE_ENFORCE(graph.get());

  // store also var type (weight, bias, etc.) and op type?
  std::unordered_set<std::string> var_names;
  int i = 0;
  std::string prefix("zmienna_");
  for (const Node* n : graph->Nodes()) {
    if (n->IsOp()) {
      auto* op = n->Op();
      PADDLE_ENFORCE_NOT_NULL(n->Op());
      if (op->HasAttr("use_int8") &&
          boost::get<bool>(op->GetAttr("use_int8"))) {
        // get all inputs
        var_names.insert(prefix + std::to_string(i));
        // get all outputs
        // get all weights
        // get all biases
      }
    }
  }
  auto quant_gather_pass = framework::ir::PassRegistry::Instance().Get(
      "quant_gather_var_names_pass");
  // std::move?
  quant_gather_pass->Set<std::unordered_set<std::string>>(
      std::string("quant_var_names"),
      new std::unordered_set<std::string>(var_names));

  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(quant_gather_var_names_pass,
              paddle::framework::ir::QuantGatherVarNamesPass);
