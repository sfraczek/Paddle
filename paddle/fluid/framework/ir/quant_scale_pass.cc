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

#include "paddle/fluid/framework/ir/quant_scale_pass.h"
#include <functional>
#include <string>
#include <vector>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/operators/math/cpu_vec.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

std::unique_ptr<ir::Graph> QuantScalePass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  VLOG(3) << "Calculates INT8 quantization scales for variable nodes.";
  PADDLE_ENFORCE(graph.get());
  FusePassBase::Init(name_scope_, graph.get());

  auto quant_var_names =
      Get<std::unordered_set<std::string>>("quant_var_names");
  // auto f32_vars_data =
  // Get<std::map<std::string, PaddleTensor>>("f32_vars_data");

  std::cout << "Nazwy zmiennych:" << std::endl;
  for (auto& name : quant_var_names) {
    std::cout << name << std::endl;
  }

  /*
   * for (const Node* n : graph->Nodes()) {
   *   if (n->IsOp()) {
   *     auto* op = n->Op();
   *     if (op->HasAttr("use_int8") &&
   *         boost::get<bool>(op->GetAttr("use_int8"))) {
   *       // for all input, output, weights and bias variable nodes
   *       // take its name
   *       // find the name in the map with data
   *       // calculate the scale values
   *       // create the scale tensor and node
   *       // link the scale node to the variable node
   *     }
   *   }
   * }
   */

  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(quant_scale_pass, paddle::framework::ir::QuantScalePass);
