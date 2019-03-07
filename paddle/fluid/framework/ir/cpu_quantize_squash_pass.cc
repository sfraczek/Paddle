// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file eint8_outcept in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either eint8_outpress or
// implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/ir/cpu_quantize_squash_pass.h"
#include <string>
#include <vector>
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

void CPUQuantizeSquashPass::SingleBranch(Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::DequantQuantRM squash_pattern{gpd.mutable_pattern(), "squash_pass"};
  squash_pattern();

  int found_squash_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle cpu quantize squash pass";
    GET_IR_NODE_FROM_SUBGRAPH(dequant, dequantize, squash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(quant, quantize, squash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(next_op, next_op, squash_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(int8_out, int8_out, squash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(dequant_out, dequant_out, squash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(quant_out, quant_out, squash_pattern);

    auto* next_op_desc = next_op->Op();
    float dequant_scale = boost::get<float>(dequant->Op()->GetAttr("Scale"));
    float quant_scale = boost::get<float>(quant->Op()->GetAttr("Scale"));
    bool is_negative =
        boost::get<bool>(quant->Op()->GetAttr("is_negative_input"));

    if (dequant_scale == quant_scale) {
      auto quant_out_var_name = quant_out->Name();
      auto next_op_inputs = next_op_desc->InputNames();
      for (auto name : next_op_inputs) {
        auto var_name = next_op_desc->Input(name)[0];
        if (var_name.compare(quant_out_var_name) == 0) {
          next_op_desc->SetInput(name,
                                 std::vector<std::string>({int8_out->Name()}));
          break;
        }
      }
      // remove the dequantize and quantize op
      GraphSafeRemoveNodes(graph, {dequant, quant, dequant_out, quant_out});
      IR_NODE_LINK_TO(int8_out, next_op);

      found_squash_count++;
    } else {
      // Create an requantize Node
      OpDesc desc;
      desc.SetType("requantize");
      desc.SetInput("Input", std::vector<std::string>({int8_out->Name()}));
      desc.SetOutput("Output", std::vector<std::string>({quant_out->Name()}));
      desc.SetAttr("Scale_dequant", dequant_scale);
      desc.SetAttr("Scale_quant", quant_scale);
      desc.SetAttr("is_negative_input", is_negative);

      auto requant_node = g->CreateOpNode(&desc);  // OpDesc will be copied.
      GraphSafeRemoveNodes(graph, {dequant, quant, dequant_out});

      IR_NODE_LINK_TO(int8_out, requant_node);
      IR_NODE_LINK_TO(requant_node, quant_out);

      found_squash_count++;
    }
  };
  gpd(graph, handler);
  AddStatis(found_squash_count);
}

std::unique_ptr<ir::Graph> CPUQuantizeSquashPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  PADDLE_ENFORCE(graph.get());
  FusePassBase::Init("cpu_quantize_squash_pass", graph.get());

  SingleBranch(graph.get());

  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(cpu_quantize_squash_pass,
              paddle::framework::ir::CPUQuantizeSquashPass);
