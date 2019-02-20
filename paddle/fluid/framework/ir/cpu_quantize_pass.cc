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

#include "paddle/fluid/framework/ir/cpu_quantize_pass.h"

namespace paddle {
namespace framework {
namespace ir {

namespace {

enum class OpTypes { conv2d, pool2d };

std::map<std::string, OpTypes> string2OpType{
    std::make_pair("conv2d", OpTypes::conv2d),
    std::make_pair("pool2d", OpTypes::pool2d)};

void UnlinkNodes(ir::Node* a, ir::Node* b) {
  a->outputs.erase(std::remove(a->outputs.begin(), a->outputs.end(), b),
                   a->outputs.end());
  b->inputs.erase(std::remove(b->inputs.begin(), b->inputs.end(), a),
                  b->inputs.end());
}

void QuantizeLoDTensor(const LoDTensor& src, LoDTensor* dst) {
  // Quantize
}

}  // namespace

std::unique_ptr<ir::Graph> CPUQuantizePass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  VLOG(3) << "Quantizes the graph.";
  std::cout << "--- This is cpu quantize pass. ---" << std::endl;
  PADDLE_ENFORCE(graph.get());
  FusePassBase::Init(name_scope_, graph.get());

  /*
   *   auto* scope = param_scope();
   *   PADDLE_ENFORCE(scope);
   *
   *   GraphPatternDetector gpd;
   *   auto pattern = gpd.mutable_pattern();
   *
   *   patterns::Conv conv_pattern{pattern, name_scope_};
   *   conv_pattern();
   *
   *   int quantize_conv_count = 0;
   *   auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
   *                      Graph* g) {
   *     VLOG(4) << "handle Conv2d quantization";
   *     GET_IR_NODE_FROM_SUBGRAPH(conv_op, conv_op, conv_pattern);
   *     GET_IR_NODE_FROM_SUBGRAPH(conv_input, conv_input, conv_pattern);
   *     GET_IR_NODE_FROM_SUBGRAPH(conv_filter, conv_filter, conv_pattern);
   *     GET_IR_NODE_FROM_SUBGRAPH(conv_output, conv_output, conv_pattern);
   *
   *     auto* conv_op_desc = conv_op->Op();
   *     if (!conv_op_desc->HasAttr("use_quantizer") ||
   *         !boost::get<bool>(conv_op_desc->GetAttr("use_quantizer")))
   *       return;
   *
   *     // insert quantize and dequantize op
   *
   *     auto* conv_input_tensor =
   *         scope->Var(conv_input->Name())->GetMutable<LoDTensor>();
   *     auto* conv_output_tensor =
   *         scope->Var(conv_output->Name())->GetMutable<LoDTensor>();
   *
   *     // Create and initialize quantize output variable
   *     VarDesc quantize_out_desc(patterns::PDNodeName("quantize",
   * "quantize_out"));
   *     auto* quantize_out_node = g->CreateVarNode(&quantize_out_desc);
   *     auto* quantize_out_tensor =
   *         scope->Var(quantize_out_node->Name())->GetMutable<LoDTensor>();
   *     quantize_out_tensor->Resize(conv_input_tensor->dims());
   *     std::fill_n(quantize_out_tensor->mutable_data<int8_t>(platform::CPUPlace()),
   *                 quantize_out_tensor->numel(), 0);
   *
   *     // Create dequantize input variable
   *     VarDesc dequantize_in_desc(
   *         patterns::PDNodeName("dequantize", "dequantize_in"));
   *     auto* dequantize_in_node = g->CreateVarNode(&dequantize_in_desc);
   *     auto* dequantize_in_tensor =
   *         scope->Var(dequantize_in_node->Name())->GetMutable<LoDTensor>();
   *     dequantize_in_tensor->Resize(conv_output_tensor->dims());
   *     std::fill_n(
   *         dequantize_in_tensor->mutable_data<int32_t>(platform::CPUPlace()),
   *         dequantize_in_tensor->numel(), 0);
   *
   *     // create a quantize op node.
   *     OpDesc q_desc;
   *     q_desc.SetType("quantize");
   *     q_desc.SetInput("Input",
   * std::vector<std::string>({conv_output->Name()}));
   *     q_desc.SetOutput("Output",
   *                      std::vector<std::string>({quantize_out_node->Name()}));
   *     q_desc.SetAttr("Scale", 1.0f);
   *     q_desc.SetAttr("is_negative_input", true);
   *     auto quantize_op = g->CreateOpNode(&q_desc);  // OpDesc will be copied.
   *
   *     // create a dequantize op node.
   *     OpDesc deq_desc;
   *     deq_desc.SetType("dequantize");
   *     deq_desc.SetInput("Input",
   *                       std::vector<std::string>({dequantize_in_node->Name()}));
   *     deq_desc.SetOutput("Output",
   *                        std::vector<std::string>({conv_output->Name()}));
   *     deq_desc.SetAttr("Scale", 1.0f);
   *     auto dequantize_op = g->CreateOpNode(&deq_desc);  // OpDesc will be
   * copied.
   *
   *     conv_op_desc->SetInput(
   *         "Input", std::vector<std::string>({quantize_out_node->Name()}));
   *     conv_op_desc->SetOutput(
   *         "Output", std::vector<std::string>({dequantize_in_node->Name()}));
   *     // conv_op_desc->SetInput("Input", std::vector<std::string>({"aaa"}));
   *
   *     // link quantize op
   *     IR_NODE_LINK_TO(conv_input, quantize_op);
   *     IR_NODE_LINK_TO(quantize_op, quantize_out_node);
   *     IR_NODE_LINK_TO(quantize_out_node, conv_op);
   *     UnlinkNodes(conv_input, conv_op);
   *
   *     // link dequantize op
   *     IR_NODE_LINK_TO(conv_op, dequantize_in_node);
   *     IR_NODE_LINK_TO(dequantize_in_node, dequantize_op);
   *     IR_NODE_LINK_TO(dequantize_op, conv_output);
   *     UnlinkNodes(conv_op, conv_output);
   *
   *     // quantize weights
   *     // quantize bias
   *     // update op?
   *
   *     ++quantize_conv_count;
   *   };
   *
   *   gpd(graph.get(), handler);
   *   AddStatis(quantize_conv_count);
   */
  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(cpu_quantize_pass, paddle::framework::ir::CPUQuantizePass)
    .RequirePassAttr("quant_var_scales");
