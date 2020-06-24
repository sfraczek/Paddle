// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/mkldnn/stack_elementwise_add_fuse_pass.h"

#include <paddle/fluid/string/pretty_log.h>

#include <vector>

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

void StackElementwiseAddMkldnnFusePass::ApplyPass(Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(graph,
                          platform::errors::InvalidArgument(
                              "Pointer to graph argument should not be NULL."));
  FusePassBase::Init(name_scope_, graph);
  GraphPatternDetector gpd;
  patterns::StackElementwiseAdd stack_elementwise_add(gpd.mutable_pattern(),
                                                      name_scope_);

  stack_elementwise_add();

  int found_stack_elementwise_add_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "handle StackElementrwiseAddMkldnn fuse";
    GET_IR_NODE_FROM_SUBGRAPH(prev_op, prev_op, stack_elementwise_add);
    GET_IR_NODE_FROM_SUBGRAPH(stack_in, stack_in, stack_elementwise_add);
    GET_IR_NODE_FROM_SUBGRAPH(stack_op, stack_op, stack_elementwise_add);
    GET_IR_NODE_FROM_SUBGRAPH(stack_out, stack_out, stack_elementwise_add);
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_add_op,
                              elementwise_add_op stack_elementwise_add);
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_add_out, elementwise_add_out,
                              stack_elementwise_add);

    if (boost::get<int>(stack_op->Op()->GetAttr("axis") != 1) return;

    if (elementwise_add_op->Op()->GetAttrIfExists("axis") != -1) return;

    if (elementwise_add_op->Op()->Inputs().at("Y").at(0) != stack_out->Name())
      return;

    elementwise_add_op->SetAttr("axis", 2);
    elementwise_add_op->SetInput("Y", {stack_in->Name()});

    std::unordered_set<const ir::node*> nodes_to_remove{stack_out};
    if (stack_op->Outputs().size() == 1) { 
      nodes_to_remove.insert(stack_op);
    }
    GraphSafeRemoveNodes(graph, nodes_to_remove);

    IR_NODE_LINK_TO(stack_in, elementwise_add_in);

    ++found_reshape_transpose_matmul_count;
  };

  gpd(graph, handler);
  AddStatis(found_stack_elementwise_add_count);

  std::stringstream msg_ss;
  msg_ss << "---    Fused " << found_stack_elementwise_add_count
         << " StackElementwiseAddMkldnnFusePass patterns";
  string::PrettyLogDetail(msg_ss.str().c_str());
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(stack_elementwise_add_mkldnn_fuse_pass,
              paddle::framework::ir::StackElementwiseAddMkldnnFusePass);
