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

#include "paddle/fluid/framework/ir/mkldnn/fc_elementwise_add_mkldnn_fuse_pass.h"
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <tuple>
#include "paddle/fluid/framework/ir/graph_traits.h"

namespace paddle {
namespace framework {
namespace ir {

#define GET_IR_NODE(node__) GET_IR_NODE_FROM_SUBGRAPH(node__, node__, fc_elementwise_add_pattern);
#define GET_NODES                    \
  GET_IR_NODE(fc_op);                \
  GET_IR_NODE(fc_out);               \
  GET_IR_NODE(elementwise_add_op);   \
  GET_IR_NODE(elementwise_add_in_y); \
  GET_IR_NODE(elementwise_add_out);

void FCResidualMKLDNNFusePass::ApplyImpl(ir::Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);

  GraphPatternDetector gpd;

  patterns::FCElementwiseadd fc_elementwise_add_pattern(gpd.mutable_pattern(), name_scope_);
  fc_elementwise_add_pattern();

  int fuse_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_NODES;

    auto fc_op_desc = fc_op->Op();

    fc_op_desc->SetInput("ResidualData", {elementwise_add_in_y->Name()});
    fc_op_desc->SetOutput("Out", {elementwise_add_out->Name()});

    IR_NODE_LINK_TO(elementwise_add_in_y, fc_op);  // ResidualData
    IR_NODE_LINK_TO(fc_op, elementwise_add_out);   // Output

    // Delete the unneeded nodes.
    GraphSafeRemoveNodes(graph, {fc_out, elementwise_add_op});
    ++fuse_count;
  };

  gpd(graph, handler);

  LOG(INFO) << "Fused graph " << fuse_count << "\n";
  AddStatis(fuse_count);
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fc_elementwise_add_mkldnn_fuse_pass,
              paddle::framework::ir::FCResidualMKLDNNFusePass);
