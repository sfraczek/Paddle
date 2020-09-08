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

void FCResidualMKLDNNFusePass::ApplyImpl(ir::Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fc_elementwise_add_mkldnn_fuse_pass,
              paddle::framework::ir::FCResidualMKLDNNFusePass);
