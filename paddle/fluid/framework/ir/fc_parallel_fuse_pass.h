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

#pragma once

#include <string>
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"

namespace paddle {
namespace framework {
namespace ir {
/*
 * Fuse parallel FC operators into one with split at the end.
 */
class FcParallelFusePass : public FusePassBase {
 public:
  virtual ~FcParallelFusePass() {}

 protected:
  void ApplyImpl(ir::Graph* graph) const override;
  const std::string name_scope_{"fc_parallel_fuse"};
};
}  // namespace ir
}  // namespace framework
}  // namespace paddle
