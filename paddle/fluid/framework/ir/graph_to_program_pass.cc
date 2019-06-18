/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/graph_to_program_pass.h"

#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <fstream>
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/operators/controlflow/while_op_helper.h"

namespace paddle {
namespace framework {
namespace ir {

void save_block_ops_use_mkldnn_info(std::string fname,
                                    const ProgramDesc& program) {
  std::ofstream out(fname);
  for (size_t i = 0; i < program.Size(); i++) {
    out << "block " << i << "\n";
    auto& block = program.Block(i);
    for (auto* op : block.AllOps()) {
      out << op->Type();
      if (op->HasAttr("use_mkldnn")) {
        out << " use_mkldnn: " << boost::get<bool>(op->GetAttr("use_mkldnn"));
      }
      out << "\n";
    }
  }
}

void GraphToProgramPass::ApplyImpl(ir::Graph* graph) const {
  // Remove the unneeded variables after memory optimization.
  std::unordered_set<std::string> vars2remove;
  if (graph->Has(kGraphToProgramVarsToRemove)) {
    vars2remove = graph->Get<std::unordered_set<std::string>>(
        kGraphToProgramVarsToRemove);
    VLOG(2) << "graph to program remove " << vars2remove.size() << " nodes";
  }

  ProgramDesc& program = Get<ProgramDesc>("program");
  save_block_ops_use_mkldnn_info("block_at_start.txt", program);

  std::unique_ptr<proto::ProgramDesc> program_pb(
      new proto::ProgramDesc(*program.Proto()));

  {
    auto block = program_pb->mutable_blocks(kRootBlockIndex);
    block->set_idx(kRootBlockIndex);
    block->clear_vars();
    std::unordered_set<std::string> visited_vars;
    for (ir::Node* n : graph->Nodes()) {
      if (n->IsVar()) {
        if (n->Var() && visited_vars.count(n->Var()->Name()) == 0 &&
            !vars2remove.count(n->Var()->Name())) {
          visited_vars.insert(n->Var()->Name());
          block->add_vars()->MergeFrom(*n->Var()->Proto());
        }
      }
    }
    block->clear_ops();

    std::vector<ir::Node*> nodes;
    if (Has(kGraphToProgramSortKind)) {
      // Inference Memory Optimize relays on this branch.
      int sort_kind = Get<int>(kGraphToProgramSortKind);
      nodes = TopologyVarientSort(
          *graph, static_cast<framework::ir::SortKind>(sort_kind));
    } else {
      nodes = TopologySortOperations(*graph);
    }

    for (ir::Node* n : nodes) {
      if (!n->Op()) continue;

      block->add_ops()->MergeFrom(*n->Op()->Proto());
    }
  }

  {
    auto* block = program_pb->mutable_blocks(1);
    // This block acquired from program is not synchronized with latest
    // program_pb, so the graph will be old and it won't work
    Graph graph2(*program.MutableBlock(1));
    block->set_idx(1);
    block->clear_vars();
    std::unordered_set<std::string> visited_vars;
    for (ir::Node* n : graph2.Nodes()) {
      if (n->IsVar()) {
        if (n->Var() && visited_vars.count(n->Var()->Name()) == 0 &&
            !vars2remove.count(n->Var()->Name())) {
          visited_vars.insert(n->Var()->Name());
          block->add_vars()->MergeFrom(*n->Var()->Proto());
        }
      }
    }
    block->clear_ops();

    std::vector<ir::Node*> nodes;
    if (Has(kGraphToProgramSortKind)) {
      // Inference Memory Optimize relays on this branch.
      int sort_kind = Get<int>(kGraphToProgramSortKind);
      nodes = TopologyVarientSort(
          graph2, static_cast<framework::ir::SortKind>(sort_kind));
    } else {
      nodes = TopologySortOperations(graph2);
    }

    for (ir::Node* n : nodes) {
      if (!n->Op()) continue;

      block->add_ops()->MergeFrom(*n->Op()->Proto());
    }
  }

  // save_block_ops_use_mkldnn_info("block_before_copyfrom.txt", program);
  program.CopyFrom(*program_pb);

  // program is not changed up to this point
  save_block_ops_use_mkldnn_info("block_after_copyfrom.txt", program);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(graph_to_program_pass, paddle::framework::ir::GraphToProgramPass);
