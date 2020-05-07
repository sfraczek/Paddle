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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "paddle/fluid/framework/ir/mkldnn/fc_parallel_mkldnn_fuse_pass.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"

namespace paddle {
namespace framework {
namespace ir {

float* AddVarToScope(Scope* param_scope, const std::string& name,
                     const DDim& dims) {
  auto* tensor = param_scope->Var(name)->GetMutable<LoDTensor>();
  tensor->Resize(dims);
  return tensor->mutable_data<float>(platform::CPUPlace());
}

Scope* CreateParamScope(int width) {
  auto param_scope = new Scope();
  auto* w1_p = AddVarToScope(param_scope, "w1", {width, width});
  auto* w2_p = AddVarToScope(param_scope, "w2", {width, width});
  auto* w3_p = AddVarToScope(param_scope, "w3", {width, width});
  auto* b1_p = AddVarToScope(param_scope, "b1", {width});
  auto* b2_p = AddVarToScope(param_scope, "b2", {width});
  auto* b3_p = AddVarToScope(param_scope, "b3", {width});
  std::iota(b1_p, b1_p + width, 0);
  std::iota(b2_p, b2_p + width, width);
  std::iota(b3_p, b3_p + width, 2 * width);
  std::iota(w1_p, w1_p + width * width, 0);
  std::iota(w2_p, w2_p + width * width, width * width);
  std::iota(w3_p, w3_p + width * width, 2 * width * width);
  return param_scope;
}

// void SetOp(ProgramDesc *prog, const std::string &type,
//            const std::vector<std::string> &inputs,
//            const std::vector<std::string> &outputs, bool use_mkldnn) {
//   auto *op = prog->MutableBlock(0)->AppendOp();
//   op->SetType(type);
//   op->SetOutput("Out", {outputs[0]});
//   if (type == "dropout") {
//     op->SetInput("X", {inputs[0]});
//   } else if (type == "fc") {
//     op->SetInput("Input", {inputs[0]});
//     op->SetInput("W", {inputs[1]});
//     op->SetInput("Bias", {inputs[2]});
//     op->SetAttr("use_mkldnn", use_mkldnn);
//     op->SetAttr("in_num_col_dims", 2);
//   } else if (type == "matmul") {
//     op->SetInput("X", {inputs[0]});
//     op->SetInput("Y", {inputs[1]});
//     op->SetAttr("use_mkldnn", use_mkldnn);
//   }
// }

// // a -> dropout -> b
// // (b, w1, b1) -> fc -> c
// // (b, w2, b2) -> fc -> d
// // (b, w3, b3) -> fc -> e
// // (c, d) -> matmul -> f
// // (e, g) -> matmul -> h
// ProgramDesc BuildProgramDesc() {
//   ProgramDesc prog;
//   for (auto &v : std::initializer_list<std::string>(
//            {"a", "b", "c", "d", "e", "f", "g", "h", "w1", "w2", "w3", "b1",
//            "b2", "b3"})) {
//     auto *var = prog.MutableBlock(0)->Var(v);
//     var->SetType(proto::VarType::SELECTED_ROWS);
//   }

//   SetOp(&prog, "dropout", {"a"}, {"b"}, true);
//   SetOp(&prog, "fc", {"b", "w1", "b1"}, {"c"}, true);
//   SetOp(&prog, "fc", {"b", "w2", "b2"}, {"d"}, true);
//   SetOp(&prog, "fc", {"b", "w3" "b3"}, {"e"}, true);
//   SetOp(&prog, "matmul", {"c", "d"}, {"f"}, true);
//   SetOp(&prog, "matmul", {"e", "g"}, {"h"}, true);

//   return prog;
// }

// void MainTest(const ProgramDesc &prog) {
//   std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));

//   int original_nodes_num = graph->Nodes().size();

//   VisualizeGraph(&graph, "fc_parallel_mkldnn_fuse_pass_tester_before");

//   auto pass =
//       PassRegistry::Instance().Get("fc_parallel_mkldnn_fuse_pass");
//   graph.reset(pass->Apply(graph.release()));

//   VisualizeGraph(&graph, "fc_parallel_mkldnn_fuse_pass_tester_after");

//   int current_nodes_num = graph->Nodes().size();

//   // Removed nodes: 2 * fc
//   // Added nodes: split
//   EXPECT_EQ(original_nodes_num - 1, current_nodes_num);

//   for (auto *node : graph->Nodes()) {
//     if (node->IsOp()) {
//       auto *op = node->Op();
//       if (op->Type() == "matmul") {
//       } else if (op->Type() == "split") {
//       }
//     }
//   }
// }

// TEST(FcParallelMkldnnFusePass, Dropout_3FCs_2MatMuls) {
//   auto prog = BuildProgramDesc();
//   MainTest(prog);
// }

std::vector<int> CheckBias(float* w_ptr, int width) {
  float(&bias_tensor_matrix)[3][width] =
      *reinterpret_cast<float(*)[3][width]>(w_ptr);
  std::vector<int> block_order{
      static_cast<int>(bias_tensor_matrix[0][0]) / width,
      static_cast<int>(bias_tensor_matrix[1][0]) / width,
      static_cast<int>(bias_tensor_matrix[2][0]) / width};
  VLOG(1) << "Printing the block order: " << block_order[0] << " "
          << block_order[1] << " " << block_order[2];
  int i = 0;
  for (int block : block_order) {
    for (int col = 0; col < width; ++col) {
      EXPECT_EQ(static_cast<int>(bias_tensor_matrix[block][col]), i++);
    }
  }
  return block_order;
}

void CheckWeights(float* w_ptr, int width,
                  const std::vector<int>& block_order) {
  float(&weights_matrix)[width][3][width] =
      *reinterpret_cast<float(*)[width][3][width]>(w_ptr);
  int i = 0;
  for (int block : block_order) {
    for (int row = 0; row < width; ++row) {
      for (int col = 0; col < width; ++col) {
        EXPECT_EQ(static_cast<int>(weights_matrix[row][block][col]), i++)
            << "row: " << row << ", block: " << block << ", col: " << col;
      }
    }
  }
}

void CheckWeightsPadded(float* w_ptr, int width,
                        const std::vector<int>& block_order) {
  float(&weights_matrix)[width - 4][3][width - 4] =
      *reinterpret_cast<float(*)[width - 4][3][width - 4]>(w_ptr);
  int block_counter = 0;
  for (int block : block_order) {
    for (int row = 0; row < width - 4; ++row) {
      for (int col = 0; col < width - 4; ++col) {
        EXPECT_EQ(static_cast<int>(weights_matrix[row][block][col]),
                  (width * width) * block_counter + width * row + col)
            << "row: " << row << ", block: " << block
            << ", block_counter: " << block_counter << ", col: " << col;
      }
    }
    ++block_counter;
  }
}

void TestMain(bool padding_weights) {
  // inputs                           operator            output
  // ------------------------------------------------------------------
  // a                                dropout        ->   b
  // (b, w1, b1)                      fc             ->   c
  // (b, w2, b2)                      fc             ->   d
  // (b, w3, b3)                      fc             ->   e
  // (c, d)                           matmul         ->   f
  // (e, g)                           matmul         ->   h
  int width = 6;
  int height = 4;
  Layers layers;
  auto* a = layers.data("a", {2, height, width});
  auto* b = layers.dropout(a, 0.1, "downgrade_in_infer");
  b->SetShape({2, height, width});

  int in_num_col_dims = 2;
  std::string activation_type = "";
  auto* w1 = layers.data("w1", {width, width}, true);
  auto* b1 = layers.data("b1", {width}, true);
  VarDesc* c =
      layers.fc(b, w1, b1, in_num_col_dims, activation_type, padding_weights);
  c->SetShape({2, height, width});

  auto* w2 = layers.data("w2", {width, width}, true);
  auto* b2 = layers.data("b2", {width}, true);
  VarDesc* d =
      layers.fc(b, w2, b2, in_num_col_dims, activation_type, padding_weights);
  d->SetShape({2, height, width});

  auto* w3 = layers.data("w3", {width, width}, true);
  auto* b3 = layers.data("b3", {width}, true);
  VarDesc* e =
      layers.fc(b, w3, b3, in_num_col_dims, activation_type, padding_weights);
  e->SetShape({2, height, width});

  layers.matmul(c, d);
  auto* g = layers.data("g", {2, height, width});
  layers.matmul(e, g);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto* scope = CreateParamScope(width);
  graph->Set("__param_scope__", scope);
  auto pass = PassRegistry::Instance().Get("fc_parallel_mkldnn_fuse_pass");

  int num_fc_nodes_before = GetNumOpNodes(graph, "fc");
  int original_nodes_num = graph->Nodes().size();

  graph.reset(pass->Apply(graph.release()));
  auto fc_ops = GetOpNodes(graph, "fc");
  int num_fc_nodes_after = fc_ops.size();
  int current_nodes_num = graph->Nodes().size();

  EXPECT_EQ(num_fc_nodes_before, 3);
  EXPECT_EQ(num_fc_nodes_after, 1);
  // removed 3 * 3 (fc,w,b) = 9
  // added 1 * 3 (fc,w,b) + 2 (split,input) = 5
  EXPECT_EQ(original_nodes_num - 4, current_nodes_num);
  auto split_ops = GetOpNodes(graph, "split");
  ASSERT_EQ(split_ops.size(), 1UL);
  EXPECT_THAT(split_ops[0]->Op()->GetAttrIfExists<int>("num"), 3);

  // Order of 3 fc ops matched by fuse might be non-deterministic,
  // so infer order from values and store in block_order.

  auto block_order = CheckBias(scope->FindVar(fc_ops[0]->Op()->Input("Bias")[0])
                                   ->GetMutable<LoDTensor>()
                                   ->data<float>(),
                               width);

  if (padding_weights) {
    CheckWeightsPadded(scope->FindVar(fc_ops[0]->Op()->Input("W")[0])
                           ->GetMutable<LoDTensor>()
                           ->data<float>(),
                       width, block_order);
  } else {
    CheckWeights(scope->FindVar(fc_ops[0]->Op()->Input("W")[0])
                     ->GetMutable<LoDTensor>()
                     ->data<float>(),
                 width, block_order);
  }
}

TEST(FcParallelMkldnnFusePass, Dropout_3FCs_2MatMuls) { TestMain(false); }
TEST(FcParallelMkldnnFusePass, Dropout_3FCs_2MatMuls_padding_weights) {
  TestMain(true);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(fc_parallel_mkldnn_fuse_pass);
