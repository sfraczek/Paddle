// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/api/quantizer_predictor.h"
#include <glog/logging.h>
#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/feed_fetch_method.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/var_type_traits.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/analysis/passes/memory_optimize_pass.h"
#include "paddle/fluid/inference/analysis/quantizer.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/api/paddle_inference_pass.h"
#include "paddle/fluid/inference/utils/singleton.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/platform/profiler.h"

#if PADDLE_WITH_TENSORRT
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/trt_int8_calibrator.h"

#endif

DECLARE_bool(profile);

namespace paddle {

using inference::Singleton;
#if PADDLE_WITH_TENSORRT
using inference::tensorrt::TRTInt8Calibrator;
using inference::tensorrt::TRTCalibratorEngine;
using inference::tensorrt::TRTCalibratorEngineManager;
#endif
using inference::analysis::Quantizer;

bool QuantizerPredictor::Quantize() {
  if (config_.quantizer_enabled()) {
    auto predictor_run =
        std::bind(&QuantizerPredictor::Run, this, std::placeholders::_1,
                  std::placeholders::_2, std::placeholders::_3);
    framework::Scope *scope = sub_scope_ ? sub_scope_ : scope_.get();
    // initialize quantizer
    quantizer_.reset(new Quantizer(scope, inference_program_, config_,
                                   argument_, predictor_run));
    // do the quantization
    if (!quantizer_->Quantize()) return false;
  }

  return true;
}

// NOTE All the members in QuantizerConfig should be copied to Argument.
void QuantizerPredictor::PrepareArgument() {
  AnalysisPredictor::PrepareArgument();

  if (config_.quantizer_enabled()) {
    LOG(INFO) << "quantization is enabled";
    argument_.SetQuantizeEnabledOpTypes(config_.enabled_op_types());
  }
}

template <>
std::unique_ptr<PaddlePredictor>
CreatePaddlePredictor<QuantizerConfig, PaddleEngineKind::kAnalysis>(
    const QuantizerConfig &config) {
  VLOG(3) << "create QuantizerConfig";
  if (config.use_gpu()) {
    // 1. GPU memory
    PADDLE_ENFORCE_GT(config.memory_pool_init_size_mb(), 0.f);
    PADDLE_ENFORCE_GE(config.gpu_device_id(), 0, "Invalid device id %d",
                      config.gpu_device_id());
    std::vector<std::string> flags;

    float fraction_of_gpu_memory = config.fraction_of_gpu_memory_for_pool();
    if (fraction_of_gpu_memory > 0.95f) {
      LOG(ERROR)
          << "Allocate too much memory for the GPU memory pool, assigned "
          << config.memory_pool_init_size_mb() << " MB";
      LOG(ERROR) << "Try to shink the value by setting "
                    "QuantizerConfig::EnableGpu(...)";
    }

    if (fraction_of_gpu_memory >= 0.0f || fraction_of_gpu_memory <= 0.95f) {
      flags.push_back("dummpy");
      std::string flag = "--fraction_of_gpu_memory_to_use=" +
                         std::to_string(fraction_of_gpu_memory);
      flags.push_back(flag);
      VLOG(3) << "set flag: " << flag;
      framework::InitGflags(flags);
    }
  }

  std::unique_ptr<PaddlePredictor> predictor(new QuantizerPredictor(config));
  auto predictor_p = dynamic_cast<QuantizerPredictor *>(predictor.get());

  if (!predictor_p->Init(nullptr)) {
    return nullptr;
  }

  if (config.quantizer_enabled()) {
    if (!predictor_p->Quantize()) return nullptr;
  }

  return std::move(predictor);
}

template <>
std::unique_ptr<PaddlePredictor> CreatePaddlePredictor<QuantizerConfig>(
    const QuantizerConfig &config) {
  return CreatePaddlePredictor<QuantizerConfig, PaddleEngineKind::kAnalysis>(
      config);
}

}  // namespace paddle
