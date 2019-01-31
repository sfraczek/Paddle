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

/*
 * This file contains the definition of a simple Inference API for Paddle.
 *
 * ATTENTION: It requires some C++11 features, for lower version C++ or C, we
 * might release another API.
 */

#pragma once

#include <cassert>
#include <memory>
#include <string>
#include <vector>

#include "paddle_analysis_config.h"  // NOLINT
#include "paddle_api.h"              // NOLINT
#include "paddle_quantize_config.h"  // NOLINT
#ifdef WITH_ANAKIN
#include "paddle_anakin_config.h"  // NOLINT
#endif
