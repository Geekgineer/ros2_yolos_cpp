// Copyright 2024 YOLOs-CPP Team
// SPDX-License-Identifier: AGPL-3.0

#ifndef ADAPTERS__TENSORRT_OPTIONS_HPP_
#define ADAPTERS__TENSORRT_OPTIONS_HPP_

#include <iostream>

#include "ros2_yolos_cpp/adapters/yolos_adapter_base.hpp"
#include "yolos/yolos.hpp"

namespace ros2_yolos_cpp {

/// @brief Passes the TensorRT settings of the config to YOLOs-CPP. Call it before creating a
/// YOLOs-CPP session. Without TensorRT support in YOLOs-CPP (see CMakeLists.txt), sessions keep
/// using CUDA or the CPU.
inline void applyTensorRTOptions(const YolosConfig& config, const char* adapter) {
#ifdef YOLOS_CPP_HAS_TENSORRT
  yolos::tensorRTOptions() = {
    config.use_tensorrt, config.tensorrt_fp16, config.tensorrt_cache_path};
#else
  if (config.use_tensorrt) {
    std::cerr << "[" << adapter << "] use_tensorrt is set, "
              << "but YOLOs-CPP was built without TensorRT support" << std::endl;
  }
#endif
}

}  // namespace ros2_yolos_cpp

#endif  // ADAPTERS__TENSORRT_OPTIONS_HPP_
