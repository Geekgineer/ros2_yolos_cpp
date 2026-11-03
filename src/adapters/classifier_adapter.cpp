// Copyright 2024 YOLOs-CPP Team
// SPDX-License-Identifier: AGPL-3.0

#include "ros2_yolos_cpp/adapters/classifier_adapter.hpp"
#include "yolos/yolos.hpp"
#include "tensorrt_options.hpp"
#include <iostream>

namespace ros2_yolos_cpp {

class ClassifierAdapter::Impl {
public:
  std::unique_ptr<yolos::cls::YOLOClassifier> classifier;
  std::vector<std::string> class_names;
  bool initialized{false};
};

ClassifierAdapter::ClassifierAdapter() : impl_(std::make_unique<Impl>()) {}
ClassifierAdapter::~ClassifierAdapter() = default;
ClassifierAdapter::ClassifierAdapter(ClassifierAdapter&&) noexcept = default;
ClassifierAdapter& ClassifierAdapter::operator=(ClassifierAdapter&&) noexcept = default;

bool ClassifierAdapter::initialize(const YolosConfig& config) {
  try {
    applyTensorRTOptions(config, "ClassifierAdapter");
    impl_->classifier = std::make_unique<yolos::cls::YOLOClassifier>(
      config.model_path,
      config.labels_path,
      config.use_gpu
    );
    impl_->class_names = impl_->classifier->getClassNames();
    impl_->initialized = true;
    std::cout << "[ClassifierAdapter] Initialized with " 
              << impl_->class_names.size() << " classes" << std::endl;
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[ClassifierAdapter] Error: " << e.what() << std::endl;
    return false;
  }
}

bool ClassifierAdapter::isInitialized() const noexcept {
  return impl_ && impl_->initialized;
}

void ClassifierAdapter::shutdown() {
  if (impl_) {
    impl_->classifier.reset();
    impl_->initialized = false;
  }
}

const std::vector<std::string>& ClassifierAdapter::getClassNames() const {
  static const std::vector<std::string> empty;
  return impl_ ? impl_->class_names : empty;
}

ClassificationResult ClassifierAdapter::classify(const cv::Mat& image) {
  ClassificationResult result;
  if (!isInitialized() || image.empty()) return result;

  try {
    auto yolos_result = impl_->classifier->classify(image);
    result.class_id = yolos_result.classId;
    result.confidence = yolos_result.confidence;
    result.class_name = yolos_result.className;
  } catch (const std::exception& e) {
    std::cerr << "[ClassifierAdapter] Classification error: " << e.what() << std::endl;
  }

  return result;
}

void ClassifierAdapter::drawResult(
  cv::Mat& image,
  const ClassificationResult& result)
{
  if (!isInitialized() || image.empty()) return;

  yolos::cls::ClassificationResult yolos_result(
    result.class_id, result.confidence, result.class_name
  );
  impl_->classifier->drawResult(image, yolos_result);
}

std::unique_ptr<IClassifierAdapter> createClassifierAdapter() {
  return std::make_unique<ClassifierAdapter>();
}

}  // namespace ros2_yolos_cpp
