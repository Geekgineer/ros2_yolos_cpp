// Copyright 2024 YOLOs-CPP Team
// SPDX-License-Identifier: AGPL-3.0

#include "ros2_yolos_cpp/adapters/detector_adapter.hpp"

// YOLOs-CPP includes - ONLY in implementation file
#include "yolos/yolos.hpp"
#include "tensorrt_options.hpp"

#include <stdexcept>
#include <iostream>

namespace ros2_yolos_cpp {

// =============================================================================
// PIMPL Implementation
// =============================================================================
class DetectorAdapter::Impl {
public:
  std::unique_ptr<yolos::det::YOLODetector> detector;
  std::vector<std::string> class_names;
  bool initialized{false};

  static yolos::YOLOVersion parseVersion(const std::string& version_str) {
    if (version_str == "v7") return yolos::YOLOVersion::V7;
    if (version_str == "v8") return yolos::YOLOVersion::V8;
    if (version_str == "v10") return yolos::YOLOVersion::V10;
    if (version_str == "v11") return yolos::YOLOVersion::V11;
    if (version_str == "v26") return yolos::YOLOVersion::V26;
    if (version_str == "nas") return yolos::YOLOVersion::NAS;
    return yolos::YOLOVersion::Auto;
  }
};

// =============================================================================
// DetectorAdapter Implementation
// =============================================================================

DetectorAdapter::DetectorAdapter()
  : impl_(std::make_unique<Impl>()) {}

DetectorAdapter::~DetectorAdapter() = default;

DetectorAdapter::DetectorAdapter(DetectorAdapter&&) noexcept = default;
DetectorAdapter& DetectorAdapter::operator=(DetectorAdapter&&) noexcept = default;

bool DetectorAdapter::initialize(const YolosConfig& config) {
  try {
    applyTensorRTOptions(config, "DetectorAdapter");
    auto version = Impl::parseVersion(config.yolo_version);
    
    impl_->detector = yolos::det::createDetector(
      config.model_path,
      config.labels_path,
      version,
      config.use_gpu
    );

    if (!impl_->detector) {
      std::cerr << "[DetectorAdapter] Failed to create detector" << std::endl;
      return false;
    }

    impl_->class_names = impl_->detector->getClassNames();
    impl_->initialized = true;

    std::cout << "[DetectorAdapter] Initialized with " 
              << impl_->class_names.size() << " classes" << std::endl;
    
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[DetectorAdapter] Initialization error: " << e.what() << std::endl;
    impl_->initialized = false;
    return false;
  }
}

bool DetectorAdapter::isInitialized() const noexcept {
  return impl_ && impl_->initialized;
}

void DetectorAdapter::shutdown() {
  if (impl_) {
    impl_->detector.reset();
    impl_->initialized = false;
  }
}

const std::vector<std::string>& DetectorAdapter::getClassNames() const {
  static const std::vector<std::string> empty;
  return impl_ ? impl_->class_names : empty;
}

std::vector<DetectionResult> DetectorAdapter::detect(
  const cv::Mat& image,
  float conf_threshold,
  float nms_threshold)
{
  std::vector<DetectionResult> results;

  if (!isInitialized() || image.empty()) {
    return results;
  }

  try {
    // Call YOLOs-CPP detection
    auto detections = impl_->detector->detect(image, conf_threshold, nms_threshold);

    // Convert to our adapter result type
    results.reserve(detections.size());
    for (const auto& det : detections) {
      DetectionResult result;
      result.bbox.x = det.box.x;
      result.bbox.y = det.box.y;
      result.bbox.width = det.box.width;
      result.bbox.height = det.box.height;
      result.confidence = det.conf;
      result.class_id = det.classId;

      if (det.classId >= 0 && 
          static_cast<size_t>(det.classId) < impl_->class_names.size()) {
        result.class_name = impl_->class_names[det.classId];
      }

      results.push_back(std::move(result));
    }
  } catch (const std::exception& e) {
    std::cerr << "[DetectorAdapter] Detection error: " << e.what() << std::endl;
  }

  return results;
}

void DetectorAdapter::drawDetections(
  cv::Mat& image,
  const std::vector<DetectionResult>& detections)
{
  if (!isInitialized() || image.empty()) return;

  // Convert back to yolos format for drawing
  std::vector<yolos::det::Detection> yolos_dets;
  yolos_dets.reserve(detections.size());

  for (const auto& det : detections) {
    yolos::BoundingBox box(
      det.bbox.x, det.bbox.y, 
      det.bbox.width, det.bbox.height
    );
    yolos_dets.emplace_back(box, det.confidence, det.class_id);
  }

  impl_->detector->drawDetections(image, yolos_dets);
}

// =============================================================================
// Factory Function
// =============================================================================
std::unique_ptr<IDetectorAdapter> createDetectorAdapter() {
  return std::make_unique<DetectorAdapter>();
}

}  // namespace ros2_yolos_cpp
