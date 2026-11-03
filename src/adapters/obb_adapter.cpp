// Copyright 2024 YOLOs-CPP Team
// SPDX-License-Identifier: AGPL-3.0

#include "ros2_yolos_cpp/adapters/obb_adapter.hpp"
#include "yolos/yolos.hpp"
#include "tensorrt_options.hpp"
#include <iostream>

namespace ros2_yolos_cpp {

class OBBAdapter::Impl {
public:
  std::unique_ptr<yolos::obb::YOLOOBBDetector> detector;
  std::vector<std::string> class_names;
  bool initialized{false};
};

OBBAdapter::OBBAdapter() : impl_(std::make_unique<Impl>()) {}
OBBAdapter::~OBBAdapter() = default;
OBBAdapter::OBBAdapter(OBBAdapter&&) noexcept = default;
OBBAdapter& OBBAdapter::operator=(OBBAdapter&&) noexcept = default;

bool OBBAdapter::initialize(const YolosConfig& config) {
  try {
    applyTensorRTOptions(config, "OBBAdapter");
    impl_->detector = std::make_unique<yolos::obb::YOLOOBBDetector>(
      config.model_path,
      config.labels_path,
      config.use_gpu
    );
    impl_->class_names = impl_->detector->getClassNames();
    impl_->initialized = true;
    std::cout << "[OBBAdapter] Initialized with " 
              << impl_->class_names.size() << " classes" << std::endl;
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[OBBAdapter] Error: " << e.what() << std::endl;
    return false;
  }
}

bool OBBAdapter::isInitialized() const noexcept {
  return impl_ && impl_->initialized;
}

void OBBAdapter::shutdown() {
  if (impl_) {
    impl_->detector.reset();
    impl_->initialized = false;
  }
}

const std::vector<std::string>& OBBAdapter::getClassNames() const {
  static const std::vector<std::string> empty;
  return impl_ ? impl_->class_names : empty;
}

std::vector<OBBResult> OBBAdapter::detect(
  const cv::Mat& image,
  float conf_threshold,
  float nms_threshold,
  int max_det)
{
  std::vector<OBBResult> results;
  if (!isInitialized() || image.empty()) return results;

  try {
    auto dets = impl_->detector->detect(image, conf_threshold, nms_threshold, max_det);
    results.reserve(dets.size());

    for (const auto& det : dets) {
      OBBResult result;
      result.bbox.center_x = det.box.x;
      result.bbox.center_y = det.box.y;
      result.bbox.width = det.box.width;
      result.bbox.height = det.box.height;
      result.bbox.angle = det.box.angle;
      result.confidence = det.conf;
      result.class_id = det.classId;

      if (det.classId >= 0 && static_cast<size_t>(det.classId) < impl_->class_names.size()) {
        result.class_name = impl_->class_names[det.classId];
      }
      results.push_back(std::move(result));
    }
  } catch (const std::exception& e) {
    std::cerr << "[OBBAdapter] Detection error: " << e.what() << std::endl;
  }

  return results;
}

void OBBAdapter::drawDetections(
  cv::Mat& image,
  const std::vector<OBBResult>& detections)
{
  if (!isInitialized() || image.empty()) return;

  std::vector<yolos::obb::OBBResult> yolos_dets;
  yolos_dets.reserve(detections.size());

  for (const auto& det : detections) {
    yolos::OrientedBoundingBox box(
      det.bbox.center_x, det.bbox.center_y,
      det.bbox.width, det.bbox.height, det.bbox.angle
    );
    yolos_dets.emplace_back(box, det.confidence, det.class_id);
  }

  impl_->detector->drawDetections(image, yolos_dets);
}

std::unique_ptr<IOBBAdapter> createOBBAdapter() {
  return std::make_unique<OBBAdapter>();
}

}  // namespace ros2_yolos_cpp
