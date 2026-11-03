// Copyright 2024 YOLOs-CPP Team
// SPDX-License-Identifier: AGPL-3.0

#include "ros2_yolos_cpp/adapters/pose_adapter.hpp"
#include "yolos/yolos.hpp"
#include "tensorrt_options.hpp"
#include <iostream>

namespace ros2_yolos_cpp {

class PoseAdapter::Impl {
public:
  std::unique_ptr<yolos::pose::YOLOPoseDetector> detector;
  std::vector<std::string> class_names;
  bool initialized{false};
};

PoseAdapter::PoseAdapter() : impl_(std::make_unique<Impl>()) {}
PoseAdapter::~PoseAdapter() = default;
PoseAdapter::PoseAdapter(PoseAdapter&&) noexcept = default;
PoseAdapter& PoseAdapter::operator=(PoseAdapter&&) noexcept = default;

bool PoseAdapter::initialize(const YolosConfig& config) {
  try {
    applyTensorRTOptions(config, "PoseAdapter");
    impl_->detector = std::make_unique<yolos::pose::YOLOPoseDetector>(
      config.model_path,
      config.labels_path,
      config.use_gpu
    );
    impl_->class_names = {"person"};  // Pose detection typically for person class
    impl_->initialized = true;
    std::cout << "[PoseAdapter] Initialized" << std::endl;
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[PoseAdapter] Error: " << e.what() << std::endl;
    return false;
  }
}

bool PoseAdapter::isInitialized() const noexcept {
  return impl_ && impl_->initialized;
}

void PoseAdapter::shutdown() {
  if (impl_) {
    impl_->detector.reset();
    impl_->initialized = false;
  }
}

const std::vector<std::string>& PoseAdapter::getClassNames() const {
  static const std::vector<std::string> empty;
  return impl_ ? impl_->class_names : empty;
}

std::vector<PoseResult> PoseAdapter::detect(
  const cv::Mat& image,
  float conf_threshold,
  float nms_threshold)
{
  std::vector<PoseResult> results;
  if (!isInitialized() || image.empty()) return results;

  try {
    auto poses = impl_->detector->detect(image, conf_threshold, nms_threshold);
    results.reserve(poses.size());

    for (const auto& pose : poses) {
      PoseResult result;
      result.bbox.x = pose.box.x;
      result.bbox.y = pose.box.y;
      result.bbox.width = pose.box.width;
      result.bbox.height = pose.box.height;
      result.confidence = pose.conf;
      result.class_id = pose.classId;
      result.class_name = "person";

      // Convert keypoints
      result.keypoints.reserve(pose.keypoints.size());
      int kpt_id = 0;
      for (const auto& kpt : pose.keypoints) {
        KeyPoint2D kp;
        kp.x = kpt.x;
        kp.y = kpt.y;
        kp.confidence = kpt.confidence;
        kp.id = kpt_id++;
        result.keypoints.push_back(kp);
      }

      results.push_back(std::move(result));
    }
  } catch (const std::exception& e) {
    std::cerr << "[PoseAdapter] Detection error: " << e.what() << std::endl;
  }

  return results;
}

void PoseAdapter::drawPoses(
  cv::Mat& image,
  const std::vector<PoseResult>& poses,
  int kpt_radius,
  float kpt_threshold)
{
  if (!isInitialized() || image.empty()) return;

  std::vector<yolos::pose::PoseResult> yolos_poses;
  yolos_poses.reserve(poses.size());

  for (const auto& pose : poses) {
    yolos::BoundingBox box(pose.bbox.x, pose.bbox.y, pose.bbox.width, pose.bbox.height);
    std::vector<yolos::KeyPoint> kpts;
    kpts.reserve(pose.keypoints.size());
    for (const auto& kp : pose.keypoints) {
      kpts.emplace_back(kp.x, kp.y, kp.confidence);
    }
    yolos_poses.emplace_back(box, pose.confidence, pose.class_id, kpts);
  }

  impl_->detector->drawPoses(image, yolos_poses, kpt_radius, kpt_threshold);
}

std::unique_ptr<IPoseAdapter> createPoseAdapter() {
  return std::make_unique<PoseAdapter>();
}

}  // namespace ros2_yolos_cpp
