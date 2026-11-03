// Copyright 2024 YOLOs-CPP Team
// SPDX-License-Identifier: AGPL-3.0

#include "ros2_yolos_cpp/adapters/segmentor_adapter.hpp"
#include "yolos/yolos.hpp"
#include "tensorrt_options.hpp"

#include <iostream>

namespace ros2_yolos_cpp {

class SegmentorAdapter::Impl {
public:
  std::unique_ptr<yolos::seg::YOLOSegDetector> segmentor;
  std::vector<std::string> class_names;
  bool initialized{false};
};

SegmentorAdapter::SegmentorAdapter() : impl_(std::make_unique<Impl>()) {}
SegmentorAdapter::~SegmentorAdapter() = default;
SegmentorAdapter::SegmentorAdapter(SegmentorAdapter&&) noexcept = default;
SegmentorAdapter& SegmentorAdapter::operator=(SegmentorAdapter&&) noexcept = default;

bool SegmentorAdapter::initialize(const YolosConfig& config) {
  try {
    applyTensorRTOptions(config, "SegmentorAdapter");
    impl_->segmentor = std::make_unique<yolos::seg::YOLOSegDetector>(
      config.model_path,
      config.labels_path,
      config.use_gpu
    );

    impl_->class_names = impl_->segmentor->getClassNames();
    impl_->initialized = true;
    std::cout << "[SegmentorAdapter] Initialized with " 
              << impl_->class_names.size() << " classes" << std::endl;
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[SegmentorAdapter] Error: " << e.what() << std::endl;
    return false;
  }
}

bool SegmentorAdapter::isInitialized() const noexcept {
  return impl_ && impl_->initialized;
}

void SegmentorAdapter::shutdown() {
  if (impl_) {
    impl_->segmentor.reset();
    impl_->initialized = false;
  }
}

const std::vector<std::string>& SegmentorAdapter::getClassNames() const {
  static const std::vector<std::string> empty;
  return impl_ ? impl_->class_names : empty;
}

std::vector<SegmentationResult> SegmentorAdapter::segment(
  const cv::Mat& image,
  float conf_threshold,
  float nms_threshold)
{
  std::vector<SegmentationResult> results;
  if (!isInitialized() || image.empty()) return results;

  try {
    auto segs = impl_->segmentor->segment(image, conf_threshold, nms_threshold);
    results.reserve(segs.size());

    for (const auto& seg : segs) {
      SegmentationResult result;
      result.bbox.x = seg.box.x;
      result.bbox.y = seg.box.y;
      result.bbox.width = seg.box.width;
      result.bbox.height = seg.box.height;
      result.confidence = seg.conf;
      result.class_id = seg.classId;
      result.mask = seg.mask.clone();

      if (seg.classId >= 0 && static_cast<size_t>(seg.classId) < impl_->class_names.size()) {
        result.class_name = impl_->class_names[seg.classId];
      }
      results.push_back(std::move(result));
    }
  } catch (const std::exception& e) {
    std::cerr << "[SegmentorAdapter] Segmentation error: " << e.what() << std::endl;
  }

  return results;
}

void SegmentorAdapter::drawSegmentations(
  cv::Mat& image,
  const std::vector<SegmentationResult>& segmentations,
  float mask_alpha)
{
  if (!isInitialized() || image.empty()) return;

  std::vector<yolos::seg::Segmentation> yolos_segs;
  yolos_segs.reserve(segmentations.size());

  for (const auto& seg : segmentations) {
    yolos::BoundingBox box(seg.bbox.x, seg.bbox.y, seg.bbox.width, seg.bbox.height);
    yolos_segs.emplace_back(box, seg.confidence, seg.class_id, seg.mask);
  }

  impl_->segmentor->drawSegmentations(image, yolos_segs, mask_alpha);
}

std::unique_ptr<ISegmentorAdapter> createSegmentorAdapter() {
  return std::make_unique<SegmentorAdapter>();
}

}  // namespace ros2_yolos_cpp
