// Copyright 2024 YOLOs-CPP Team
// SPDX-License-Identifier: AGPL-3.0

#ifndef ROS2_YOLOS_CPP__ADAPTERS__YOLOS_ADAPTER_BASE_HPP_
#define ROS2_YOLOS_CPP__ADAPTERS__YOLOS_ADAPTER_BASE_HPP_

// =============================================================================
// Abstract Adapter Base Interfaces
// =============================================================================
// Define stable interfaces that decouple ROS 2 nodes from YOLOs-CPP library.
// When YOLOs-CPP v2 releases, only adapter implementations need updating.
// =============================================================================

#include <memory>
#include <string>
#include <vector>
#include <optional>

#include <opencv2/opencv.hpp>

#include "ros2_yolos_cpp/visibility_control.hpp"

namespace ros2_yolos_cpp {

// =============================================================================
// Configuration Structs (Library-agnostic)
// =============================================================================

/// @brief Common configuration for all YOLO adapters
struct YolosConfig {
  std::string model_path;
  std::string labels_path;
  bool use_gpu{false};
  bool use_tensorrt{false};  // TensorRT execution provider (with use_gpu), CUDA as fallback
  bool tensorrt_fp16{true};
  std::string tensorrt_cache_path;  // Engine cache directory, empty: no cache
  float conf_threshold{0.4f};
  float nms_threshold{0.45f};
  std::string yolo_version{"auto"};  // auto, v7, v8, v10, v11, v26, nas
};

// =============================================================================
// Result Structs (Library-agnostic DTOs)
// =============================================================================
// These are our own result types, not depending on yolos:: types directly.
// Converters will map these to ROS 2 messages.

/// @brief Axis-aligned bounding box
struct BoundingBox2D {
  int x{0};
  int y{0};
  int width{0};
  int height{0};
};

/// @brief Oriented (rotated) bounding box
struct OrientedBoundingBox2D {
  float center_x{0.0f};
  float center_y{0.0f};
  float width{0.0f};
  float height{0.0f};
  float angle{0.0f};  // radians
};

/// @brief Keypoint for pose estimation
struct KeyPoint2D {
  float x{0.0f};
  float y{0.0f};
  float confidence{0.0f};
  int id{0};
};

/// @brief Detection result
struct DetectionResult {
  BoundingBox2D bbox;
  float confidence{0.0f};
  int class_id{0};
  std::string class_name;
};

/// @brief Segmentation result
struct SegmentationResult {
  BoundingBox2D bbox;
  float confidence{0.0f};
  int class_id{0};
  std::string class_name;
  cv::Mat mask;  // Binary mask
};

/// @brief Pose estimation result
struct PoseResult {
  BoundingBox2D bbox;
  float confidence{0.0f};
  int class_id{0};
  std::string class_name;
  std::vector<KeyPoint2D> keypoints;
};

/// @brief OBB detection result
struct OBBResult {
  OrientedBoundingBox2D bbox;
  float confidence{0.0f};
  int class_id{0};
  std::string class_name;
};

/// @brief Classification result
struct ClassificationResult {
  int class_id{0};
  float confidence{0.0f};
  std::string class_name;
};

// =============================================================================
// Abstract Adapter Interfaces
// =============================================================================

/// @brief Abstract base for all YOLO adapters
class ROS2_YOLOS_CPP_PUBLIC IYolosAdapterBase {
public:
  virtual ~IYolosAdapterBase() = default;

  /// @brief Initialize the adapter with configuration
  /// @return true on success, false on failure
  virtual bool initialize(const YolosConfig& config) = 0;

  /// @brief Check if adapter is initialized
  [[nodiscard]] virtual bool isInitialized() const noexcept = 0;

  /// @brief Release resources
  virtual void shutdown() = 0;

  /// @brief Get class names loaded from labels file
  [[nodiscard]] virtual const std::vector<std::string>& getClassNames() const = 0;
};

/// @brief Detection adapter interface
class ROS2_YOLOS_CPP_PUBLIC IDetectorAdapter : public IYolosAdapterBase {
public:
  virtual std::vector<DetectionResult> detect(
    const cv::Mat& image,
    float conf_threshold,
    float nms_threshold) = 0;

  /// @brief Draw detections on image (optional debug)
  virtual void drawDetections(
    cv::Mat& image,
    const std::vector<DetectionResult>& detections) = 0;
};

/// @brief Segmentation adapter interface
class ROS2_YOLOS_CPP_PUBLIC ISegmentorAdapter : public IYolosAdapterBase {
public:
  virtual std::vector<SegmentationResult> segment(
    const cv::Mat& image,
    float conf_threshold,
    float nms_threshold) = 0;

  virtual void drawSegmentations(
    cv::Mat& image,
    const std::vector<SegmentationResult>& segmentations,
    float mask_alpha = 0.5f) = 0;
};

/// @brief Pose estimation adapter interface
class ROS2_YOLOS_CPP_PUBLIC IPoseAdapter : public IYolosAdapterBase {
public:
  virtual std::vector<PoseResult> detect(
    const cv::Mat& image,
    float conf_threshold,
    float nms_threshold) = 0;

  virtual void drawPoses(
    cv::Mat& image,
    const std::vector<PoseResult>& poses,
    int kpt_radius = 4,
    float kpt_threshold = 0.5f) = 0;
};

/// @brief OBB detection adapter interface
class ROS2_YOLOS_CPP_PUBLIC IOBBAdapter : public IYolosAdapterBase {
public:
  virtual std::vector<OBBResult> detect(
    const cv::Mat& image,
    float conf_threshold,
    float nms_threshold,
    int max_det = 300) = 0;

  virtual void drawDetections(
    cv::Mat& image,
    const std::vector<OBBResult>& detections) = 0;
};

/// @brief Classification adapter interface
class ROS2_YOLOS_CPP_PUBLIC IClassifierAdapter : public IYolosAdapterBase {
public:
  virtual ClassificationResult classify(const cv::Mat& image) = 0;

  virtual void drawResult(
    cv::Mat& image,
    const ClassificationResult& result) = 0;
};

// =============================================================================
// Factory Functions
// =============================================================================

ROS2_YOLOS_CPP_PUBLIC
std::unique_ptr<IDetectorAdapter> createDetectorAdapter();

ROS2_YOLOS_CPP_PUBLIC
std::unique_ptr<ISegmentorAdapter> createSegmentorAdapter();

ROS2_YOLOS_CPP_PUBLIC
std::unique_ptr<IPoseAdapter> createPoseAdapter();

ROS2_YOLOS_CPP_PUBLIC
std::unique_ptr<IOBBAdapter> createOBBAdapter();

ROS2_YOLOS_CPP_PUBLIC
std::unique_ptr<IClassifierAdapter> createClassifierAdapter();

}  // namespace ros2_yolos_cpp

#endif  // ROS2_YOLOS_CPP__ADAPTERS__YOLOS_ADAPTER_BASE_HPP_
