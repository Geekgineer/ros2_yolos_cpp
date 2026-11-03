// Copyright 2024 YOLOs-CPP Team
// SPDX-License-Identifier: AGPL-3.0

#include "ros2_yolos_cpp/nodes/detector_node.hpp"
#include "ros2_yolos_cpp/conversion/detection_converter.hpp"
#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif
#include <chrono>

#include <lifecycle_msgs/msg/state.hpp>
#include <lifecycle_msgs/msg/transition.hpp>

namespace ros2_yolos_cpp {

YolosDetectorNode::YolosDetectorNode(const rclcpp::NodeOptions& options)
  : rclcpp_lifecycle::LifecycleNode("yolos_detector", options) {
  RCLCPP_INFO(get_logger(), "YolosDetectorNode created");
  declareParameters();

  // Lifecycle state to start in (lifecycle_msgs/State ids): 1 unconfigured, 2 inactive, 3 active
  const auto startup_state = get_parameter("startup_state").as_int();
  if (startup_state > lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED) {
    trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_CONFIGURE);
  }
  if (startup_state > lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE) {
    trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE);
  }
}

void YolosDetectorNode::declareParameters() {
  declare_parameter("model_path", rclcpp::PARAMETER_STRING);
  declare_parameter("labels_path", rclcpp::PARAMETER_STRING);
  declare_parameter("use_gpu", false);
  declare_parameter("use_tensorrt", false);
  declare_parameter("tensorrt_fp16", true);
  declare_parameter("tensorrt_cache_path", "");
  declare_parameter("conf_threshold", 0.4);
  declare_parameter("nms_threshold", 0.45);
  declare_parameter("yolo_version", "auto");
  declare_parameter("publish_debug_image", false);
  declare_parameter("publish_timing", false);
  declare_parameter("startup_state", static_cast<int64_t>(lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED));
}

YolosConfig YolosDetectorNode::loadConfig() {
  YolosConfig config;
  config.model_path = get_parameter("model_path").as_string();
  config.labels_path = get_parameter("labels_path").as_string();
  config.use_gpu = get_parameter("use_gpu").as_bool();
  config.use_tensorrt = get_parameter("use_tensorrt").as_bool();
  config.tensorrt_fp16 = get_parameter("tensorrt_fp16").as_bool();
  config.tensorrt_cache_path = get_parameter("tensorrt_cache_path").as_string();
  config.conf_threshold = static_cast<float>(get_parameter("conf_threshold").as_double());
  config.nms_threshold = static_cast<float>(get_parameter("nms_threshold").as_double());
  config.yolo_version = get_parameter("yolo_version").as_string();
  
  model_path_ = config.model_path;
  labels_path_ = config.labels_path;
  use_gpu_ = config.use_gpu;
  conf_threshold_ = config.conf_threshold;
  nms_threshold_ = config.nms_threshold;
  yolo_version_ = config.yolo_version;
  publish_debug_image_ = get_parameter("publish_debug_image").as_bool();
  publish_timing_ = get_parameter("publish_timing").as_bool();

  return config;
}

YolosDetectorNode::CallbackReturn YolosDetectorNode::on_configure(const rclcpp_lifecycle::State&) {
  RCLCPP_INFO(get_logger(), "Configuring...");
  try {
    auto config = loadConfig();
    if (config.model_path.empty() || config.labels_path.empty()) {
      RCLCPP_ERROR(get_logger(), "model_path and labels_path are required");
      return CallbackReturn::FAILURE;
    }
    detector_ = createDetectorAdapter();
    if (!detector_->initialize(config)) {
      RCLCPP_ERROR(get_logger(), "Failed to initialize detector");
      return CallbackReturn::FAILURE;
    }
    inference_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    det_pub_ = create_publisher<vision_msgs::msg::Detection2DArray>("~/detections", 10);
    if (publish_debug_image_) debug_pub_ = create_publisher<sensor_msgs::msg::Image>("~/debug_image", 1);
    if (publish_timing_) timing_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("~/timing", 10);
    RCLCPP_INFO(get_logger(), "Configured successfully");
    return CallbackReturn::SUCCESS;
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_logger(), "Configuration failed: %s", e.what());
    return CallbackReturn::FAILURE;
  }
}

YolosDetectorNode::CallbackReturn YolosDetectorNode::on_activate(const rclcpp_lifecycle::State&) {
  RCLCPP_INFO(get_logger(), "Activating...");
  det_pub_->on_activate();
  if (debug_pub_) debug_pub_->on_activate();
  if (timing_pub_) timing_pub_->on_activate();

  auto sub_options = rclcpp::SubscriptionOptions();
  sub_options.callback_group = inference_cb_group_;
  image_sub_ = create_subscription<sensor_msgs::msg::Image>(
    "~/image_raw", rclcpp::SensorDataQoS(),
    std::bind(&YolosDetectorNode::imageCallback, this, std::placeholders::_1),
    sub_options);

  RCLCPP_INFO(get_logger(), "Activated - listening on ~/image_raw");
  return CallbackReturn::SUCCESS;
}

YolosDetectorNode::CallbackReturn YolosDetectorNode::on_deactivate(const rclcpp_lifecycle::State&) {
  image_sub_.reset();
  det_pub_->on_deactivate();
  if (debug_pub_) debug_pub_->on_deactivate();
  if (timing_pub_) timing_pub_->on_deactivate();
  return CallbackReturn::SUCCESS;
}

YolosDetectorNode::CallbackReturn YolosDetectorNode::on_cleanup(const rclcpp_lifecycle::State&) {
  if (detector_) { detector_->shutdown(); detector_.reset(); }
  det_pub_.reset(); debug_pub_.reset(); timing_pub_.reset();
  return CallbackReturn::SUCCESS;
}

YolosDetectorNode::CallbackReturn YolosDetectorNode::on_shutdown(const rclcpp_lifecycle::State& s) {
  return on_cleanup(s);
}

void YolosDetectorNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
  if (!detector_ || !detector_->isInitialized()) return;
  auto t_start = std::chrono::high_resolution_clock::now();
  try {
    auto cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
    auto t_preprocess = std::chrono::high_resolution_clock::now();
    auto detections = detector_->detect(cv_ptr->image, conf_threshold_, nms_threshold_);
    auto t_inference = std::chrono::high_resolution_clock::now();
    det_pub_->publish(conversion::toDetection2DArray(detections, msg->header, msg->width, msg->height));
    auto t_postprocess = std::chrono::high_resolution_clock::now();

    if (publish_debug_image_ && debug_pub_ && debug_pub_->is_activated()) {
      cv::Mat debug = cv_ptr->image.clone();
      detector_->drawDetections(debug, detections);
      debug_pub_->publish(*cv_bridge::CvImage(msg->header, "bgr8", debug).toImageMsg());
    }
    if (publish_timing_ && timing_pub_ && timing_pub_->is_activated()) {
      std_msgs::msg::Float64MultiArray timing;
      timing.data = {
        std::chrono::duration<double, std::milli>(t_preprocess - t_start).count(),
        std::chrono::duration<double, std::milli>(t_inference - t_preprocess).count(),
        std::chrono::duration<double, std::milli>(t_postprocess - t_inference).count()
      };
      timing_pub_->publish(timing);
    }
  } catch (const std::exception& e) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Detection failed: %s", e.what());
  }
}

}  // namespace ros2_yolos_cpp

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(ros2_yolos_cpp::YolosDetectorNode)
