// Copyright 2024 YOLOs-CPP Team
// SPDX-License-Identifier: AGPL-3.0

#include "ros2_yolos_cpp/nodes/pose_node.hpp"
#include "ros2_yolos_cpp/conversion/pose_converter.hpp"
#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif

namespace ros2_yolos_cpp {

YolosPoseNode::YolosPoseNode(const rclcpp::NodeOptions& o) : rclcpp_lifecycle::LifecycleNode("yolos_pose", o) { declareParameters(); }

void YolosPoseNode::declareParameters() {
  declare_parameter("model_path", rclcpp::PARAMETER_STRING);
  declare_parameter("labels_path", "");
  declare_parameter("use_gpu", false);
  declare_parameter("use_tensorrt", false);
  declare_parameter("tensorrt_fp16", true);
  declare_parameter("tensorrt_cache_path", "");
  declare_parameter("conf_threshold", 0.4);
  declare_parameter("nms_threshold", 0.5);
  declare_parameter("publish_debug_image", false);
}

YolosConfig YolosPoseNode::loadConfig() {
  YolosConfig c;
  c.model_path = get_parameter("model_path").as_string();
  c.labels_path = get_parameter("labels_path").as_string();
  c.use_gpu = get_parameter("use_gpu").as_bool();
  c.use_tensorrt = get_parameter("use_tensorrt").as_bool();
  c.tensorrt_fp16 = get_parameter("tensorrt_fp16").as_bool();
  c.tensorrt_cache_path = get_parameter("tensorrt_cache_path").as_string();
  c.conf_threshold = static_cast<float>(get_parameter("conf_threshold").as_double());
  c.nms_threshold = static_cast<float>(get_parameter("nms_threshold").as_double());
  conf_ = c.conf_threshold; nms_ = c.nms_threshold;
  debug_ = get_parameter("publish_debug_image").as_bool();
  return c;
}

YolosPoseNode::CallbackReturn YolosPoseNode::on_configure(const rclcpp_lifecycle::State&) {
  auto c = loadConfig();
  if (c.model_path.empty()) return CallbackReturn::FAILURE;
  pose_ = createPoseAdapter();
  if (!pose_->initialize(c)) return CallbackReturn::FAILURE;
  cb_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  pub_ = create_publisher<vision_msgs::msg::Detection2DArray>("~/detections", 10);
  if (debug_) debug_pub_ = create_publisher<sensor_msgs::msg::Image>("~/debug_image", 1);
  return CallbackReturn::SUCCESS;
}

YolosPoseNode::CallbackReturn YolosPoseNode::on_activate(const rclcpp_lifecycle::State&) {
  pub_->on_activate(); if (debug_pub_) debug_pub_->on_activate();
  auto opt = rclcpp::SubscriptionOptions(); opt.callback_group = cb_;
  sub_ = create_subscription<sensor_msgs::msg::Image>("~/image_raw", rclcpp::SensorDataQoS(),
    std::bind(&YolosPoseNode::imageCallback, this, std::placeholders::_1), opt);
  return CallbackReturn::SUCCESS;
}

YolosPoseNode::CallbackReturn YolosPoseNode::on_deactivate(const rclcpp_lifecycle::State&) {
  sub_.reset(); pub_->on_deactivate(); if (debug_pub_) debug_pub_->on_deactivate();
  return CallbackReturn::SUCCESS;
}

YolosPoseNode::CallbackReturn YolosPoseNode::on_cleanup(const rclcpp_lifecycle::State&) {
  if (pose_) { pose_->shutdown(); pose_.reset(); }
  pub_.reset(); debug_pub_.reset();
  return CallbackReturn::SUCCESS;
}

void YolosPoseNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
  if (!pose_ || !pose_->isInitialized()) return;
  try {
    auto cv = cv_bridge::toCvShare(msg, "bgr8");
    auto poses = pose_->detect(cv->image, conf_, nms_);
    pub_->publish(conversion::toDetection2DArray(poses, msg->header, msg->width, msg->height));
    if (debug_ && debug_pub_ && debug_pub_->is_activated()) {
      cv::Mat d = cv->image.clone(); pose_->drawPoses(d, poses);
      debug_pub_->publish(*cv_bridge::CvImage(msg->header, "bgr8", d).toImageMsg());
    }
  } catch (const std::exception& e) { RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "%s", e.what()); }
}

}  // namespace ros2_yolos_cpp

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(ros2_yolos_cpp::YolosPoseNode)
