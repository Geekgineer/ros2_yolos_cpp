// Copyright 2024 YOLOs-CPP Team
// SPDX-License-Identifier: AGPL-3.0

#include "ros2_yolos_cpp/nodes/classifier_node.hpp"
#include "ros2_yolos_cpp/conversion/classification_converter.hpp"
#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif

namespace ros2_yolos_cpp {

YolosClassifierNode::YolosClassifierNode(const rclcpp::NodeOptions& o) : rclcpp_lifecycle::LifecycleNode("yolos_classifier", o) { declareParameters(); }

void YolosClassifierNode::declareParameters() {
  declare_parameter("model_path", rclcpp::PARAMETER_STRING);
  declare_parameter("labels_path", rclcpp::PARAMETER_STRING);
  declare_parameter("use_gpu", false);
  declare_parameter("use_tensorrt", false);
  declare_parameter("tensorrt_fp16", true);
  declare_parameter("tensorrt_cache_path", "");
  declare_parameter("publish_debug_image", false);
}

YolosConfig YolosClassifierNode::loadConfig() {
  YolosConfig c;
  c.model_path = get_parameter("model_path").as_string();
  c.labels_path = get_parameter("labels_path").as_string();
  c.use_gpu = get_parameter("use_gpu").as_bool();
  c.use_tensorrt = get_parameter("use_tensorrt").as_bool();
  c.tensorrt_fp16 = get_parameter("tensorrt_fp16").as_bool();
  c.tensorrt_cache_path = get_parameter("tensorrt_cache_path").as_string();
  debug_ = get_parameter("publish_debug_image").as_bool();
  return c;
}

YolosClassifierNode::CallbackReturn YolosClassifierNode::on_configure(const rclcpp_lifecycle::State&) {
  auto c = loadConfig();
  cls_ = createClassifierAdapter();
  if (!cls_->initialize(c)) return CallbackReturn::FAILURE;
  cb_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  pub_ = create_publisher<vision_msgs::msg::Classification>("~/classification", 10);
  if (debug_) dbg_ = create_publisher<sensor_msgs::msg::Image>("~/debug_image", 1);
  return CallbackReturn::SUCCESS;
}

YolosClassifierNode::CallbackReturn YolosClassifierNode::on_activate(const rclcpp_lifecycle::State&) {
  pub_->on_activate(); if (dbg_) dbg_->on_activate();
  auto opt = rclcpp::SubscriptionOptions(); opt.callback_group = cb_;
  sub_ = create_subscription<sensor_msgs::msg::Image>("~/image_raw", rclcpp::SensorDataQoS(),
    std::bind(&YolosClassifierNode::imageCallback, this, std::placeholders::_1), opt);
  return CallbackReturn::SUCCESS;
}

YolosClassifierNode::CallbackReturn YolosClassifierNode::on_deactivate(const rclcpp_lifecycle::State&) {
  sub_.reset(); pub_->on_deactivate(); if (dbg_) dbg_->on_deactivate();
  return CallbackReturn::SUCCESS;
}

YolosClassifierNode::CallbackReturn YolosClassifierNode::on_cleanup(const rclcpp_lifecycle::State&) {
  if (cls_) { cls_->shutdown(); cls_.reset(); }
  pub_.reset(); dbg_.reset();
  return CallbackReturn::SUCCESS;
}

void YolosClassifierNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
  if (!cls_ || !cls_->isInitialized()) return;
  try {
    auto cv = cv_bridge::toCvShare(msg, "bgr8");
    auto result = cls_->classify(cv->image);
    pub_->publish(conversion::toClassification(result, msg->header));
    if (debug_ && dbg_ && dbg_->is_activated()) {
      cv::Mat d = cv->image.clone(); cls_->drawResult(d, result);
      dbg_->publish(*cv_bridge::CvImage(msg->header, "bgr8", d).toImageMsg());
    }
  } catch (const std::exception& e) { RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "%s", e.what()); }
}

}  // namespace ros2_yolos_cpp

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(ros2_yolos_cpp::YolosClassifierNode)
