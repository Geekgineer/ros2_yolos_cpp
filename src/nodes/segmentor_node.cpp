// Copyright 2024 YOLOs-CPP Team
// SPDX-License-Identifier: AGPL-3.0

#include "ros2_yolos_cpp/nodes/segmentor_node.hpp"
#include "ros2_yolos_cpp/conversion/segmentation_converter.hpp"
#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif

namespace ros2_yolos_cpp {

YolosSegmentorNode::YolosSegmentorNode(const rclcpp::NodeOptions& o) : rclcpp_lifecycle::LifecycleNode("yolos_segmentor", o) { declareParameters(); }

void YolosSegmentorNode::declareParameters() {
  declare_parameter("model_path", rclcpp::PARAMETER_STRING);
  declare_parameter("labels_path", rclcpp::PARAMETER_STRING);
  declare_parameter("use_gpu", false);
  declare_parameter("use_tensorrt", false);
  declare_parameter("tensorrt_fp16", true);
  declare_parameter("tensorrt_cache_path", "");
  declare_parameter("conf_threshold", 0.4);
  declare_parameter("nms_threshold", 0.45);
  declare_parameter("publish_debug_image", false);
}

YolosConfig YolosSegmentorNode::loadConfig() {
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

YolosSegmentorNode::CallbackReturn YolosSegmentorNode::on_configure(const rclcpp_lifecycle::State&) {
  auto c = loadConfig();
  if (c.model_path.empty() || c.labels_path.empty()) return CallbackReturn::FAILURE;
  segmentor_ = createSegmentorAdapter();
  if (!segmentor_->initialize(c)) return CallbackReturn::FAILURE;
  cb_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  det_pub_ = create_publisher<vision_msgs::msg::Detection2DArray>("~/detections", 10);
  mask_pub_ = create_publisher<sensor_msgs::msg::Image>("~/mask", 1);
  if (debug_) debug_pub_ = create_publisher<sensor_msgs::msg::Image>("~/debug_image", 1);
  return CallbackReturn::SUCCESS;
}

YolosSegmentorNode::CallbackReturn YolosSegmentorNode::on_activate(const rclcpp_lifecycle::State&) {
  det_pub_->on_activate(); mask_pub_->on_activate();
  if (debug_pub_) debug_pub_->on_activate();
  auto opt = rclcpp::SubscriptionOptions(); opt.callback_group = cb_;
  sub_ = create_subscription<sensor_msgs::msg::Image>("~/image_raw", rclcpp::SensorDataQoS(),
    std::bind(&YolosSegmentorNode::imageCallback, this, std::placeholders::_1), opt);
  return CallbackReturn::SUCCESS;
}

YolosSegmentorNode::CallbackReturn YolosSegmentorNode::on_deactivate(const rclcpp_lifecycle::State&) {
  sub_.reset(); det_pub_->on_deactivate(); mask_pub_->on_deactivate();
  if (debug_pub_) debug_pub_->on_deactivate();
  return CallbackReturn::SUCCESS;
}

YolosSegmentorNode::CallbackReturn YolosSegmentorNode::on_cleanup(const rclcpp_lifecycle::State&) {
  if (segmentor_) { segmentor_->shutdown(); segmentor_.reset(); }
  det_pub_.reset(); mask_pub_.reset(); debug_pub_.reset();
  return CallbackReturn::SUCCESS;
}

void YolosSegmentorNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
  if (!segmentor_ || !segmentor_->isInitialized()) return;
  try {
    auto cv = cv_bridge::toCvShare(msg, "bgr8");
    auto segs = segmentor_->segment(cv->image, conf_, nms_);
    det_pub_->publish(conversion::toDetection2DArray(segs, msg->header, msg->width, msg->height));
    mask_pub_->publish(conversion::toCombinedMaskImage(segs, msg->header, msg->width, msg->height));
    if (debug_ && debug_pub_ && debug_pub_->is_activated()) {
      cv::Mat d = cv->image.clone(); segmentor_->drawSegmentations(d, segs);
      debug_pub_->publish(*cv_bridge::CvImage(msg->header, "bgr8", d).toImageMsg());
    }
  } catch (const std::exception& e) { RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "%s", e.what()); }
}

}  // namespace ros2_yolos_cpp

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(ros2_yolos_cpp::YolosSegmentorNode)
