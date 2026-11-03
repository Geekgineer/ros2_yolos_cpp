# =============================================================================
# ros2_yolos_cpp - ROS 2 Wrapper for YOLOs-CPP Inference Library
# =============================================================================
# Production-grade YOLO inference nodes with lifecycle management,
# zero-copy composition, and adapter layer for loose coupling.
# =============================================================================
cmake_minimum_required(VERSION 3.16)
project(ros2_yolos_cpp VERSION 1.0.0 LANGUAGES C CXX)

# =============================================================================
# C++ Standard
# =============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release build if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# =============================================================================
# Build Options
# =============================================================================
option(BUILD_TESTING "Build tests" ON)
option(ENABLE_GPU "Enable CUDA GPU support" OFF)

# =============================================================================
# Find ROS 2 Dependencies
# =============================================================================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(lifecycle_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# =============================================================================
# Find OpenCV
# =============================================================================
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# =============================================================================
# ONNX Runtime - Find or Download
# =============================================================================

# Check if ONNX Runtime is already installed
find_library(ONNXRUNTIME_LIB onnxruntime
  PATHS
    ${ONNXRUNTIME_DIR}/lib
    /usr/local/lib
    /opt/onnxruntime/lib
    $ENV{ONNXRUNTIME_DIR}/lib
)

if(ONNXRUNTIME_LIB)
  message(STATUS "Found ONNX Runtime: ${ONNXRUNTIME_LIB}")
  # Find include directory
  find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    PATHS
      ${ONNXRUNTIME_DIR}/include
      /usr/local/include/onnxruntime
      /opt/onnxruntime/include
      $ENV{ONNXRUNTIME_DIR}/include
  )
  if(NOT ONNXRUNTIME_INCLUDE_DIR)
    message(FATAL_ERROR "ONNX Runtime headers not found")
  endif()
else()
  # Download ONNX Runtime manually (CMake 3.22 compatible)
  set(ONNXRUNTIME_VERSION "1.20.1")
  set(ONNXRUNTIME_DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/onnxruntime-download")
  set(ONNXRUNTIME_EXTRACT_DIR "${CMAKE_BINARY_DIR}/onnxruntime")
  
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(ONNXRUNTIME_ARCHIVE "onnxruntime-linux-aarch64-${ONNXRUNTIME_VERSION}.tgz")
  else()
    if(ENABLE_GPU)
      set(ONNXRUNTIME_ARCHIVE "onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz")
    else()
      set(ONNXRUNTIME_ARCHIVE "onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz")
    endif()
  endif()
  
  set(ONNXRUNTIME_ARCHIVE_PATH "${ONNXRUNTIME_DOWNLOAD_DIR}/${ONNXRUNTIME_ARCHIVE}")
  set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${ONNXRUNTIME_ARCHIVE}")
  
  if(NOT EXISTS "${ONNXRUNTIME_EXTRACT_DIR}/lib/libonnxruntime.so")
    message(STATUS "Downloading ONNX Runtime from: ${ONNXRUNTIME_URL}")
    
    file(MAKE_DIRECTORY ${ONNXRUNTIME_DOWNLOAD_DIR})
    file(DOWNLOAD
      "${ONNXRUNTIME_URL}"
      "${ONNXRUNTIME_ARCHIVE_PATH}"
      SHOW_PROGRESS
      STATUS DOWNLOAD_STATUS
    )
    list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
    if(NOT STATUS_CODE EQUAL 0)
      message(FATAL_ERROR "Failed to download ONNX Runtime: ${DOWNLOAD_STATUS}")
    endif()
    
    message(STATUS "Extracting ONNX Runtime...")
    file(MAKE_DIRECTORY ${ONNXRUNTIME_EXTRACT_DIR})
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E tar xzf "${ONNXRUNTIME_ARCHIVE_PATH}"
      WORKING_DIRECTORY "${ONNXRUNTIME_DOWNLOAD_DIR}"
      RESULT_VARIABLE EXTRACT_RESULT
    )
    if(NOT EXTRACT_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to extract ONNX Runtime")
    endif()
    
    # Move contents to extract dir (strip top-level folder)
    file(GLOB ONNX_CONTENTS "${ONNXRUNTIME_DOWNLOAD_DIR}/onnxruntime-*/*")
    foreach(item ${ONNX_CONTENTS})
      get_filename_component(item_name "${item}" NAME)
      file(RENAME "${item}" "${ONNXRUNTIME_EXTRACT_DIR}/${item_name}")
    endforeach()
  endif()
  
  set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_EXTRACT_DIR}/include")
  set(ONNXRUNTIME_LIB "${ONNXRUNTIME_EXTRACT_DIR}/lib/libonnxruntime.so")
endif()

message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime lib: ${ONNXRUNTIME_LIB}")

# =============================================================================
# YOLOs-CPP Library - Download at Configuration Time
# =============================================================================
# Branch, tag or commit of YOLOs-CPP. patches/yolos_cpp_tensorrt.patch was made for commit 2e50d31; pin it with
# -DYOLOS_CPP_VERSION=2e50d31f0837e96160b72a3311d8a5d4e30b0e47 for reproducible builds.
set(YOLOS_CPP_VERSION "master" CACHE STRING "YOLOs-CPP branch, tag or commit to download")
set(YOLOS_CPP_DIR "${CMAKE_BINARY_DIR}/YOLOs-CPP")
set(YOLOS_CPP_URL "https://github.com/Geekgineer/YOLOs-CPP/archive/${YOLOS_CPP_VERSION}.tar.gz")

if(NOT EXISTS "${YOLOS_CPP_DIR}/include/yolos/yolos.hpp")
  message(STATUS "Downloading YOLOs-CPP from: ${YOLOS_CPP_URL}")
  
  file(DOWNLOAD
    "${YOLOS_CPP_URL}"
    "${CMAKE_BINARY_DIR}/yolos_cpp.tar.gz"
    SHOW_PROGRESS
    STATUS DOWNLOAD_STATUS
  )
  
  list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
  if(NOT STATUS_CODE EQUAL 0)
    message(FATAL_ERROR "Failed to download YOLOs-CPP: ${DOWNLOAD_STATUS}")
  endif()
  
  message(STATUS "Extracting YOLOs-CPP...")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xzf "${CMAKE_BINARY_DIR}/yolos_cpp.tar.gz"
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    RESULT_VARIABLE EXTRACT_RESULT
  )
  
  if(NOT EXTRACT_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to extract YOLOs-CPP")
  endif()
  
  # Find the extracted folder (can vary by branch, e.g. YOLOs-CPP-main or YOLOs-CPP-master)
  file(GLOB EXTRACTED_DIRS "${CMAKE_BINARY_DIR}/YOLOs-CPP-*")
  if(NOT EXTRACTED_DIRS)
    message(FATAL_ERROR "Extraction failed: Could not find YOLOs-CPP-* directory in ${CMAKE_BINARY_DIR}")
  endif()
  
  # Use the first match
  list(GET EXTRACTED_DIRS 0 EXTRACTED_DIR)
  
  # Remove build dir if it exists from previous run
  file(REMOVE_RECURSE "${YOLOS_CPP_DIR}")
  
  # Rename to standard build directory
  file(RENAME "${EXTRACTED_DIR}" "${YOLOS_CPP_DIR}")
  file(REMOVE "${CMAKE_BINARY_DIR}/yolos_cpp.tar.gz")

  # TensorRT execution provider (yolos::tensorRTOptions), unless this YOLOs-CPP version already provides it
  set(YOLOS_SESSION_BASE "")
  if(EXISTS "${YOLOS_CPP_DIR}/include/yolos/core/session_base.hpp")
    file(READ "${YOLOS_CPP_DIR}/include/yolos/core/session_base.hpp" YOLOS_SESSION_BASE)
  endif()
  if(NOT YOLOS_SESSION_BASE MATCHES "tensorRTOptions")
    find_package(Git QUIET)
    set(PATCH_RESULT 1)
    if(GIT_FOUND)
      execute_process(
        COMMAND ${GIT_EXECUTABLE} apply "${CMAKE_CURRENT_SOURCE_DIR}/patches/yolos_cpp_tensorrt.patch"
        WORKING_DIRECTORY "${YOLOS_CPP_DIR}"
        RESULT_VARIABLE PATCH_RESULT
        ERROR_QUIET
      )
    endif()
    if(NOT PATCH_RESULT EQUAL 0)
      message(WARNING "Could not apply patches/yolos_cpp_tensorrt.patch to YOLOs-CPP ${YOLOS_CPP_VERSION}, "
                      "building without TensorRT support (use_tensorrt)")
    endif()
  endif()
endif()

message(STATUS "YOLOs-CPP Source: ${YOLOS_CPP_DIR}")

# TensorRT settings in YOLOs-CPP (yolos::tensorRTOptions), patched in above or provided by YOLOs-CPP itself
set(YOLOS_SESSION_BASE "")
if(EXISTS "${YOLOS_CPP_DIR}/include/yolos/core/session_base.hpp")
  file(READ "${YOLOS_CPP_DIR}/include/yolos/core/session_base.hpp" YOLOS_SESSION_BASE)
endif()
if(YOLOS_SESSION_BASE MATCHES "tensorRTOptions")
  set(YOLOS_CPP_HAS_TENSORRT ON)
else()
  set(YOLOS_CPP_HAS_TENSORRT OFF)
endif()
message(STATUS "YOLOs-CPP TensorRT support: ${YOLOS_CPP_HAS_TENSORRT}")

# =============================================================================
# Custom Messages (OBB Detection)
# =============================================================================
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/OrientedBoundingBox2D.msg"
  "msg/OBBDetection2D.msg"
  "msg/OBBDetection2DArray.msg"
  "msg/KeyPoint2D.msg"
  "msg/KeyPoint2DArray.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

# =============================================================================
# Include Directories
# =============================================================================
include_directories(
  include
  ${YOLOS_CPP_DIR}/include
  ${ONNXRUNTIME_INCLUDE_DIR}
)

# =============================================================================
# Component Library
# =============================================================================
add_library(${PROJECT_NAME}_components SHARED
  # Adapters
  src/adapters/detector_adapter.cpp
  src/adapters/segmentor_adapter.cpp
  src/adapters/pose_adapter.cpp
  src/adapters/obb_adapter.cpp
  src/adapters/classifier_adapter.cpp
  # Message Converters
  src/conversion/detection_converter.cpp
  src/conversion/segmentation_converter.cpp
  src/conversion/pose_converter.cpp
  src/conversion/obb_converter.cpp
  src/conversion/classification_converter.cpp
  # Nodes
  src/nodes/detector_node.cpp
  src/nodes/segmentor_node.cpp
  src/nodes/pose_node.cpp
  src/nodes/obb_node.cpp
  src/nodes/classifier_node.cpp
)

target_include_directories(${PROJECT_NAME}_components PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

if(YOLOS_CPP_HAS_TENSORRT)
  target_compile_definitions(${PROJECT_NAME}_components PRIVATE YOLOS_CPP_HAS_TENSORRT)
endif()

target_link_libraries(${PROJECT_NAME}_components
  ${ONNXRUNTIME_LIB}
  ${OpenCV_LIBS}
)

ament_target_dependencies(${PROJECT_NAME}_components
  rclcpp
  rclcpp_lifecycle
  rclcpp_components
  image_transport
  cv_bridge
  sensor_msgs
  vision_msgs
  std_msgs
  geometry_msgs
  lifecycle_msgs
)

# Link custom messages to component library
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(${PROJECT_NAME}_components "${cpp_typesupport_target}")

# =============================================================================
# Register Components
# =============================================================================
rclcpp_components_register_node(${PROJECT_NAME}_components
  PLUGIN "ros2_yolos_cpp::YolosDetectorNode"
  EXECUTABLE yolos_detector_node
)

rclcpp_components_register_node(${PROJECT_NAME}_components
  PLUGIN "ros2_yolos_cpp::YolosSegmentorNode"
  EXECUTABLE yolos_segmentor_node
)

rclcpp_components_register_node(${PROJECT_NAME}_components
  PLUGIN "ros2_yolos_cpp::YolosPoseNode"
  EXECUTABLE yolos_pose_node
)

rclcpp_components_register_node(${PROJECT_NAME}_components
  PLUGIN "ros2_yolos_cpp::YolosOBBNode"
  EXECUTABLE yolos_obb_node
)

rclcpp_components_register_node(${PROJECT_NAME}_components
  PLUGIN "ros2_yolos_cpp::YolosClassifierNode"
  EXECUTABLE yolos_classifier_node
)

# =============================================================================
# Set RPATH for ONNX Runtime
# =============================================================================
if(UNIX AND NOT APPLE)
  get_filename_component(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_LIB} DIRECTORY)
  set_target_properties(${PROJECT_NAME}_components PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}"
    INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
  )
endif()

# =============================================================================
# Installation
# =============================================================================
install(TARGETS ${PROJECT_NAME}_components
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# =============================================================================
# Testing
# =============================================================================
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  
  find_package(ament_cmake_gtest REQUIRED)
  
  # Unit tests (always run)
  ament_add_gtest(test_detection_converter
    test/unit/test_detection_converter.cpp
  )
  target_link_libraries(test_detection_converter ${PROJECT_NAME}_components)
  ament_target_dependencies(test_detection_converter rclcpp vision_msgs)
  
  # Integration tests (require model files via environment variables)
  # Set YOLOS_TEST_MODEL, YOLOS_TEST_LABELS, YOLOS_TEST_IMAGE to run
  
  macro(add_integration_test target_name source_file)
    ament_add_gtest(${target_name} ${source_file})
    target_link_libraries(${target_name} 
      ${PROJECT_NAME}_components 
      ${OpenCV_LIBS}
    )
    ament_target_dependencies(${target_name} rclcpp rclcpp_lifecycle)
  endmacro()

  add_integration_test(test_integration_detection test/integration/test_detection.cpp)
  add_integration_test(test_integration_segmentation test/integration/test_segmentation.cpp)
  add_integration_test(test_integration_pose test/integration/test_pose.cpp)
  add_integration_test(test_integration_obb test/integration/test_obb.cpp)
  add_integration_test(test_integration_classification test/integration/test_classification.cpp)
endif()

# =============================================================================
# Export
# =============================================================================
ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME}_components)
ament_export_dependencies(
  rclcpp
  rclcpp_lifecycle
  rclcpp_components
  image_transport
  cv_bridge
  sensor_msgs
  vision_msgs
)

ament_package()

# =============================================================================
# Summary
# =============================================================================
message(STATUS "")
message(STATUS "=== ros2_yolos_cpp Configuration ===")
message(STATUS "  Version:       ${PROJECT_VERSION}")
message(STATUS "  Build Type:    ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard:  ${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenCV:        ${OpenCV_VERSION}")
message(STATUS "  ONNX Runtime:  ${ONNXRUNTIME_LIB}")
message(STATUS "  YOLOs-CPP:     ${YOLOS_CPP_DIR}")
message(STATUS "  GPU Support:   ${ENABLE_GPU}")
message(STATUS "")
