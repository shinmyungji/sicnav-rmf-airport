cmake_minimum_required(VERSION 3.20)
project(sicnav_state_reader)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

find_package(gz_sim_vendor REQUIRED)
find_package(gz-sim REQUIRED)

find_package(gz_plugin_vendor REQUIRED)
find_package(gz-plugin REQUIRED)

add_library(
  sicnav_state_reader
  SHARED
  src/state_reader.cpp
)

ament_target_dependencies(
  sicnav_state_reader
  rclcpp
  std_msgs
)

target_link_libraries(
  sicnav_state_reader
  gz-sim::core
  gz-plugin::core
)

install(
  TARGETS sicnav_state_reader
  LIBRARY DESTINATION lib
)

ament_package()
