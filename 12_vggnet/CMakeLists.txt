cmake_minimum_required(VERSION 2.6)

project(vggnet)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE DEBUG)

message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
message(STATUS "PROJECT_BINARY_DIR: ${PROJECT_BINARY_DIR}")

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt
# cuda
include_directories(/usr/local/cuda-10.2/include/)
link_directories(/usr/local/cuda-10.2/lib64/)
# tensorrt
include_directories(/usr/include/aarch64-linux-gnu/)
link_directories(/usr/lib/aarch64-linux-gnu/)

add_executable(vggnet ${PROJECT_SOURCE_DIR}/vggnet_engine.cpp)
target_link_libraries(vggnet nvinfer)
target_link_libraries(vggnet cudart)

add_definitions(-O2 -pthread)
