cmake_minimum_required(VERSION 2.6)

project(resnet18)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE DEBUG)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt
# cuda
include_directories(/usr/local/cuda-10.2/include/)
link_directories(/usr/local/cuda-10.2/lib64/)
# tensorrt
include_directories(/usr/include/aarch64-linux-gnu/)
link_directories(/usr/lib/aarch64-linux-gnu/)

add_executable(resnet18 ${PROJECT_SOURCE_DIR}/resnet18_deploy.cpp)
target_link_libraries(resnet18 nvinfer)
target_link_libraries(resnet18 cudart)

add_definitions(-O2 -pthread)