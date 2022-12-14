cmake_minimum_required(VERSION 2.6)

project(lenet)

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

add_executable(lenet ${PROJECT_SOURCE_DIR}/lenet5_engine.cpp)
target_link_libraries(lenet nvinfer)
target_link_libraries(lenet cudart)

add_definitions(-O2 -pthread)