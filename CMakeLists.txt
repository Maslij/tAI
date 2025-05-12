cmake_minimum_required(VERSION 3.10)
project(tAI VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set OpenCV_DIR to point to OpenCV 4.8 installation
set(OpenCV_DIR "/home/alec/opencv_build/opencv/build")

# Find required packages
find_package(OpenCV 4.8 REQUIRED)
find_package(Boost COMPONENTS system filesystem REQUIRED)
find_package(nlohmann_json 3.10.5 REQUIRED)
find_package(CURL REQUIRED)

# ONNX Runtime is required
set(ONNXRUNTIME_ROOTDIR "/usr/local" CACHE PATH "ONNX Runtime root directory")
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime PATHS ${ONNXRUNTIME_ROOTDIR}/lib)
    
if(NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "ONNX Runtime library is required but not found. Please install ONNX Runtime.")
endif()

message(STATUS "ONNX Runtime found at: ${ONNXRUNTIME_LIBRARY}")
add_definitions(-DUSE_ONNXRUNTIME)
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime" CACHE PATH "ONNX Runtime include directory")

# Add subdirectories
add_subdirectory(src)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${CURL_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
) 