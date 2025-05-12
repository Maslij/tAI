cmake_minimum_required(VERSION 3.10)
project(tAI VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build options
option(USE_ONNXRUNTIME "Build with ONNXRuntime support (requires ONNXRuntime library)" ON)

# Set OpenCV_DIR to point to OpenCV 4.8 installation
set(OpenCV_DIR "/home/alec/opencv_build/opencv/build")

# Find required packages
find_package(OpenCV 4.8 REQUIRED)
find_package(Boost COMPONENTS system filesystem REQUIRED)
find_package(nlohmann_json 3.10.5 REQUIRED)
find_package(CURL REQUIRED)

# Add compile definitions based on options
if(USE_ONNXRUNTIME)
    # Find ONNX Runtime package
    set(ONNXRUNTIME_ROOTDIR "/usr/local" CACHE PATH "ONNX Runtime root directory")
    find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime PATHS ${ONNXRUNTIME_ROOTDIR}/lib)
    
    if(ONNXRUNTIME_LIBRARY)
        message(STATUS "ONNX Runtime found at: ${ONNXRUNTIME_LIBRARY}")
        add_definitions(-DUSE_ONNXRUNTIME)
        set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime" CACHE PATH "ONNX Runtime include directory")
    else()
        message(WARNING "ONNX Runtime library not found, disabling ONNX support")
        set(USE_ONNXRUNTIME OFF)
    endif()
endif()

# Add subdirectories
add_subdirectory(src)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${CURL_INCLUDE_DIRS}
)

if(USE_ONNXRUNTIME)
    include_directories(${ONNXRUNTIME_INCLUDE_DIR})
endif() 