cmake_minimum_required(VERSION 3.10)
project(tAI VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set OpenCV_DIR to point to OpenCV 4.8 installation
set(OpenCV_DIR "/home/alec/Downloads/opencv/build")

# Find required packages
find_package(OpenCV 4.8 REQUIRED)
find_package(Boost COMPONENTS system filesystem REQUIRED)
find_package(nlohmann_json 3.10.5 REQUIRED)
find_package(CURL REQUIRED)

# Add subdirectories
add_subdirectory(src)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${CURL_INCLUDE_DIRS}
) 