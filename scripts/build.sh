#!/bin/bash
set -e

# Default build type
BUILD_TYPE=Release
BUILD_TESTS=OFF

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --debug)
      BUILD_TYPE=Debug
      shift
      ;;
    --tests)
      BUILD_TESTS=ON
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --debug    Build in debug mode"
      echo "  --tests    Build tests"
      echo "  --help     Show this help"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Get project directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( dirname "$SCRIPT_DIR" )"

# Create build directory
mkdir -p "$PROJECT_DIR/build"
cd "$PROJECT_DIR/build"

# Configure and build
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DBUILD_TESTS=$BUILD_TESTS ..
cmake --build . -- -j$(nproc)

echo "Build completed successfully!"
if [[ "$BUILD_TESTS" == "ON" ]]; then
  echo "Run tests with: cd build && ctest"
fi

echo "Run tAI with: ./build/tAI_server <path_to_yolo_model>" 