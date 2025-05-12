FROM nvcr.io/nvidia/l4t-ml:r35.1.0-py3

# Install additional dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libboost-all-dev \
    libcurl4-openssl-dev \
    nlohmann-json3-dev \
    && rm -rf /var/lib/apt/lists/*

# Verify OpenCV with CUDA support
RUN python3 -c "import cv2; print('OpenCV version:', cv2.__version__)"

# Copy the project
WORKDIR /app
COPY . .

# Update the CMakeLists.txt to use the correct paths and versions
RUN sed -i 's|set(OpenCV_DIR "/home/alec/opencv_build/opencv/build")|set(OpenCV_DIR "/usr/lib/aarch64-linux-gnu/cmake/opencv4")|g' CMakeLists.txt && \
    sed -i 's|find_package(OpenCV 4.8 REQUIRED)|find_package(OpenCV 4.5 REQUIRED)|g' CMakeLists.txt && \
    sed -i 's|find_package(nlohmann_json 3.10.5 REQUIRED)|find_package(nlohmann_json 3.7.3 REQUIRED)|g' CMakeLists.txt

# Configure ONNXRuntime paths - the l4t-ml image has it in /usr/lib/aarch64-linux-gnu
RUN sed -i 's|set(ONNXRUNTIME_ROOTDIR "/usr/local" CACHE PATH "ONNX Runtime root directory")|set(ONNXRUNTIME_ROOTDIR "/usr" CACHE PATH "ONNX Runtime root directory")|g' CMakeLists.txt && \
    sed -i '/find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime PATHS ${ONNXRUNTIME_ROOTDIR}\/lib)/c\    find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime libonnxruntime.so PATHS ${ONNXRUNTIME_ROOTDIR}/lib ${ONNXRUNTIME_ROOTDIR}/lib/aarch64-linux-gnu)' CMakeLists.txt && \
    sed -i 's|set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime" CACHE PATH "ONNX Runtime include directory")|set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOTDIR}/include" CACHE PATH "ONNX Runtime include directory")|g' CMakeLists.txt

# Download models
RUN chmod +x scripts/download_models.sh scripts/download_face_models.sh scripts/download_classification_model.sh scripts/build.sh && \
    ./scripts/download_models.sh && \
    ./scripts/download_face_models.sh && \
    ./scripts/download_classification_model.sh

# Clean and build the project
RUN rm -rf build && \
    mkdir -p build && \
    cd build && \
    # Force display of build output to see if ONNXRuntime is found
    cmake -DCMAKE_VERBOSE_MAKEFILE=ON .. && \
    make -j$(nproc)

# Expose port
EXPOSE 8080

# Run the server
CMD ["./build/src/tAI_server"] 