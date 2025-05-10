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

# Update the CMakeLists.txt to use the correct versions
RUN sed -i 's|set(OpenCV_DIR "/home/alec/Downloads/opencv/build")|set(OpenCV_DIR "/usr/lib/aarch64-linux-gnu/cmake/opencv4")|g' CMakeLists.txt && \
    sed -i 's|find_package(OpenCV 4.8 REQUIRED)|find_package(OpenCV 4.5 REQUIRED)|g' CMakeLists.txt && \
    sed -i 's|find_package(nlohmann_json 3.10.5 REQUIRED)|find_package(nlohmann_json 3.7.3 REQUIRED)|g' CMakeLists.txt

# Download models
RUN chmod +x scripts/download_models.sh scripts/download_face_models.sh scripts/download_classification_model.sh scripts/build.sh && \
    ./scripts/download_models.sh && \
    ./scripts/download_face_models.sh && \
    ./scripts/download_classification_model.sh

# Clean and build the project
RUN rm -rf build && \
    mkdir -p build && \
    cd build && \
    cmake .. && \
    make -j$(nproc)

# Expose port
EXPOSE 8080

# Run the server
CMD ["./build/src/tAI_server"] 