#!/bin/bash

# Create directory for face detection models if it doesn't exist
FACE_MODEL_DIR="$(pwd)/models/face_detection"
mkdir -p $FACE_MODEL_DIR

echo "Downloading face detection model files to $FACE_MODEL_DIR"

# Download face detection model files (using a caffemodel-compatible face detector)
wget -q https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/deploy.prototxt -O $FACE_MODEL_DIR/deploy.prototxt
wget -q https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel -O $FACE_MODEL_DIR/deploy.caffemodel

# Check if downloads succeeded
if [ -f "$FACE_MODEL_DIR/deploy.prototxt" ] && [ -f "$FACE_MODEL_DIR/deploy.caffemodel" ]; then
    echo "Face detection model files downloaded successfully!"
    chmod +x $FACE_MODEL_DIR/deploy.prototxt
    chmod +x $FACE_MODEL_DIR/deploy.caffemodel
else
    echo "Failed to download face detection model files."
    exit 1
fi 