#!/bin/bash

# Script to download object detection models for tAI
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( dirname "$SCRIPT_DIR" )"
MODELS_DIR="$PROJECT_DIR/models"

mkdir -p "$MODELS_DIR"

echo "Downloading models to $MODELS_DIR"

# Download YOLOv3 weights and cfg
echo "Downloading YOLOv3..."
wget -O "$MODELS_DIR/yolov3.weights" https://pjreddie.com/media/files/yolov3.weights
wget -O "$MODELS_DIR/yolov3.cfg" https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
wget -O "$MODELS_DIR/coco.names" https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

# Download YOLOv4 weights and cfg
echo "Downloading YOLOv4..."
wget -O "$MODELS_DIR/yolov4.weights" https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
wget -O "$MODELS_DIR/yolov4.cfg" https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg

# Download YOLOv4-tiny weights and cfg
echo "Downloading YOLOv4-tiny..."
wget -O "$MODELS_DIR/yolov4-tiny.weights" https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
wget -O "$MODELS_DIR/yolov4-tiny.cfg" https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg

# Download YOLOv3-tiny weights and cfg
echo "Downloading YOLOv3-tiny..."
wget -O "$MODELS_DIR/yolov3-tiny.weights" https://pjreddie.com/media/files/yolov3-tiny.weights
wget -O "$MODELS_DIR/yolov3-tiny.cfg" https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg

# Download MobileNet-SSD
echo "Downloading MobileNet-SSD..."
mkdir -p "$MODELS_DIR/mobilenet-ssd"
wget -O "$MODELS_DIR/mobilenet-ssd/mobilenet-ssd.caffemodel" https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/mobilenet_iter_73000.caffemodel
wget -O "$MODELS_DIR/mobilenet-ssd/mobilenet-ssd.prototxt" https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt
wget -O "$MODELS_DIR/mobilenet-ssd/mobilenet-ssd.labels" https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/labelmap_voc.prototxt

# Download YOLOX-Nano
echo "Downloading YOLOX-Nano..."
mkdir -p "$MODELS_DIR/yolox-nano"
wget -O "$MODELS_DIR/yolox-nano/yolox_nano.onnx" https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx

# Download NanoDet
echo "Downloading NanoDet..."
mkdir -p "$MODELS_DIR/nanodet"
wget -O "$MODELS_DIR/nanodet/nanodet-m.onnx" https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/nanodet-m.onnx

echo "All models downloaded."
echo "Available models:"
ls -la "$MODELS_DIR" 