#!/bin/bash
set -e

# Create model directory
mkdir -p ../models/classification

# Download the pre-trained model
echo "Downloading classification model files..."

# Download model files
wget https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/bvlc_googlenet.prototxt -O ../models/classification/deploy.prototxt
wget http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel -O ../models/classification/deploy.caffemodel

# Download the class names (ImageNet classes)
echo "Downloading class names..."
wget https://raw.githubusercontent.com/HoldenCaulfieldRye/caffe/master/data/ilsvrc12/synset_words.txt -O ../models/classification/classes.txt

# Make the script executable
chmod +x ../models/classification/deploy.prototxt

echo "Classification model downloaded successfully!"
echo "Model: GoogLeNet (ImageNet)"
echo "Location: ../models/classification/" 