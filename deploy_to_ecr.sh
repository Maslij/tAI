#!/bin/bash
set -e

# Configuration
AWS_REGION="ap-southeast-2"
ECR_REGISTRY="246261010633.dkr.ecr.ap-southeast-2.amazonaws.com"
ECR_REPOSITORY="bbai-jetson"
IMAGE_TAG=$(date +%Y%m%d-%H%M%S)

echo "Building and deploying tAI ARM64 image to ECR..."
echo "Repository: ${ECR_REGISTRY}/${ECR_REPOSITORY}"
echo "Tag: ${IMAGE_TAG}"

# Authenticate Docker to ECR
echo "Authenticating with AWS ECR..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REGISTRY}

# Check if repository exists, create if not
aws ecr describe-repositories --repository-names ${ECR_REPOSITORY} --region ${AWS_REGION} > /dev/null 2>&1 || \
  aws ecr create-repository --repository-name ${ECR_REPOSITORY} --region ${AWS_REGION}

# Set up Docker Buildx for multi-architecture builds
echo "Setting up Docker Buildx..."
docker buildx inspect jetson-builder >/dev/null 2>&1 || docker buildx create --name jetson-builder --platform linux/arm64
docker buildx use jetson-builder
docker buildx inspect --bootstrap

# Build and push the image
echo "Building and pushing Docker image for ARM64 (NVIDIA Jetson)..."
docker buildx build --platform linux/arm64 \
  --tag ${ECR_REGISTRY}/${ECR_REPOSITORY}:${IMAGE_TAG} \
  --tag ${ECR_REGISTRY}/${ECR_REPOSITORY}:latest \
  --push \
  .

echo "Deployment completed successfully!"
echo "Image: ${ECR_REGISTRY}/${ECR_REPOSITORY}:${IMAGE_TAG}"
echo "Image: ${ECR_REGISTRY}/${ECR_REPOSITORY}:latest"
echo ""
echo "To pull this image on your Jetson device:"
echo "1. Authenticate with ECR:"
echo "   aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REGISTRY}"
echo ""
echo "2. Pull the image:"
echo "   docker pull ${ECR_REGISTRY}/${ECR_REPOSITORY}:latest"
echo ""
echo "3. Run the container:"
echo "   docker run --rm -it --gpus all -p 8080:8080 ${ECR_REGISTRY}/${ECR_REPOSITORY}:latest" 