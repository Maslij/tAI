#include "RESTServer.hpp"
#include "ObjectDetector.hpp"
#include "FaceDetector.hpp"
#include "ImageClassifier.hpp"
#include <iostream>
#include <string>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

namespace fs = std::filesystem;

void loadModel(const std::string& modelName, const std::string& modelPath) {
    auto yolo = std::make_shared<tAI::YOLODetector>();
    if (!yolo->loadModel(modelPath)) {
        std::cerr << "Failed to load " << modelName << " from " << modelPath << std::endl;
        return;
    }
    tAI::ModelManager::getInstance().registerModel(modelName, yolo);
    std::cout << "Successfully loaded " << modelName << std::endl;
}

void loadFaceModel(const std::string& modelName, const std::string& modelPath) {
    auto faceDetector = std::make_shared<tAI::FaceDetector>();
    if (!faceDetector->loadModel(modelPath)) {
        std::cerr << "Failed to load " << modelName << " from " << modelPath << std::endl;
        return;
    }
    tAI::ModelManager::getInstance().registerModel(modelName, faceDetector);
    std::cout << "Successfully loaded " << modelName << std::endl;
}

void loadImageClassifier(const std::string& modelName, const std::string& modelPath) {
    auto classifier = std::make_shared<tAI::CVImageClassifier>();
    if (!classifier->loadModel(modelPath)) {
        std::cerr << "Failed to load " << modelName << " from " << modelPath << std::endl;
        return;
    }
    tAI::ModelManager::getInstance().registerModel(modelName, classifier);
    std::cout << "Successfully loaded " << modelName << std::endl;
}

int main() {
    try {
        // Print OpenCV version and CUDA information
        std::cout << "OpenCV Version: " << CV_VERSION << std::endl;
        std::cout << "OpenCV CUDA Support: " << cv::cuda::getCudaEnabledDeviceCount() << " CUDA device(s) found" << std::endl;
        if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
            cv::cuda::printCudaDeviceInfo(0);  // Print info for the first CUDA device
        }
        std::cout << std::endl;

        // Get the project directory
        fs::path exePath = fs::canonical("/proc/self/exe");
        fs::path projectDir = exePath.parent_path().parent_path().parent_path();  // Go up one more level
        fs::path modelsDir = projectDir / "models";

        std::cout << "Looking for models in: " << modelsDir << std::endl;

        // Load YOLOv4
        std::string yolov4Base = (modelsDir / "yolov4").string();
        loadModel("yolov4", yolov4Base);

        // Load YOLOv4-tiny
        std::string yolov4TinyBase = (modelsDir / "yolov4-tiny").string();
        loadModel("yolov4-tiny", yolov4TinyBase);
        
        // Load face detection model - assuming model file is deploy.prototxt
        std::string faceModelPath = (modelsDir / "face_detection" / "deploy.prototxt").string();
        loadFaceModel("face_detection", faceModelPath);

        // Load image classification model
        std::string classificationPath = (modelsDir / "classification" / "deploy").string();
        loadImageClassifier("image_classification", classificationPath);

        // Create and start the REST server
        tAI::RESTServer server("0.0.0.0", 8080);
        
        std::cout << "Starting server on http://0.0.0.0:8080" << std::endl;
        std::cout << "Available endpoints:" << std::endl;
        std::cout << "  POST /detect" << std::endl;
        std::cout << "    Request body: {" << std::endl;
        std::cout << "      \"model_id\": \"yolov4\" or \"yolov4-tiny\"," << std::endl;
        std::cout << "      \"image\": \"<base64_encoded_image>\"" << std::endl;
        std::cout << "    }" << std::endl;
        std::cout << "  POST /detect_faces" << std::endl;
        std::cout << "    Request body: {" << std::endl;
        std::cout << "      \"model_id\": \"face_detection\"," << std::endl;
        std::cout << "      \"image\": \"<base64_encoded_image>\"" << std::endl;
        std::cout << "    }" << std::endl;
        std::cout << "  POST /classify" << std::endl;
        std::cout << "    Request body: {" << std::endl;
        std::cout << "      \"model_id\": \"image_classification\"," << std::endl;
        std::cout << "      \"image\": \"<base64_encoded_image>\"" << std::endl;
        std::cout << "    }" << std::endl;
        std::cout << "  GET /module_health" << std::endl;
        std::cout << "    Returns: JSON with status of all loaded models" << std::endl;
        
        server.start();
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
} 