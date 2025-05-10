#include "RESTServer.hpp"
#include "ObjectDetector.hpp"
#include "FaceDetector.hpp"
#include "ImageClassifier.hpp"
#include "AgeGenderDetector.hpp"
#include <iostream>
#include <string>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

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

void loadImageClassifier(const std::string& modelName, const std::string& modelPath, const std::string& modelId = "googlenet") {
    auto classifier = std::make_shared<tAI::CVImageClassifier>();
    if (!classifier->loadModel(modelPath, modelId)) {
        std::cerr << "Failed to load " << modelName << " from " << modelPath << std::endl;
        return;
    }
    tAI::ModelManager::getInstance().registerModel(modelName, classifier);
    std::cout << "Successfully loaded " << modelName << " (" << modelId << ")" << std::endl;
}

void loadAgeGenderModel(const std::string& modelName, const std::string& modelPath) {
    auto ageGenderDetector = std::make_shared<tAI::AgeGenderDetector>();
    if (!ageGenderDetector->loadModel(modelPath)) {
        std::cerr << "Failed to load " << modelName << " from " << modelPath << std::endl;
        return;
    }
    tAI::ModelManager::getInstance().registerModel(modelName, ageGenderDetector);
    std::cout << "Successfully loaded " << modelName << std::endl;
}

int main() {
    try {
        // Print OpenCV version and CUDA information
        std::cout << "OpenCV Version: " << CV_VERSION << std::endl;
        
        // Check CUDA availability
        int cudaDeviceCount = cv::cuda::getCudaEnabledDeviceCount();
        std::cout << "OpenCV CUDA Support: " << cudaDeviceCount << " CUDA device(s) found" << std::endl;
        
        if (cudaDeviceCount > 0) {
            try {
                cv::cuda::printCudaDeviceInfo(0);  // Print info for the first CUDA device
                std::cout << "CUDA capability available, but models will check compatibility individually" << std::endl;
            } catch (const cv::Exception& e) {
                std::cout << "Warning: CUDA devices found but error getting device info: " << e.what() << std::endl;
                std::cout << "Will fall back to CPU for all operations" << std::endl;
            }
        } else {
            std::cout << "No CUDA devices found. Running in CPU-only mode." << std::endl;
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
        std::string facePath = (modelsDir / "face_detection" / "deploy.prototxt").string();
        loadFaceModel("face_detection", facePath);

        // Load image classifier model
        std::string classificationPath = (modelsDir / "classification").string();
        loadImageClassifier("image_classification", classificationPath);

        // Load age and gender detection models
        std::string ageGenderPath = (modelsDir / "age_gender").string();
        loadAgeGenderModel("age_gender_detection", ageGenderPath);

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
        std::cout << "  POST /detect_age_gender" << std::endl;
        std::cout << "    Request body: {" << std::endl;
        std::cout << "      \"model_id\": \"age_gender_detection\"," << std::endl;
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