#include "RESTServer.hpp"
#include "ObjectDetector.hpp"
#include "FaceDetector.hpp"
#include "ImageClassifier.hpp"
#include <iostream>
#include <string>
#include <filesystem>

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
        // Get the project directory
        fs::path exePath = fs::canonical("/proc/self/exe");
        fs::path projectDir = exePath.parent_path().parent_path().parent_path();  // Go up one more level
        fs::path modelsDir = projectDir / "models";

        std::cout << "Looking for models in: " << modelsDir << std::endl;

        // Load YOLOv3
        std::string yolov3Base = (modelsDir / "yolov3").string();
        loadModel("yolov3", yolov3Base);

        // Load YOLOv4
        std::string yolov4Base = (modelsDir / "yolov4").string();
        loadModel("yolov4", yolov4Base);

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
        std::cout << "      \"model_id\": \"yolov3\" or \"yolov4\"," << std::endl;
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
        
        server.start();
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
} 