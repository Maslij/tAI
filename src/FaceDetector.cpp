#include "FaceDetector.hpp"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

namespace tAI {

class FaceDetector::Impl {
public:
    cv::dnn::Net net;
    float confThreshold = 0.5f;
    int inputWidth = 300;
    int inputHeight = 300;
    float meanValues[3] = {104.0, 177.0, 123.0};
};

FaceDetector::FaceDetector() : pImpl_(std::make_unique<Impl>()) {}
FaceDetector::~FaceDetector() = default;

bool FaceDetector::loadModel(const std::string& modelPath) {
    try {
        // The model path should point to the .prototxt file
        // The weights file should be in the same directory with .caffemodel extension
        fs::path prototxtPath = fs::path(modelPath);
        fs::path caffemodelPath = prototxtPath.parent_path() / (prototxtPath.stem().string() + ".caffemodel");
        
        if (!fs::exists(prototxtPath) || !fs::exists(caffemodelPath)) {
            std::cerr << "Model files not found: " << prototxtPath << " or " << caffemodelPath << std::endl;
            return false;
        }
        
        // Load the network
        pImpl_->net = cv::dnn::readNetFromCaffe(prototxtPath.string(), caffemodelPath.string());
        
        // Use CUDA if available
        try {
            pImpl_->net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            pImpl_->net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            std::cout << "Using CUDA backend for face detection" << std::endl;
        } catch (const cv::Exception& e) {
            std::cout << "CUDA not available for face detection, using CPU" << std::endl;
            pImpl_->net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
            pImpl_->net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }
        
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading face detection model: " << e.what() << std::endl;
        return false;
    }
}

std::vector<FaceDetection> FaceDetector::detect(const cv::Mat& image) {
    std::vector<FaceDetection> detections;
    
    try {
        // Prepare input blob and set input
        cv::Mat inputBlob = cv::dnn::blobFromImage(
            image, 1.0, 
            cv::Size(pImpl_->inputWidth, pImpl_->inputHeight), 
            cv::Scalar(pImpl_->meanValues[0], pImpl_->meanValues[1], pImpl_->meanValues[2]), 
            false, false);
        
        pImpl_->net.setInput(inputBlob);
        
        // Forward pass
        cv::Mat detection = pImpl_->net.forward();
        
        // Process detections
        cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
        
        for (int i = 0; i < detectionMat.rows; i++) {
            float confidence = detectionMat.at<float>(i, 2);
            
            if (confidence > pImpl_->confThreshold) {
                // Get face box dimensions
                int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * image.cols);
                int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * image.rows);
                int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * image.cols);
                int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * image.rows);
                
                // Ensure box is within image boundaries
                x1 = std::max(0, std::min(x1, image.cols - 1));
                y1 = std::max(0, std::min(y1, image.rows - 1));
                x2 = std::max(0, std::min(x2, image.cols - 1));
                y2 = std::max(0, std::min(y2, image.rows - 1));
                
                // Create face detection
                FaceDetection face;
                face.bbox = cv::Rect(x1, y1, x2 - x1, y2 - y1);
                face.confidence = confidence;
                
                detections.push_back(face);
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error during face detection: " << e.what() << std::endl;
    }
    
    return detections;
}

} // namespace tAI 