#include "ObjectDetector.hpp"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

namespace tAI {

class YOLODetector::Impl {
public:
    cv::dnn::Net net;
    std::vector<std::string> classNames;
    float confThreshold = 0.5f;
    float nmsThreshold = 0.4f;
    int inputWidth = 416;
    int inputHeight = 416;
    
    void postprocess(const cv::Mat& frame, const std::vector<cv::Mat>& outs, std::vector<Detection>& detections);
};

YOLODetector::YOLODetector() : pImpl_(std::make_unique<Impl>()) {}
YOLODetector::~YOLODetector() = default;

bool YOLODetector::loadModel(const std::string& modelPath) {
    try {
        // Load names of classes from coco.names
        fs::path modelDir = fs::path(modelPath).parent_path();
        std::string classesFile = (modelDir / "coco.names").string();
        std::ifstream ifs(classesFile.c_str());
        if (!ifs.is_open()) {
            std::cerr << "Failed to open classes file: " << classesFile << std::endl;
            return false;
        }
        
        std::string line;
        while (std::getline(ifs, line)) {
            pImpl_->classNames.push_back(line);
        }
        
        // Load the network
        std::string configFile = modelPath + ".cfg";
        std::string weightsFile = modelPath + ".weights";
        
        if (!fs::exists(configFile) || !fs::exists(weightsFile)) {
            std::cerr << "Config or weights file not found" << std::endl;
            return false;
        }
        
        pImpl_->net = cv::dnn::readNetFromDarknet(configFile, weightsFile);
        
        // Use CUDA if available
        try {
            pImpl_->net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            pImpl_->net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            std::cout << "Using CUDA backend" << std::endl;
        } catch (const cv::Exception& e) {
            std::cout << "CUDA not available, using CPU" << std::endl;
            pImpl_->net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
            pImpl_->net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }
        
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return false;
    }
}

std::vector<Detection> YOLODetector::detect(const cv::Mat& image) {
    std::vector<Detection> detections;
    
    try {
        cv::Mat blob;
        cv::dnn::blobFromImage(image, blob, 1/255.0, 
                              cv::Size(pImpl_->inputWidth, pImpl_->inputHeight),
                              cv::Scalar(0,0,0), true, false);
        
        pImpl_->net.setInput(blob);
        
        std::vector<cv::Mat> outs;
        pImpl_->net.forward(outs, pImpl_->net.getUnconnectedOutLayersNames());
        
        pImpl_->postprocess(image, outs, detections);
    }
    catch (const std::exception& e) {
        std::cerr << "Error during detection: " << e.what() << std::endl;
    }
    
    return detections;
}

std::vector<std::string> YOLODetector::getClassNames() const {
    return pImpl_->classNames;
}

void YOLODetector::Impl::postprocess(const cv::Mat& frame, const std::vector<cv::Mat>& outs, std::vector<Detection>& detections) {
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    
    for (const auto& out : outs) {
        for (int i = 0; i < out.rows; ++i) {
            const float* data = (float*)out.row(i).data;
            
            cv::Mat scores = out.row(i).colRange(5, out.cols);
            cv::Point classIdPoint;
            double confidence;
            cv::minMaxLoc(scores, nullptr, &confidence, nullptr, &classIdPoint);
            
            if (confidence > confThreshold) {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }
    
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        Detection det;
        det.bbox = boxes[idx];
        det.confidence = confidences[idx];
        det.classId = classIds[idx];
        det.className = classNames[classIds[idx]];
        detections.push_back(det);
    }
}
} 