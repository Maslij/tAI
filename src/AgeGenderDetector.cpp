#include "AgeGenderDetector.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/cuda.hpp>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

namespace tAI {

AgeGenderDetector::AgeGenderDetector() : modelsLoaded_(false) {
}

AgeGenderDetector::~AgeGenderDetector() {
}

bool AgeGenderDetector::loadModel(const std::string& modelPath) {
    try {
        // Check if the model directory exists
        if (!fs::exists(modelPath)) {
            std::cerr << "Model path does not exist: " << modelPath << std::endl;
            return false;
        }

        // Paths to the age and gender model files
        std::string agePath = fs::path(modelPath) / "age_googlenet.onnx";
        std::string genderPath = fs::path(modelPath) / "gender_googlenet.onnx";

        // Check if the model files exist
        if (!fs::exists(agePath) || !fs::exists(genderPath)) {
            std::cerr << "Age or gender model file not found in: " << modelPath << std::endl;
            return false;
        }

        // Load the age network
        ageNet_ = cv::dnn::readNetFromONNX(agePath);
        if (ageNet_.empty()) {
            std::cerr << "Failed to load age model from: " << agePath << std::endl;
            return false;
        }

        // Load the gender network
        genderNet_ = cv::dnn::readNetFromONNX(genderPath);
        if (genderNet_.empty()) {
            std::cerr << "Failed to load gender model from: " << genderPath << std::endl;
            return false;
        }

        // Use CUDA if available
        bool useGPU = false;
        try {
            // First check if CUDA is available
            if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
                // Try setting CUDA backend and target
                ageNet_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                ageNet_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
                genderNet_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                genderNet_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
                std::cout << "Using CUDA backend for age-gender detection" << std::endl;
                useGPU = true;
            } else {
                throw cv::Exception(0, "No CUDA devices found", "AgeGenderDetector::loadModel", __FILE__, __LINE__);
            }
        } catch (const cv::Exception& e) {
            std::cout << "CUDA not available or error setting up GPU inference, using CPU instead: " << e.what() << std::endl;
            ageNet_.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
            ageNet_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            genderNet_.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
            genderNet_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            std::cout << "Using CPU backend for age-gender detection" << std::endl;
        }

        modelsLoaded_ = true;
        std::cout << "Successfully loaded age and gender models" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading age-gender models: " << e.what() << std::endl;
        return false;
    }
}

cv::Mat AgeGenderDetector::preprocess(const cv::Mat& faceROI) {
    // Create a copy of the face region
    cv::Mat face = faceROI.clone();
    
    // Resize to the input size expected by the model (224x224)
    cv::resize(face, face, cv::Size(224, 224));
    
    // Convert to float and normalize
    cv::Mat blob = cv::dnn::blobFromImage(
        face, 
        1.0,                // scale factor
        cv::Size(224, 224), // spatial size
        cv::Scalar(104, 117, 123), // mean subtraction
        false,              // swapRB
        false               // crop
    );
    
    return blob;
}

std::vector<AgeGenderPrediction> AgeGenderDetector::predict(const cv::Mat& image, const std::vector<cv::Rect>& faces) {
    std::vector<AgeGenderPrediction> predictions;
    
    if (!modelsLoaded_) {
        std::cerr << "Age-gender models not loaded" << std::endl;
        return predictions;
    }
    
    if (image.empty() || faces.empty()) {
        return predictions;
    }
    
    for (const auto& faceRect : faces) {
        try {
            // Ensure face rectangle is within image bounds
            cv::Rect safeRect = faceRect & cv::Rect(0, 0, image.cols, image.rows);
            if (safeRect.width <= 0 || safeRect.height <= 0) {
                continue;
            }
            
            // Extract the face ROI
            cv::Mat faceROI = image(safeRect);
            
            // Preprocess the face for the network
            cv::Mat blob = preprocess(faceROI);
            
            // Age prediction
            ageNet_.setInput(blob);
            cv::Mat ageOutputs = ageNet_.forward();
            
            // Find the age class with the highest confidence
            cv::Point ageClassIdPoint;
            double ageConfidence;
            cv::minMaxLoc(ageOutputs, nullptr, &ageConfidence, nullptr, &ageClassIdPoint);
            int ageClassId = ageClassIdPoint.x;
            
            // Gender prediction
            genderNet_.setInput(blob);
            cv::Mat genderOutputs = genderNet_.forward();
            
            // Find the gender class with the highest confidence
            cv::Point genderClassIdPoint;
            double genderConfidence;
            cv::minMaxLoc(genderOutputs, nullptr, &genderConfidence, nullptr, &genderClassIdPoint);
            int genderClassId = genderClassIdPoint.x;
            
            // Create the prediction
            AgeGenderPrediction prediction;
            prediction.bbox = faceRect;
            prediction.gender = genderList[genderClassId];
            prediction.genderConfidence = static_cast<float>(genderConfidence);
            
            // Get the age range from the age list
            std::string ageRange = ageList[ageClassId];
            // Extract the average age from the range
            int startAge = 0, endAge = 0;
            sscanf(ageRange.c_str(), "(%d-%d)", &startAge, &endAge);
            prediction.age = (startAge + endAge) / 2;
            prediction.ageConfidence = static_cast<float>(ageConfidence);
            
            predictions.push_back(prediction);
        } catch (const cv::Exception& e) {
            std::cerr << "OpenCV error during age-gender prediction: " << e.what() << std::endl;
            continue;
        } catch (const std::exception& e) {
            std::cerr << "Error during age-gender prediction: " << e.what() << std::endl;
            continue;
        }
    }
    
    return predictions;
}

} // namespace tAI 