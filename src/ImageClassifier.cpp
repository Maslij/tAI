#include "ImageClassifier.hpp"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <fstream>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

namespace tAI {

class CVImageClassifier::Impl {
public:
    Impl() = default;
    ~Impl() = default;

    bool loadModel(const std::string& modelPath) {
        try {
            // Expect modelPath to be the base path, without extension
            fs::path basePath = modelPath;
            fs::path modelProtoPath = basePath.parent_path() / "deploy.prototxt";
            fs::path modelWeightsPath = basePath.parent_path() / "deploy.caffemodel";
            fs::path classNamesPath = basePath.parent_path() / "classes.txt";
            
            // Load the DNN model
            net_ = cv::dnn::readNetFromCaffe(modelProtoPath.string(), modelWeightsPath.string());
            
            // Check if GPU/CUDA is available and use it
            bool useGPU = false;
            try {
                // First check if CUDA is available
                if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
                    // Try setting CUDA backend and target
                    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
                    std::cout << "Using CUDA for image classification model" << std::endl;
                    useGPU = true;
                } else {
                    throw cv::Exception(0, "No CUDA devices found", "CVImageClassifier::loadModel", __FILE__, __LINE__);
                }
            } catch (const cv::Exception& e) {
                std::cout << "CUDA not available or error setting up GPU inference, using CPU instead: " << e.what() << std::endl;
                net_.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
                net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
                std::cout << "Using CPU backend for image classification" << std::endl;
            }
            
            // Load class names
            classNames_.clear();
            std::ifstream classNamesFile(classNamesPath);
            if (classNamesFile.is_open()) {
                std::string line;
                while (std::getline(classNamesFile, line)) {
                    classNames_.push_back(line);
                }
                classNamesFile.close();
            } else {
                std::cerr << "Failed to open class names file: " << classNamesPath << std::endl;
                return false;
            }
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error loading classification model: " << e.what() << std::endl;
            return false;
        }
    }

    std::vector<Classification> classify(const cv::Mat& image) {
        std::vector<Classification> classifications;
        
        try {
            // Preprocess the image - resize to 224x224 which is standard for many classification models
            cv::Mat blob = cv::dnn::blobFromImage(image, 1.0, cv::Size(224, 224), 
                                                 cv::Scalar(104, 117, 123), // Subtract mean values
                                                 true, false);
            
            // Set the input and forward pass
            net_.setInput(blob);
            cv::Mat prob = net_.forward();
            
            // Get the results - find top N predictions
            const int topN = 5; // Return top 5 classes
            cv::Mat probMat = prob.reshape(1, 1);
            
            // Get indices of top responses
            cv::Mat sorted;
            cv::sortIdx(probMat, sorted, cv::SORT_EVERY_ROW + cv::SORT_DESCENDING);
            
            for (int i = 0; i < std::min(topN, static_cast<int>(classNames_.size())); i++) {
                int idx = sorted.at<int>(i);
                
                // Only include classifications with decent confidence
                float confidence = probMat.at<float>(idx);
                if (confidence > 0.01f) {  // 1% minimum threshold
                    Classification cls;
                    cls.className = idx < classNames_.size() ? classNames_[idx] : "Unknown";
                    cls.confidence = confidence;
                    classifications.push_back(cls);
                }
            }
        } catch (const cv::Exception& e) {
            std::cerr << "Error during classification: " << e.what() << std::endl;
        }
        
        return classifications;
    }

    std::vector<std::string> getClassNames() const {
        return classNames_;
    }

private:
    cv::dnn::Net net_;
    std::vector<std::string> classNames_;
};

// Public interface implementation

CVImageClassifier::CVImageClassifier() : pImpl_(std::make_unique<Impl>()) {}

CVImageClassifier::~CVImageClassifier() = default;

std::vector<Classification> CVImageClassifier::classify(const cv::Mat& image) {
    return pImpl_->classify(image);
}

bool CVImageClassifier::loadModel(const std::string& modelPath) {
    return pImpl_->loadModel(modelPath);
}

std::vector<std::string> CVImageClassifier::getClassNames() const {
    return pImpl_->getClassNames();
}

} // namespace tAI 