#include "ImageClassifier.hpp"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <unordered_map>

namespace fs = std::filesystem;

namespace tAI {

// Define supported model types
enum class ClassifierModelType {
    GOOGLENET,
    RESNET50,
    MOBILENET,
    // Add more as needed
};

class CVImageClassifier::Impl {
public:
    Impl() = default;
    ~Impl() = default;

    bool loadModel(const std::string& modelPath, const std::string& modelId = "googlenet") {
        try {
            // Convert modelId to model type
            ClassifierModelType modelType = modelIdToType(modelId);
            
            // Get model configuration based on type
            ModelConfig config = getModelConfig(modelType, modelPath);
            
            // Load the DNN model
            net_ = cv::dnn::readNet(config.protoPath, config.weightsPath);
            
            // Check if GPU/CUDA is available and use it
            bool useGPU = false;
            try {
                // First check if CUDA is available
                if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
                    // Try setting CUDA backend and target
                    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
                    std::cout << "Using CUDA for image classification model: " << modelId << std::endl;
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
            std::ifstream classNamesFile(config.classesPath);
            if (classNamesFile.is_open()) {
                std::string line;
                while (std::getline(classNamesFile, line)) {
                    classNames_.push_back(line);
                }
                classNamesFile.close();
            } else {
                std::cerr << "Failed to open class names file: " << config.classesPath << std::endl;
                return false;
            }
            
            // Store current model configuration
            currentModelType_ = modelType;
            currentModelId_ = modelId;
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error loading classification model: " << e.what() << std::endl;
            return false;
        }
    }

    std::vector<Classification> classify(const cv::Mat& image) {
        std::vector<Classification> classifications;
        
        try {
            // Get model-specific preprocessing parameters
            PreprocessParams params = getPreprocessParams(currentModelType_);
            
            // Preprocess the image based on model type
            cv::Mat blob = cv::dnn::blobFromImage(image, params.scale, 
                                                  params.size,
                                                  params.mean,
                                                  params.swapRB,
                                                  params.crop);
            
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
    
    std::string getCurrentModelId() const {
        return currentModelId_;
    }

private:
    struct ModelConfig {
        std::string protoPath;
        std::string weightsPath; 
        std::string classesPath;
    };
    
    struct PreprocessParams {
        double scale;
        cv::Size size;
        cv::Scalar mean;
        bool swapRB;
        bool crop;
    };
    
    ClassifierModelType modelIdToType(const std::string& modelId) {
        static const std::unordered_map<std::string, ClassifierModelType> modelMap = {
            {"googlenet", ClassifierModelType::GOOGLENET},
            {"resnet50", ClassifierModelType::RESNET50},
            {"mobilenet", ClassifierModelType::MOBILENET}
            // Add more as needed
        };
        
        auto it = modelMap.find(modelId);
        if (it != modelMap.end()) {
            return it->second;
        }
        // Default to GoogLeNet if not found
        std::cerr << "Warning: Unknown model ID '" << modelId << "', defaulting to googlenet" << std::endl;
        return ClassifierModelType::GOOGLENET;
    }
    
    ModelConfig getModelConfig(ClassifierModelType type, const std::string& basePath) {
        ModelConfig config;
        fs::path baseDir = fs::path(basePath);
        
        // Make sure basePath is treated as a directory path, not as a file path
        if (!fs::is_directory(baseDir)) {
            baseDir = baseDir.parent_path();
        }
        
        switch (type) {
            case ClassifierModelType::GOOGLENET:
                config.protoPath = (baseDir / "deploy.prototxt").string();
                config.weightsPath = (baseDir / "deploy.caffemodel").string();
                config.classesPath = (baseDir / "classes.txt").string();
                break;
                
            case ClassifierModelType::RESNET50:
                config.protoPath = (baseDir / "resnet50.prototxt").string();
                config.weightsPath = (baseDir / "resnet50.caffemodel").string();
                config.classesPath = (baseDir / "classes.txt").string();
                break;
                
            case ClassifierModelType::MOBILENET:
                config.protoPath = (baseDir / "mobilenet.prototxt").string();
                config.weightsPath = (baseDir / "mobilenet.caffemodel").string();
                config.classesPath = (baseDir / "classes.txt").string();
                break;
                
            default:
                // Default to GoogLeNet
                config.protoPath = (baseDir / "deploy.prototxt").string();
                config.weightsPath = (baseDir / "deploy.caffemodel").string();
                config.classesPath = (baseDir / "classes.txt").string();
                break;
        }
        
        return config;
    }
    
    PreprocessParams getPreprocessParams(ClassifierModelType type) {
        PreprocessParams params;
        
        switch (type) {
            case ClassifierModelType::GOOGLENET:
                params.scale = 1.0;
                params.size = cv::Size(224, 224);
                params.mean = cv::Scalar(104, 117, 123);
                params.swapRB = true;
                params.crop = false;
                break;
                
            case ClassifierModelType::RESNET50:
                params.scale = 1.0;
                params.size = cv::Size(224, 224);
                params.mean = cv::Scalar(104, 117, 123);
                params.swapRB = true;
                params.crop = false;
                break;
                
            case ClassifierModelType::MOBILENET:
                params.scale = 1.0/255.0;
                params.size = cv::Size(224, 224);
                params.mean = cv::Scalar(0, 0, 0);
                params.swapRB = true; 
                params.crop = false;
                break;
                
            default:
                // Default to GoogLeNet parameters
                params.scale = 1.0;
                params.size = cv::Size(224, 224);
                params.mean = cv::Scalar(104, 117, 123);
                params.swapRB = true;
                params.crop = false;
                break;
        }
        
        return params;
    }

    cv::dnn::Net net_;
    std::vector<std::string> classNames_;
    ClassifierModelType currentModelType_ = ClassifierModelType::GOOGLENET;
    std::string currentModelId_ = "googlenet";
};

// Public interface implementation

CVImageClassifier::CVImageClassifier() : pImpl_(std::make_unique<Impl>()) {}

CVImageClassifier::~CVImageClassifier() = default;

std::vector<Classification> CVImageClassifier::classify(const cv::Mat& image) {
    return pImpl_->classify(image);
}

bool CVImageClassifier::loadModel(const std::string& modelPath) {
    // Default to GoogleNet for backward compatibility
    return pImpl_->loadModel(modelPath, "googlenet");
}

bool CVImageClassifier::loadModel(const std::string& modelPath, const std::string& modelId) {
    return pImpl_->loadModel(modelPath, modelId);
}

std::vector<std::string> CVImageClassifier::getClassNames() const {
    return pImpl_->getClassNames();
}

std::string CVImageClassifier::getCurrentModelId() const {
    return pImpl_->getCurrentModelId();
}

} // namespace tAI 