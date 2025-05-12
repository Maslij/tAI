#include "ObjectDetector.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/cuda.hpp>
#include <fstream>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

namespace tAI {

class YOLODetector::Impl {
public:
#ifndef USE_ONNXRUNTIME
    cv::dnn::Net net;
#endif
    std::vector<std::string> classNames;
    float confThreshold = 0.5f;
    float nmsThreshold = 0.4f;
#ifdef USE_ONNXRUNTIME
    int inputWidth = 640;
    int inputHeight = 640;
#else
    int inputWidth = 416;
    int inputHeight = 416;
    
    void postprocess(const cv::Mat& frame, const std::vector<cv::Mat>& outs, std::vector<Detection>& detections);
#endif
};

YOLODetector::YOLODetector() : pImpl_(std::make_unique<Impl>()) {}
YOLODetector::~YOLODetector() = default;

#ifdef USE_ONNXRUNTIME
bool YOLODetector::loadModel(const std::string& modelPath) {
    try {
        // First call the base class implementation to load the ONNX model
        if (!ONNXObjectDetector::loadModel(modelPath)) {
            return false;
        }
        
        // Load class names from coco.names
        fs::path modelDir = fs::path(modelPath).parent_path();
        std::string classesFile = (modelDir / "coco.names").string();
        std::ifstream ifs(classesFile.c_str());
        
        if (!ifs.is_open()) {
            std::cerr << "Failed to open classes file: " << classesFile << std::endl;
            std::cerr << "Will use default class names based on model output" << std::endl;
            
            // Get number of classes from model output shape
            if (!output_shapes_.empty() && output_shapes_[0].size() >= 3) {
                int numClasses = output_shapes_[0][2] - 5; // Typical YOLO output format
                for (int i = 0; i < numClasses; i++) {
                    pImpl_->classNames.push_back("class" + std::to_string(i));
                }
            }
        } else {
            std::string line;
            while (std::getline(ifs, line)) {
                pImpl_->classNames.push_back(line);
            }
        }
        
        std::cout << "Loaded YOLO model with " << pImpl_->classNames.size() << " classes" << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading YOLO model: " << e.what() << std::endl;
        return false;
    }
}
#else
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
        bool useGPU = false;
        try {
            // First check if CUDA is available
            if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
                // Try setting CUDA backend and target
                pImpl_->net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                pImpl_->net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
                std::cout << "Using CUDA backend for YOLOv4 detection" << std::endl;
                useGPU = true;
            } else {
                throw cv::Exception(0, "No CUDA devices found", "YOLODetector::loadModel", __FILE__, __LINE__);
            }
        } catch (const cv::Exception& e) {
            std::cout << "CUDA not available or error setting up GPU inference, using CPU instead: " << e.what() << std::endl;
            pImpl_->net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
            pImpl_->net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            std::cout << "Using CPU backend for YOLOv4 detection" << std::endl;
        }
        
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return false;
    }
}
#endif

std::vector<Detection> YOLODetector::detect(const cv::Mat& image) {
    std::vector<Detection> detections;
    
    try {
#ifdef USE_ONNXRUNTIME
        if (!session_) {
            throw std::runtime_error("Model not loaded");
        }
        
        // Preprocess image
        cv::Mat resizedImage, floatImage;
        cv::resize(image, resizedImage, cv::Size(pImpl_->inputWidth, pImpl_->inputHeight));
        
        // Convert to float32, NCHW layout format (batch, channels, height, width)
        resizedImage.convertTo(floatImage, CV_32F, 1.0/255.0);
        
        // Create tensor for input data
        std::vector<float> input_tensor_values;
        input_tensor_values.reserve(pImpl_->inputWidth * pImpl_->inputHeight * 3);
        
        // NHWC to NCHW conversion (assuming RGB image)
        std::vector<cv::Mat> channels(3);
        cv::split(floatImage, channels);
        
        // HWC to CHW conversion
        for (int c = 0; c < 3; c++) {
            const float* data = channels[c].ptr<float>();
            input_tensor_values.insert(input_tensor_values.end(), data, data + pImpl_->inputWidth * pImpl_->inputHeight);
        }
        
        // Define input tensor shape (NCHW format)
        std::vector<int64_t> input_shape = {1, 3, pImpl_->inputHeight, pImpl_->inputWidth};
        
        // Create input tensor
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_tensor_values.data(),
            input_tensor_values.size(),
            input_shape.data(),
            input_shape.size()
        );
        
        // Run inference
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            getInputNames().data(),
            &input_tensor,
            1,
            getOutputNames().data(),
            getOutputNames().size()
        );
        
        // Process detections
        detections = processDetections(output_tensors, image.size());
#else
        cv::Mat blob;
        cv::dnn::blobFromImage(image, blob, 1/255.0, 
                              cv::Size(pImpl_->inputWidth, pImpl_->inputHeight),
                              cv::Scalar(0,0,0), true, false);
        
        pImpl_->net.setInput(blob);
        
        std::vector<cv::Mat> outs;
        pImpl_->net.forward(outs, pImpl_->net.getUnconnectedOutLayersNames());
        
        pImpl_->postprocess(image, outs, detections);
#endif
    }
    catch (const std::exception& e) {
        std::cerr << "Error during detection: " << e.what() << std::endl;
    }
    
    return detections;
}

#ifdef USE_ONNXRUNTIME
std::vector<Detection> YOLODetector::processDetections(
    const std::vector<Ort::Value>& output_tensors,
    const cv::Size& original_image_size) 
{
    std::vector<Detection> detections;
    
    // Check if output is valid
    if (output_tensors.empty() || !output_tensors[0].IsTensor()) {
        return detections;
    }
    
    // Get pointer to output tensor
    const float* output_data = output_tensors[0].GetTensorData<float>();
    
    // Get output tensor dimensions
    auto tensor_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    auto output_shape = tensor_info.GetShape();
    
    // For YOLOv8 output format (boxes, scores, class indices)
    // YOLO outputs are typically in the format [batch, num_detections, xywh+confidence+num_classes]
    int num_detections = output_shape[1]; // Number of detected boxes
    int elements_per_detection = output_shape[2]; // Elements per detection (xywh + conf + classes)
    
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    
    float x_factor = static_cast<float>(original_image_size.width) / pImpl_->inputWidth;
    float y_factor = static_cast<float>(original_image_size.height) / pImpl_->inputHeight;
    
    for (int i = 0; i < num_detections; i++) {
        const float* detection = output_data + i * elements_per_detection;
        
        float confidence = detection[4];
        
        if (confidence >= pImpl_->confThreshold) {
            // Find the class with highest confidence
            int class_id = 0;
            float max_class_score = 0;
            for (int j = 5; j < elements_per_detection; j++) {
                if (detection[j] > max_class_score) {
                    max_class_score = detection[j];
                    class_id = j - 5;
                }
            }
            
            // Calculate scaled bounding box
            float x = detection[0] * x_factor;
            float y = detection[1] * y_factor;
            float w = detection[2] * x_factor;
            float h = detection[3] * y_factor;
            
            // Convert to top-left coordinates
            int left = int(x - w / 2);
            int top = int(y - h / 2);
            
            boxes.push_back(cv::Rect(left, top, int(w), int(h)));
            confidences.push_back(confidence);
            class_ids.push_back(class_id);
        }
    }
    
    // Apply Non-maximum Suppression
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, pImpl_->confThreshold, pImpl_->nmsThreshold, indices);
    
    // Create detection objects
    for (int idx : indices) {
        Detection det;
        det.bbox = boxes[idx];
        det.confidence = confidences[idx];
        det.classId = class_ids[idx];
        
        // Ensure class_id is valid
        if (class_ids[idx] < pImpl_->classNames.size()) {
            det.className = pImpl_->classNames[class_ids[idx]];
        } else {
            det.className = "unknown";
        }
        
        detections.push_back(det);
    }
    
    return detections;
}
#else
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
#endif

std::vector<std::string> YOLODetector::getClassNames() const {
    return pImpl_->classNames;
}

} // namespace tAI 