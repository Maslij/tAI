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
    float confThreshold = 0.25f;
    float nmsThreshold = 0.4f;
#ifdef USE_ONNXRUNTIME
    int inputWidth = 416;
    int inputHeight = 416;
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
    if (output_tensors.empty()) {
        return detections;
    }
    
    // Setup variables for detection results
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    
    float x_factor = static_cast<float>(original_image_size.width) / pImpl_->inputWidth;
    float y_factor = static_cast<float>(original_image_size.height) / pImpl_->inputHeight;
    
    // YOLOv4 anchor boxes for each detection scale (typically 3 scales)
    // These are the default anchors used in the original darknet CFG files
    const std::vector<std::vector<std::vector<int>>> anchors = {
        {{12, 16}, {19, 36}, {40, 28}},            // Small objects (yolo-layer0)
        {{36, 75}, {76, 55}, {72, 146}},           // Medium objects (yolo-layer1) 
        {{142, 110}, {192, 243}, {459, 401}}       // Large objects (yolo-layer2)
    };
    
    // Number of classes (COCO has 80 classes)
    const int num_classes = 80;
    const int num_attributes = 5 + num_classes;  // x, y, w, h, objectness + classes
    const int num_anchors = 3;
    
    // Process each output layer
    for (size_t i = 0; i < output_tensors.size(); i++) {
        if (!output_tensors[i].IsTensor()) {
            continue;
        }
        
        // Get output tensor data and shape
        auto tensor_info = output_tensors[i].GetTensorTypeAndShapeInfo();
        auto output_shape = tensor_info.GetShape();
        
        // YOLOv4 outputs are in the shape [batch, num_anchors * (num_attributes), grid_h, grid_w]
        if (output_shape.size() != 4) {
            std::cerr << "Unexpected output shape: " << output_shape.size() << " dimensions" << std::endl;
            continue;
        }
        
        // Get dimensions
        int batch_size = output_shape[0];
        int channels = output_shape[1];
        int grid_h = output_shape[2];
        int grid_w = output_shape[3];
        
        // Get pointer to output tensor
        const float* output_data = output_tensors[i].GetTensorData<float>();
        
        // Each anchor predicts bounding boxes
        for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++) {
            // Get anchor dimensions
            int anchor_w = anchors[i][anchor_idx][0];
            int anchor_h = anchors[i][anchor_idx][1];
            
            // Process grid cells
            for (int row = 0; row < grid_h; row++) {
                for (int col = 0; col < grid_w; col++) {
                    // Calculate the starting index for this grid cell's channel data
                    // For each anchor, we need to access its predictions in the channel dimension
                    // The channel dimension stores all anchors sequentially
                    int channel_offset = anchor_idx * num_attributes;
                    
                    // Calculate indices for this grid cell's bbox coordinates and objectness
                    int idx_x = GetIndex(batch_size, channels, grid_h, grid_w, 0, channel_offset + 0, row, col);
                    int idx_y = GetIndex(batch_size, channels, grid_h, grid_w, 0, channel_offset + 1, row, col);
                    int idx_w = GetIndex(batch_size, channels, grid_h, grid_w, 0, channel_offset + 2, row, col);
                    int idx_h = GetIndex(batch_size, channels, grid_h, grid_w, 0, channel_offset + 3, row, col);
                    int idx_obj = GetIndex(batch_size, channels, grid_h, grid_w, 0, channel_offset + 4, row, col);
                    
                    // Get raw values
                    float bbox_x = output_data[idx_x];
                    float bbox_y = output_data[idx_y];
                    float bbox_w = output_data[idx_w];
                    float bbox_h = output_data[idx_h];
                    float objectness = output_data[idx_obj];
                    
                    // Apply sigmoid to x, y, and objectness
                    bbox_x = Sigmoid(bbox_x);
                    bbox_y = Sigmoid(bbox_y);
                    objectness = Sigmoid(objectness);
                    
                    // Process only if objectness is above threshold
                    if (objectness >= pImpl_->confThreshold) {
                        // Find the class with highest confidence
                        int class_id = 0;
                        float max_class_score = 0;
                        
                        // Loop through all class scores
                        for (int cls = 0; cls < num_classes; cls++) {
                            int idx_cls = GetIndex(batch_size, channels, grid_h, grid_w, 0, channel_offset + 5 + cls, row, col);
                            float class_score = Sigmoid(output_data[idx_cls]);
                            
                            if (class_score > max_class_score) {
                                max_class_score = class_score;
                                class_id = cls;
                            }
                        }
                        
                        // Calculate final confidence as objectness * class_score
                        float confidence = objectness * max_class_score;
                        
                        // Filter weak predictions
                        if (confidence >= pImpl_->confThreshold) {
                            // Transform bbox coordinates
                            // x,y are offsets within grid cell (0-1)
                            bbox_x = (col + bbox_x) / grid_w;
                            bbox_y = (row + bbox_y) / grid_h;
                            
                            // w,h are exponential and relative to anchors
                            bbox_w = std::exp(bbox_w) * anchor_w / pImpl_->inputWidth;
                            bbox_h = std::exp(bbox_h) * anchor_h / pImpl_->inputHeight;
                            
                            // Convert to corner coordinates (left, top, right, bottom)
                            int left = static_cast<int>((bbox_x - bbox_w/2) * original_image_size.width);
                            int top = static_cast<int>((bbox_y - bbox_h/2) * original_image_size.height);
                            int width = static_cast<int>(bbox_w * original_image_size.width);
                            int height = static_cast<int>(bbox_h * original_image_size.height);
                            
                            // Ensure box is within image boundaries
                            left = std::max(0, left);
                            top = std::max(0, top);
                            width = std::min(width, original_image_size.width - left);
                            height = std::min(height, original_image_size.height - top);
                            
                            // Store the detection
                            boxes.push_back(cv::Rect(left, top, width, height));
                            confidences.push_back(confidence);
                            class_ids.push_back(class_id);
                        }
                    }
                }
            }
        }
    }
    
    // Apply Non-Maximum Suppression to remove overlapping boxes
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, pImpl_->confThreshold, pImpl_->nmsThreshold, indices);
    
    // Create detection objects from filtered boxes
    for (int idx : indices) {
        Detection det;
        det.bbox = boxes[idx];
        det.confidence = confidences[idx];
        det.classId = class_ids[idx];
        
        // Set class name
        if (class_ids[idx] < pImpl_->classNames.size()) {
            det.className = pImpl_->classNames[class_ids[idx]];
        } else {
            det.className = "unknown";
        }
        
        detections.push_back(det);
    }
    
    return detections;
}

// Helper function to calculate indices for NCHW tensor format
inline int YOLODetector::GetIndex(int batch, int channels, int height, int width, int b, int c, int h, int w) {
    return ((b * channels + c) * height + h) * width + w;
}

// Helper sigmoid function
inline float YOLODetector::Sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
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