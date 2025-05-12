#pragma once

#include <memory>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "ONNXInferenceEngine.hpp"

namespace tAI {

struct Detection {
    cv::Rect bbox;
    float confidence;
    int classId;
    std::string className;
};

class BaseModel {
public:
    virtual ~BaseModel() = default;
};

class ObjectDetector : public BaseModel {
public:
    virtual ~ObjectDetector() = default;
    
    virtual std::vector<Detection> detect(const cv::Mat& image) = 0;
    virtual bool loadModel(const std::string& modelPath) = 0;
    virtual std::vector<std::string> getClassNames() const = 0;
};

#ifdef USE_ONNXRUNTIME
// ONNX-based object detector
class ONNXObjectDetector : public ObjectDetector, public ONNXInferenceEngine {
public:
    ONNXObjectDetector() = default;
    ~ONNXObjectDetector() override = default;
    
    bool loadModel(const std::string& modelPath) override;
    
protected:
    // Process detections from ONNX output
    virtual std::vector<Detection> processDetections(
        const std::vector<Ort::Value>& output_tensors,
        const cv::Size& original_image_size) = 0;
};

class YOLODetector : public ONNXObjectDetector {
#else
// Fallback to OpenCV-based detector when ONNXRuntime is not available
class YOLODetector : public ObjectDetector {
#endif
public:
    YOLODetector();
    ~YOLODetector() override;

    std::vector<Detection> detect(const cv::Mat& image) override;
    bool loadModel(const std::string& modelPath) override;
    std::vector<std::string> getClassNames() const override;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
    
#ifdef USE_ONNXRUNTIME
    // Process YOLO detections from ONNX output
    std::vector<Detection> processDetections(
        const std::vector<Ort::Value>& output_tensors,
        const cv::Size& original_image_size) override;
        
    // Helper functions for YOLO detection processing
    inline int GetIndex(int batch, int channels, int height, int width, int b, int c, int h, int w);
    inline float Sigmoid(float x);
#endif
};

} // namespace tAI 