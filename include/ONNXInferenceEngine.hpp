#pragma once

#include <memory>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "/usr/local/include/onnxruntime/onnxruntime_cxx_api.h"

namespace tAI {

// Base class for all ONNX inference engines
class ONNXInferenceEngine {
public:
    ONNXInferenceEngine();
    virtual ~ONNXInferenceEngine();

    // Initialize and load an ONNX model
    virtual bool loadModel(const std::string& modelPath);
    
    // Get inference session options (CPU/GPU)
    Ort::SessionOptions createSessionOptions(bool enableCUDA = true);
    
    // Check if GPU is available
    bool isGPUAvailable() const;
    
protected:
    // ONNX Runtime environment
    std::shared_ptr<Ort::Env> env_;
    
    // ONNX Runtime session
    std::unique_ptr<Ort::Session> session_;
    
    // ONNX Runtime memory info
    Ort::MemoryInfo memory_info_{nullptr};
    
    // Model input/output names
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    
    // Model input/output shapes
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;
    
    // Flag indicating if model is using GPU
    bool using_cuda_ = false;
    
    // Extract input/output node information
    void getModelInfo();
    
    // Convert names to Ort format
    std::vector<const char*> getInputNames() const;
    std::vector<const char*> getOutputNames() const;
};

} // namespace tAI 