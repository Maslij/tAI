#include "ONNXInferenceEngine.hpp"
#include <iostream>

namespace tAI {

#ifdef USE_ONNXRUNTIME
ONNXInferenceEngine::ONNXInferenceEngine() 
    : env_(std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ONNXInferenceEngine")),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) 
{
}

ONNXInferenceEngine::~ONNXInferenceEngine() = default;

bool ONNXInferenceEngine::loadModel(const std::string& modelPath) {
    try {
        // Try to use GPU if available, otherwise fallback to CPU
        auto sessionOptions = createSessionOptions(true);
        
        // Create session
        session_ = std::make_unique<Ort::Session>(*env_, modelPath.c_str(), sessionOptions);
        
        // Extract input/output node information
        getModelInfo();
        
        return true;
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error during model loading: " << e.what() << std::endl;
        return false;
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading ONNX model: " << e.what() << std::endl;
        return false;
    }
}

Ort::SessionOptions ONNXInferenceEngine::createSessionOptions(bool enableCUDA) {
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    if (enableCUDA && isGPUAvailable()) {
        // Use CUDA provider if available
        OrtCUDAProviderOptions cuda_options;
        sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
        using_cuda_ = true;
        std::cout << "Using CUDA provider for ONNX inference" << std::endl;
    } else {
        using_cuda_ = false;
        std::cout << "Using CPU provider for ONNX inference" << std::endl;
    }
    
    return sessionOptions;
}

bool ONNXInferenceEngine::isGPUAvailable() const {
    // Check if CUDA provider is available in this build
    try {
        auto providers = Ort::GetAvailableProviders();
        for (const auto& provider : providers) {
            if (provider == "CUDAExecutionProvider") {
                return true;
            }
        }
        return false;
    }
    catch (const std::exception& e) {
        std::cerr << "Error checking CUDA availability: " << e.what() << std::endl;
        return false;
    }
}

void ONNXInferenceEngine::getModelInfo() {
    if (!session_) {
        throw std::runtime_error("Session not initialized");
    }
    
    Ort::AllocatorWithDefaultOptions allocator;
    
    // Get input info
    size_t num_input_nodes = session_->GetInputCount();
    input_names_.resize(num_input_nodes);
    input_shapes_.resize(num_input_nodes);
    
    for (size_t i = 0; i < num_input_nodes; i++) {
        // Get input node name
        auto input_name = session_->GetInputNameAllocated(i, allocator);
        input_names_[i] = input_name.get();
        
        // Get input node shape
        auto type_info = session_->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        input_shapes_[i] = tensor_info.GetShape();
    }
    
    // Get output info
    size_t num_output_nodes = session_->GetOutputCount();
    output_names_.resize(num_output_nodes);
    output_shapes_.resize(num_output_nodes);
    
    for (size_t i = 0; i < num_output_nodes; i++) {
        // Get output node name
        auto output_name = session_->GetOutputNameAllocated(i, allocator);
        output_names_[i] = output_name.get();
        
        // Get output node shape
        auto type_info = session_->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        output_shapes_[i] = tensor_info.GetShape();
    }
}

std::vector<const char*> ONNXInferenceEngine::getInputNames() const {
    std::vector<const char*> result;
    result.reserve(input_names_.size());
    for (const auto& name : input_names_) {
        result.push_back(name.c_str());
    }
    return result;
}

std::vector<const char*> ONNXInferenceEngine::getOutputNames() const {
    std::vector<const char*> result;
    result.reserve(output_names_.size());
    for (const auto& name : output_names_) {
        result.push_back(name.c_str());
    }
    return result;
}
#endif // USE_ONNXRUNTIME

} // namespace tAI 