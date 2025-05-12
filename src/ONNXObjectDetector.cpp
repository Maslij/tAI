#include "ObjectDetector.hpp"
#include <iostream>

namespace tAI {

#ifdef USE_ONNXRUNTIME
bool ONNXObjectDetector::loadModel(const std::string& modelPath) {
    // Call the base implementation from ONNXInferenceEngine
    return ONNXInferenceEngine::loadModel(modelPath);
}
#endif // USE_ONNXRUNTIME

} // namespace tAI 