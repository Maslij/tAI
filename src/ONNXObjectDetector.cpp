#include "ObjectDetector.hpp"
#include <iostream>

namespace tAI {

bool ONNXObjectDetector::loadModel(const std::string& modelPath) {
    // Call the base implementation from ONNXInferenceEngine
    return ONNXInferenceEngine::loadModel(modelPath);
}

} // namespace tAI 