#pragma once

#include <memory>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "ObjectDetector.hpp"

namespace tAI {

struct FaceDetection {
    cv::Rect bbox;
    float confidence;
    std::vector<cv::Point> landmarks; // Optional face landmarks
};

class FaceDetector : public BaseModel {
public:
    FaceDetector();
    ~FaceDetector() override;

    std::vector<FaceDetection> detect(const cv::Mat& image);
    bool loadModel(const std::string& modelPath);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

} // namespace tAI 