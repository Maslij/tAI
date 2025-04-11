#pragma once

#include <memory>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "ObjectDetector.hpp"  // For BaseModel

namespace tAI {

struct Classification {
    std::string className;
    float confidence;
};

class ImageClassifier : public BaseModel {
public:
    virtual ~ImageClassifier() = default;
    
    virtual std::vector<Classification> classify(const cv::Mat& image) = 0;
    virtual bool loadModel(const std::string& modelPath) = 0;
    virtual std::vector<std::string> getClassNames() const = 0;
};

class CVImageClassifier : public ImageClassifier {
public:
    CVImageClassifier();
    ~CVImageClassifier() override;

    std::vector<Classification> classify(const cv::Mat& image) override;
    bool loadModel(const std::string& modelPath) override;
    std::vector<std::string> getClassNames() const override;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

} // namespace tAI 