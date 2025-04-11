#pragma once

#include <memory>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

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

class YOLODetector : public ObjectDetector {
public:
    YOLODetector();
    ~YOLODetector() override;

    std::vector<Detection> detect(const cv::Mat& image) override;
    bool loadModel(const std::string& modelPath) override;
    std::vector<std::string> getClassNames() const override;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

} // namespace tAI 