#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include "ObjectDetector.hpp"  // For BaseModel

namespace tAI {

struct AgeGenderPrediction {
    int age;
    std::string gender;
    float ageConfidence;
    float genderConfidence;
    cv::Rect bbox;  // The face bounding box
};

class AgeGenderDetector : public BaseModel {
public:
    AgeGenderDetector();
    ~AgeGenderDetector();

    bool loadModel(const std::string& modelPath);
    std::vector<AgeGenderPrediction> predict(const cv::Mat& image, const std::vector<cv::Rect>& faces);

private:
    cv::dnn::Net ageNet_;
    cv::dnn::Net genderNet_;
    bool modelsLoaded_;

    // Age ranges from the model
    const std::vector<std::string> ageList = {"(0-2)", "(4-6)", "(8-12)", "(15-20)", 
                                              "(25-32)", "(38-43)", "(48-53)", "(60-100)"};
    // Gender classes
    const std::vector<std::string> genderList = {"Male", "Female"};

    // Pre-process input for the network
    cv::Mat preprocess(const cv::Mat& faceROI);
};

} // namespace tAI 