#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <mutex>

namespace tAI {

class BaseModel;

class ModelManager {
public:
    static ModelManager& getInstance() {
        static ModelManager instance;
        return instance;
    }

    template<typename T>
    std::shared_ptr<T> getModel(const std::string& modelId) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = models_.find(modelId);
        if (it != models_.end()) {
            return std::static_pointer_cast<T>(it->second);
        }
        return nullptr;
    }

    template<typename T>
    bool registerModel(const std::string& modelId, std::shared_ptr<T> model) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (models_.find(modelId) != models_.end()) {
            return false;
        }
        
        models_[modelId] = model;
        return true;
    }

private:
    ModelManager() = default;
    ~ModelManager() = default;
    ModelManager(const ModelManager&) = delete;
    ModelManager& operator=(const ModelManager&) = delete;

    std::unordered_map<std::string, std::shared_ptr<BaseModel>> models_;
    std::mutex mutex_;
};

} // namespace tAI 