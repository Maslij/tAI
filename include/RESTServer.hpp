#pragma once

#include <string>
#include <memory>
#include <functional>
#include "ModelManager.hpp"

namespace tAI {

class RESTServer {
public:
    RESTServer(const std::string& host, int port);
    ~RESTServer();

    void start();
    void stop();

    // Register endpoints
    void registerDetectionEndpoint(const std::string& endpoint);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

} // namespace tAI 