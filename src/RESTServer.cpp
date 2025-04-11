#include "RESTServer.hpp"
#include "ObjectDetector.hpp"
#include "FaceDetector.hpp"
#include "ImageClassifier.hpp"
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio.hpp>
#include <nlohmann/json.hpp>
#include <thread>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <string>

namespace beast = boost::beast;
namespace http = beast::http;
namespace net = boost::asio;
using tcp = boost::asio::ip::tcp;
using json = nlohmann::json;

namespace {
    // Base64 decoding table
    const unsigned char base64_table[256] = {
        64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
        64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
        64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 62, 64, 64, 64, 63,
        52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 64, 64, 64, 64, 64, 64,
        64,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
        15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 64, 64, 64, 64, 64,
        64, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 64, 64, 64, 64, 64,
        64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
        64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
        64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
        64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
        64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
        64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
        64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64
    };

    std::vector<unsigned char> base64_decode(const std::string& input) {
        if (input.empty())
            return std::vector<unsigned char>();

        // Remove any padding characters
        size_t padding = 0;
        if (input[input.size() - 1] == '=') padding++;
        if (input[input.size() - 2] == '=') padding++;

        // Calculate output size
        std::vector<unsigned char> decoded((input.size() * 3) / 4 - padding);
        size_t i = 0, j = 0;

        // Process groups of 4 characters
        while (i < input.size() - padding) {
            uint32_t triple = 0;
            for (int k = 0; k < 4; k++) {
                triple <<= 6;
                if (i < input.size() && input[i] != '=')
                    triple |= base64_table[static_cast<unsigned char>(input[i])];
                i++;
            }

            // Extract bytes from triple
            if (j < decoded.size()) decoded[j++] = (triple >> 16) & 0xFF;
            if (j < decoded.size()) decoded[j++] = (triple >> 8) & 0xFF;
            if (j < decoded.size()) decoded[j++] = triple & 0xFF;
        }

        return decoded;
    }
}

namespace tAI {

class RESTServer::Impl {
public:
    Impl(const std::string& host, int port)
        : host_(host), port_(port), ioc_(), acceptor_(ioc_) {}

    void start() {
        try {
            auto const address = net::ip::make_address(host_);
            tcp::endpoint endpoint{address, static_cast<unsigned short>(port_)};
            
            acceptor_.open(endpoint.protocol());
            acceptor_.set_option(net::socket_base::reuse_address(true));
            acceptor_.bind(endpoint);
            acceptor_.listen(net::socket_base::max_listen_connections);
            
            std::cout << "Starting server on http://" << host_ << ":" << port_ << std::endl;
            std::cout << "Available endpoints:" << std::endl;
            std::cout << "  GET/HEAD /health" << std::endl;
            std::cout << "    Returns: 200 OK if server is healthy" << std::endl;
            std::cout << "  POST /detect" << std::endl;
            std::cout << "    Request body: {" << std::endl;
            std::cout << "      \"model_id\": \"yolov3\" or \"yolov4\"," << std::endl;
            std::cout << "      \"image\": \"<base64_encoded_image>\"" << std::endl;
            std::cout << "    }" << std::endl;
            
            accept();
            ioc_.run();
        }
        catch(const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }

    void stop() {
        ioc_.stop();
    }

private:
    void accept() {
        acceptor_.async_accept(
            [this](beast::error_code ec, tcp::socket socket) {
                if(!ec) {
                    std::make_shared<Session>(std::move(socket))->start();
                }
                accept();
            });
    }

    class Session : public std::enable_shared_from_this<Session> {
    public:
        Session(tcp::socket socket) : socket_(std::move(socket)) {}

        void start() {
            read_request();
        }

    private:
        void read_request() {
            auto self = shared_from_this();
            
            http::async_read(
                socket_,
                buffer_,
                request_,
                [self](beast::error_code ec, std::size_t) {
                    if(!ec) {
                        self->process_request();
                    }
                });
        }

        void process_request() {
            response_.version(request_.version());
            response_.keep_alive(false);

            // Handle health endpoint for both HEAD and GET methods
            if(request_.target() == "/health") {
                response_.result(http::status::ok);
                response_.set(http::field::content_type, "application/json");
                if(request_.method() == http::verb::get) {
                    // For GET requests, include a JSON body
                    response_.body() = "{\"status\":\"ok\",\"service\":\"tAI\"}";
                }
                // For HEAD requests, the body is ignored but the 200 status is returned
            }
            else if(request_.method() == http::verb::post) {
                handle_post();
            }
            else {
                response_.result(http::status::bad_request);
                response_.set(http::field::content_type, "text/plain");
                response_.body() = "Invalid request method";
            }

            write_response();
        }

        void handle_post() {
            if(request_.target() == "/detect") {
                try {
                    // Parse the JSON request
                    auto req_body = json::parse(request_.body());
                    std::string model_id = req_body["model_id"];
                    std::string image_base64 = req_body["image"];

                    // Remove data URL prefix if present
                    size_t comma_pos = image_base64.find(',');
                    if (comma_pos != std::string::npos) {
                        image_base64 = image_base64.substr(comma_pos + 1);
                    }

                    // Decode base64 image
                    std::vector<unsigned char> image_data = base64_decode(image_base64);
                    if (image_data.empty()) {
                        throw std::runtime_error("Failed to decode base64 image");
                    }

                    cv::Mat image = cv::imdecode(image_data, cv::IMREAD_COLOR);
                    if (image.empty()) {
                        throw std::runtime_error("Failed to decode image data");
                    }

                    // Get the model from ModelManager
                    auto detector = ModelManager::getInstance().getModel<ObjectDetector>(model_id);
                    if(!detector) {
                        throw std::runtime_error("Model not found");
                    }

                    // Perform detection
                    auto detections = detector->detect(image);

                    // Convert detections to JSON
                    json response_json = json::array();
                    for(const auto& det : detections) {
                        json detection;
                        detection["class_name"] = det.className;
                        detection["confidence"] = det.confidence;
                        detection["bbox"] = {
                            {"x", det.bbox.x},
                            {"y", det.bbox.y},
                            {"width", det.bbox.width},
                            {"height", det.bbox.height}
                        };
                        response_json.push_back(detection);
                    }

                    response_.result(http::status::ok);
                    response_.set(http::field::content_type, "application/json");
                    response_.body() = response_json.dump();
                }
                catch(const std::exception& e) {
                    std::cerr << "Error processing request: " << e.what() << std::endl;
                    response_.result(http::status::internal_server_error);
                    response_.set(http::field::content_type, "text/plain");
                    response_.body() = std::string("Error: ") + e.what();
                }
            }
            else if(request_.target() == "/detect_faces") {
                try {
                    // Parse the JSON request
                    auto req_body = json::parse(request_.body());
                    std::string model_id = req_body["model_id"];
                    std::string image_base64 = req_body["image"];

                    // Remove data URL prefix if present
                    size_t comma_pos = image_base64.find(',');
                    if (comma_pos != std::string::npos) {
                        image_base64 = image_base64.substr(comma_pos + 1);
                    }

                    // Decode base64 image
                    std::vector<unsigned char> image_data = base64_decode(image_base64);
                    if (image_data.empty()) {
                        throw std::runtime_error("Failed to decode base64 image");
                    }

                    cv::Mat image = cv::imdecode(image_data, cv::IMREAD_COLOR);
                    if (image.empty()) {
                        throw std::runtime_error("Failed to decode image data");
                    }

                    // Get the model from ModelManager
                    auto detector = ModelManager::getInstance().getModel<FaceDetector>(model_id);
                    if(!detector) {
                        throw std::runtime_error("Face detector model not found");
                    }

                    // Perform face detection
                    auto detections = detector->detect(image);

                    // Convert detections to JSON
                    json response_json = json::array();
                    for(const auto& face : detections) {
                        json detection;
                        detection["confidence"] = face.confidence;
                        detection["bbox"] = {
                            {"x", face.bbox.x},
                            {"y", face.bbox.y},
                            {"width", face.bbox.width},
                            {"height", face.bbox.height}
                        };
                        
                        // Add landmarks if available
                        if (!face.landmarks.empty()) {
                            json landmarks_json = json::array();
                            for (const auto& point : face.landmarks) {
                                landmarks_json.push_back({
                                    {"x", point.x},
                                    {"y", point.y}
                                });
                            }
                            detection["landmarks"] = landmarks_json;
                        }
                        
                        response_json.push_back(detection);
                    }

                    response_.result(http::status::ok);
                    response_.set(http::field::content_type, "application/json");
                    response_.body() = response_json.dump();
                }
                catch(const std::exception& e) {
                    std::cerr << "Error processing face detection request: " << e.what() << std::endl;
                    response_.result(http::status::internal_server_error);
                    response_.set(http::field::content_type, "text/plain");
                    response_.body() = std::string("Error: ") + e.what();
                }
            }
            else if(request_.target() == "/classify") {
                try {
                    // Parse the JSON request
                    auto req_body = json::parse(request_.body());
                    std::string model_id = req_body["model_id"];
                    std::string image_base64 = req_body["image"];

                    // Remove data URL prefix if present
                    size_t comma_pos = image_base64.find(',');
                    if (comma_pos != std::string::npos) {
                        image_base64 = image_base64.substr(comma_pos + 1);
                    }

                    // Decode base64 image
                    std::vector<unsigned char> image_data = base64_decode(image_base64);
                    if (image_data.empty()) {
                        throw std::runtime_error("Failed to decode base64 image");
                    }

                    cv::Mat image = cv::imdecode(image_data, cv::IMREAD_COLOR);
                    if (image.empty()) {
                        throw std::runtime_error("Failed to decode image data");
                    }

                    // Get the model from ModelManager
                    auto classifier = ModelManager::getInstance().getModel<ImageClassifier>(model_id);
                    if(!classifier) {
                        throw std::runtime_error("Classification model not found");
                    }

                    // Perform classification
                    auto classifications = classifier->classify(image);

                    // Convert classifications to JSON
                    json response_json = json::array();
                    for(const auto& cls : classifications) {
                        json classification;
                        classification["class_name"] = cls.className;
                        classification["confidence"] = cls.confidence;
                        response_json.push_back(classification);
                    }

                    response_.result(http::status::ok);
                    response_.set(http::field::content_type, "application/json");
                    response_.body() = response_json.dump();
                }
                catch(const std::exception& e) {
                    std::cerr << "Error processing classification request: " << e.what() << std::endl;
                    response_.result(http::status::internal_server_error);
                    response_.set(http::field::content_type, "text/plain");
                    response_.body() = std::string("Error: ") + e.what();
                }
            }
            else {
                response_.result(http::status::not_found);
                response_.set(http::field::content_type, "text/plain");
                response_.body() = "Endpoint not found";
            }
        }

        void write_response() {
            auto self = shared_from_this();
            
            response_.set(http::field::content_length, std::to_string(response_.body().size()));
            
            http::async_write(
                socket_,
                response_,
                [self](beast::error_code ec, std::size_t) {
                    self->socket_.shutdown(tcp::socket::shutdown_send, ec);
                });
        }

        tcp::socket socket_;
        beast::flat_buffer buffer_;
        http::request<http::string_body> request_;
        http::response<http::string_body> response_;
    };

    std::string host_;
    int port_;
    net::io_context ioc_;
    tcp::acceptor acceptor_;
};

RESTServer::RESTServer(const std::string& host, int port)
    : pImpl_(std::make_unique<Impl>(host, port)) {}

RESTServer::~RESTServer() = default;

void RESTServer::start() {
    pImpl_->start();
}

void RESTServer::stop() {
    pImpl_->stop();
}

void RESTServer::registerDetectionEndpoint(const std::string& endpoint) {
    // This method can be used to register additional endpoints or
    // configure existing ones. For now, we have a fixed /detect endpoint.
}

} // namespace tAI 