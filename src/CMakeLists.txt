set(SOURCES
    YOLODetector.cpp
    FaceDetector.cpp
    ImageClassifier.cpp
    RESTServer.cpp
    main.cpp
)

add_executable(tAI_server ${SOURCES})

target_include_directories(tAI_server
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(tAI_server
    PRIVATE
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
    ${CURL_LIBRARIES}
    nlohmann_json::nlohmann_json
    rt
    pthread
) 