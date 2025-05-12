set(SOURCES
    YOLODetector.cpp
    FaceDetector.cpp
    ImageClassifier.cpp
    AgeGenderDetector.cpp
    RESTServer.cpp
    main.cpp
)

if(USE_ONNXRUNTIME)
    list(APPEND SOURCES 
        ONNXInferenceEngine.cpp
        ONNXObjectDetector.cpp
    )
endif()

add_executable(tAI_server ${SOURCES})

target_include_directories(tAI_server
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

# Libraries to link
set(LINK_LIBRARIES
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
    ${CURL_LIBRARIES}
    nlohmann_json::nlohmann_json
    rt
    pthread
)

if(USE_ONNXRUNTIME)
    list(APPEND LINK_LIBRARIES ${ONNXRUNTIME_LIBRARY})
endif()

target_link_libraries(tAI_server PRIVATE ${LINK_LIBRARIES}) 