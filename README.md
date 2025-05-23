# tAI (Tiny AI)

A C++ service for efficient AI model serving, focusing on computer vision tasks. The service ensures single model instantiation for memory efficiency and provides a RESTful API interface for model inference.

## Features

- Single model instance management
- CUDA-accelerated inference
- RESTful API interface
- Support for multiple model types:
  - YOLOv3
  - YOLOv4
  - YOLOv3-tiny 
  - YOLOv4-tiny
  - MobileNet-SSD
  - YOLOX-Nano
  - NanoDet
  - Face Detection
  - Image Classification
  - (More to come)

## Prerequisites

- CMake (>= 3.10)
- OpenCV (with CUDA support)
- Boost
- nlohmann-json
- CUDA Toolkit

## Building

```bash
# Download required models
./scripts/download_models.sh

# Build the project
./scripts/build.sh
```

For debug build:
```bash
./scripts/build.sh --debug
```

## Running the Server

```bash
./build/src/tAI_server
```

The server will start on `http://0.0.0.0:8080` by default.

## API Endpoints

### Health Check

**Endpoint:** `GET /health`

**Response:**
```json
{
    "status": "ok",
    "service": "tAI"
}
```

### Module Health Check

**Endpoint:** `GET /module_health`

Returns detailed information about the status of all models that are registered in the system, including whether they are successfully loaded.

**Response:**
```json
{
    "status": "ok",
    "service": "tAI",
    "models": [
        {
            "id": "yolov3",
            "status": "loaded", 
            "type": "object_detection",
            "classes": ["person", "car", "dog", "..."]
        },
        {
            "id": "yolov4",
            "status": "not_loaded",
            "type": "object_detection"
        },
        {
            "id": "face_detection",
            "status": "loaded",
            "type": "face_detection"
        },
        {
            "id": "image_classification",
            "status": "loaded",
            "type": "image_classification"
        }
    ]
}
```

The response includes all registered models with their loading status (`loaded` or `not_loaded`), model type, and for loaded models, additional information like available class names.

### Object Detection

**Endpoint:** `POST /detect`

**Request Body:**
```json
{
    "model_id": "yolov3",  // or "yolov4", "yolov4-tiny", "yolov3-tiny", "mobilenet-ssd", "yolox-nano", "nanodet"
    "image": "<base64_encoded_image>"
}
```

**Response:**
```json
[
    {
        "class_name": "person",
        "confidence": 0.95,
        "bbox": {
            "x": 100,
            "y": 200,
            "width": 50,
            "height": 100
        }
    }
]
```

### Face Detection

**Endpoint:** `POST /detect_faces`

**Request Body:**
```json
{
    "model_id": "face_detection",
    "image": "<base64_encoded_image>"
}
```

**Response:**
```json
[
    {
        "confidence": 0.98,
        "bbox": {
            "x": 120,
            "y": 80,
            "width": 100,
            "height": 100
        },
        "landmarks": [  // Optional, if landmarks are detected
            {"x": 140, "y": 100},
            {"x": 180, "y": 100},
            {"x": 160, "y": 130},
            {"x": 145, "y": 150},
            {"x": 175, "y": 150}
        ]
    }
]
```

### Image Classification

**Endpoint:** `POST /classify`

**Request Body:**
```json
{
    "model_id": "image_classification",
    "image": "<base64_encoded_image>"
}
```

**Response:**
```json
[
    {
        "class_name": "cat",
        "confidence": 0.92
    },
    {
        "class_name": "tabby cat",
        "confidence": 0.85
    },
    {
        "class_name": "tiger cat",
        "confidence": 0.67
    }
]
```

## Testing

A Python test client is provided for easy testing:

```bash
python3 scripts/test_client.py <image_path> --model yolov3
```

Options:
- `--model`: Choose between "yolov3" or "yolov4" (default: yolov3)
- `--server`: Specify server URL (default: http://localhost:8080)

## Project Structure

```
tAI/
├── include/           # Header files
├── src/              # Source files
├── models/           # Model files
├── scripts/          # Utility scripts
├── test_images/      # Test images
└── tests/            # Unit tests
```

## License

MIT License

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request 

## Model Setup

### Face Detection Model

Download the face detection model files:

```bash
mkdir -p models/face_detection
cd models/face_detection
wget https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180220_uint8/opencv_face_detector.pbtxt -O deploy.prototxt
wget https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180220_uint8/opencv_face_detector_uint8.pb -O deploy.caffemodel
```

Alternatively, you can use the script:

```bash
./scripts/download_face_models.sh
``` 