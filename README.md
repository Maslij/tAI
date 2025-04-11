# tAI (Tiny AI)

A C++ service for efficient AI model serving, focusing on computer vision tasks. The service ensures single model instantiation for memory efficiency and provides a RESTful API interface for model inference.

## Features

- Single model instance management
- CUDA-accelerated inference
- RESTful API interface
- Support for multiple model types:
  - YOLOv3
  - YOLOv4
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

### Object Detection

**Endpoint:** `POST /detect`

**Request Body:**
```json
{
    "model_id": "yolov3",  // or "yolov4"
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