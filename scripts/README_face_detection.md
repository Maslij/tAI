# Face Detection Test Scripts

This directory contains scripts to test the face detection functionality in the tAI service.

## Prerequisites

Make sure you have:
- The tAI server running
- The face detection model properly installed (run `./scripts/download_face_models.sh`)
- Python dependencies: `requests`, `Pillow` (PIL)

## Available Scripts

### test_face_detection.py

Generic test script for face detection that can be used with any image.

```bash
# Usage
./test_face_detection.py <path_to_image>

# Example
./test_face_detection.py ../test_images/people.jpg
```

### test_people_faces.py

Specialized script that automatically uses the `people.jpg` image from the test_images directory.

```bash
# Usage (no arguments needed)
./test_people_faces.py
```

## Output

Both scripts will:
1. Send the image to the face detection endpoint
2. Print details about detected faces
3. Create an annotated image with face bounding boxes (and landmarks if available) 
4. Save the annotated image with a `_faces_detected` suffix
5. The `test_people_faces.py` script will also attempt to display the image

## Example Result

The scripts will print information like:

```
Face detections using face_detection:
Face #1, Confidence: 0.99
Bbox: x=100, y=50, width=120, height=150

Annotated image saved to: ../test_images/people_faces_detected.jpg
```

And will generate an image with red rectangles around detected faces. 