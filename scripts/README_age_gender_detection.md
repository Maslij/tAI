# Age and Gender Detection Test Scripts

This directory contains scripts to test the age and gender detection functionality in the tAI service.

## Prerequisites

Make sure you have:
- The tAI server running
- The age-gender detection model properly installed (run `./scripts/download_gender_model.sh`)
- Python dependencies: `requests`, `Pillow` (PIL)

## Model Information

The age-gender detection feature uses the InsightFace genderage_v1 model to predict both gender and age from face images. The model:
- Requires a face detection step first (uses the face_detection model)
- Works best on aligned face images (112x112 pixels)
- Provides gender prediction (male/female) with confidence scores
- Provides age estimation in years with confidence scores

## Available Scripts

### download_gender_model.sh

Downloads the age-gender detection model from InsightFace repository.

```bash
# Usage
./download_gender_model.sh
```

### test_age_gender_detection.py

Test script for age and gender detection that can be used with any image containing faces.

```bash
# Usage
./test_age_gender_detection.py <path_to_image>

# Example
./test_age_gender_detection.py ../test_images/people.jpg
```

## API Usage

The age-gender detection functionality is exposed through the `/detect_age_gender` endpoint:

```bash
# Example POST request using curl
curl -X POST -H "Content-Type: application/json" \
  -d '{"model_id": "age_gender_detection", "image": "<base64_encoded_image>"}' \
  http://localhost:8080/detect_age_gender
```

The response is a JSON array where each element contains:
- Face bounding box information
- Face detection confidence
- Gender prediction ("male" or "female") 
- Gender prediction confidence
- Age prediction (in years)
- Age prediction confidence

Example response:
```json
[
  {
    "bbox": {
      "x": 100,
      "y": 50,
      "width": 120,
      "height": 150
    },
    "detection_confidence": 0.99,
    "gender": "male",
    "gender_confidence": 0.95,
    "age": 35,
    "age_confidence": 0.80
  }
]
```

## Backward Compatibility

For backward compatibility, the older `/detect_gender` endpoint is still available and will also return age information along with gender prediction.

## Output

The `test_age_gender_detection.py` script will:
1. Send the image to the age-gender detection endpoint
2. Print details about detected faces and their age-gender predictions
3. Create an annotated image with face bounding boxes colored by gender (blue for male, pink for female)
4. Display the predicted age alongside the gender
5. Save the annotated image with an `_age_gender_detected` suffix

## Performance Considerations

For edge devices:
- The age-gender detector is designed to be lightweight and efficient
- It works on aligned face regions to minimize processing overhead
- You can optimize further by controlling the input image size
- The model uses the same forward pass for both age and gender prediction, so there's no performance penalty for getting both attributes 