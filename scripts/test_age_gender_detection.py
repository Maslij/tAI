#!/usr/bin/env python3

import requests
import base64
from PIL import Image, ImageDraw, ImageFont
import json
import argparse
import sys
import io

def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def detect_age_gender(image_path, model_id='age_gender_detection', server_url='http://localhost:8080'):
    try:
        # Prepare the request
        image_base64 = encode_image(image_path)
        payload = {
            'model_id': model_id,
            'image': image_base64
        }

        # Send request to server
        response = requests.post(f'{server_url}/detect_age_gender', json=payload)
        response.raise_for_status()

        # Parse and display results
        detections = response.json()
        print(f"\nAge-Gender detections using {model_id}:")
        for i, detection in enumerate(detections):
            print(f"Face #{i+1}, Detection Confidence: {detection['detection_confidence']:.2f}")
            print(f"Gender: {detection['gender']}, Confidence: {detection['gender_confidence']:.2f}")
            print(f"Age: {detection['age']}, Confidence: {detection['age_confidence']:.2f}")
            bbox = detection['bbox']
            print(f"Bbox: x={bbox['x']}, y={bbox['y']}, w={bbox['width']}, h={bbox['height']}")

        # Draw boxes on image
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        
        for detection in detections:
            bbox = detection['bbox']
            x, y = bbox['x'], bbox['y']
            w, h = bbox['width'], bbox['height']
            gender = detection['gender']
            gender_conf = detection['gender_confidence']
            age = detection['age']
            
            # Draw rectangle - blue for male, pink for female
            color = 'blue' if gender == 'male' else 'magenta'
            draw.rectangle([(x, y), (x + w, y + h)], outline=color, width=2)
            
            # Draw gender and age label
            label = f"{gender} ({gender_conf:.2f}), {age} years"
            draw.text((x, y - 10), label, fill=color)

        # Save annotated image
        output_path = image_path.rsplit('.', 1)[0] + '_age_gender_detected.' + image_path.rsplit('.', 1)[1]
        img.save(output_path)
        print(f"\nAnnotated image saved to: {output_path}")
        
        return output_path

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with server: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Test the tAI age-gender detection server')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--model', default='age_gender_detection',
                      help='Model to use for detection (default: age_gender_detection)')
    parser.add_argument('--server', default='http://localhost:8080',
                      help='Server URL (default: http://localhost:8080)')

    args = parser.parse_args()
    detect_age_gender(args.image_path, args.model, args.server)

if __name__ == '__main__':
    main() 