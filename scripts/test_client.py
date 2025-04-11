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

def detect_objects(image_path, model_id='yolov3', server_url='http://localhost:8080'):
    try:
        # Prepare the request
        image_base64 = encode_image(image_path)
        payload = {
            'model_id': model_id,
            'image': image_base64
        }

        # Send request to server
        response = requests.post(f'{server_url}/detect', json=payload)
        response.raise_for_status()

        # Parse and display results
        detections = response.json()
        print(f"\nDetections using {model_id}:")
        for det in detections:
            print(f"Class: {det['class_name']}, Confidence: {det['confidence']:.2f}")
            print(f"Bbox: x={det['bbox']['x']}, y={det['bbox']['y']}, w={det['bbox']['width']}, h={det['bbox']['height']}")

        # Draw boxes on image
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        
        for det in detections:
            bbox = det['bbox']
            x, y = bbox['x'], bbox['y']
            w, h = bbox['width'], bbox['height']
            conf = det['confidence']
            label = f"{det['class_name']} ({conf:.2f})"
            
            # Draw rectangle
            draw.rectangle([(x, y), (x + w, y + h)], outline='green', width=2)
            
            # Draw label
            draw.text((x, y - 10), label, fill='green')

        # Save annotated image
        output_path = image_path.rsplit('.', 1)[0] + '_detected.' + image_path.rsplit('.', 1)[1]
        img.save(output_path)
        print(f"\nAnnotated image saved to: {output_path}")

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with server: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Test the tAI object detection server')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--model', choices=['yolov3', 'yolov4'], default='yolov3',
                      help='Model to use for detection (default: yolov3)')
    parser.add_argument('--server', default='http://localhost:8080',
                      help='Server URL (default: http://localhost:8080)')

    args = parser.parse_args()
    detect_objects(args.image_path, args.model, args.server)

if __name__ == '__main__':
    main() 