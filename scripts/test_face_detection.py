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

def detect_faces(image_path, model_id='face_detection', server_url='http://localhost:8080'):
    try:
        # Prepare the request
        image_base64 = encode_image(image_path)
        payload = {
            'model_id': model_id,
            'image': image_base64
        }

        # Send request to server
        response = requests.post(f'{server_url}/detect_faces', json=payload)
        response.raise_for_status()

        # Parse and display results
        detections = response.json()
        print(f"\nFace detections using {model_id}:")
        for i, face in enumerate(detections):
            print(f"Face #{i+1}, Confidence: {face['confidence']:.2f}")
            bbox = face['bbox']
            print(f"Bbox: x={bbox['x']}, y={bbox['y']}, w={bbox['width']}, h={bbox['height']}")
            
            # Print landmarks if available
            if 'landmarks' in face:
                print(f"Landmarks detected: {len(face['landmarks'])}")

        # Draw boxes on image
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        
        for face in detections:
            bbox = face['bbox']
            x, y = bbox['x'], bbox['y']
            w, h = bbox['width'], bbox['height']
            conf = face['confidence']
            
            # Draw rectangle - red for faces
            draw.rectangle([(x, y), (x + w, y + h)], outline='red', width=2)
            
            # Draw confidence label
            label = f"Face ({conf:.2f})"
            draw.text((x, y - 10), label, fill='red')
            
            # Draw landmarks if available
            if 'landmarks' in face:
                for landmark in face['landmarks']:
                    lx, ly = landmark['x'], landmark['y']
                    # Draw small circle for each landmark
                    draw.ellipse([(lx-2, ly-2), (lx+2, ly+2)], fill='blue')

        # Save annotated image
        output_path = image_path.rsplit('.', 1)[0] + '_faces_detected.' + image_path.rsplit('.', 1)[1]
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
    parser = argparse.ArgumentParser(description='Test the tAI face detection server')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--model', default='face_detection',
                      help='Model to use for detection (default: face_detection)')
    parser.add_argument('--server', default='http://localhost:8080',
                      help='Server URL (default: http://localhost:8080)')

    args = parser.parse_args()
    detect_faces(args.image_path, args.model, args.server)

if __name__ == '__main__':
    main() 