#!/usr/bin/env python3

import argparse
import base64
import json
import requests
import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont, ImageOps
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Test the image classification API')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--model', default='image_classification', help='Model ID to use (default: image_classification)')
    parser.add_argument('--server', default='http://localhost:8080', help='Server URL (default: http://localhost:8080)')
    args = parser.parse_args()

    # Load the image
    image = cv2.imread(args.image_path)
    if image is None:
        print(f"Error: Could not load image from {args.image_path}")
        return

    # Convert to base64
    _, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    # Prepare the request
    endpoint = f"{args.server}/classify"
    payload = {
        "model_id": args.model,
        "image": image_base64
    }

    # Make the request
    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        
        # Parse the results
        results = response.json()
        print("\nClassification Results:")
        print("----------------------")
        
        if not results:
            print("No classifications found")
            return
            
        # Display results
        for idx, result in enumerate(results):
            class_name = result['class_name']
            confidence = result['confidence']
            print(f"{idx+1}. {class_name} ({confidence:.2f})")
            
        # Load and display the image with results
        pil_image = Image.open(args.image_path)
        draw = ImageDraw.Draw(pil_image)
        
        # Try to load a font, or use default
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
            
        # Add classifications to the image
        y_pos = 10
        for idx, result in enumerate(results[:3]):  # Show top 3
            class_name = result['class_name']
            confidence = result['confidence']
            text = f"{class_name}: {confidence:.2f}"
            
            # Draw text background
            text_bbox = draw.textbbox((10, y_pos), text, font=font)
            draw.rectangle([text_bbox[0]-5, text_bbox[1]-5, text_bbox[2]+5, text_bbox[3]+5], 
                           fill=(0, 0, 0, 200))
            
            # Draw text
            draw.text((10, y_pos), text, font=font, fill=(255, 255, 255, 255))
            y_pos += 30
            
        # Show the result
        plt.figure(figsize=(12, 8))
        plt.imshow(np.array(pil_image))
        plt.axis('off')
        plt.show()
        
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")

if __name__ == "__main__":
    main() 