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
    parser.add_argument('--model-type', default='googlenet', help='Model type to use (default: googlenet, options: googlenet, resnet50, mobilenet)')
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
        "model_type": args.model_type,
        "image": image_base64
    }

    # Send the request
    print(f"Sending request to {endpoint} with model_id={args.model} and model_type={args.model_type}")
    response = requests.post(endpoint, json=payload)
    print(f"Response status code: {response.status_code}")

    if response.status_code == 200:
        results = response.json()
        print("\nClassification Results:")
        for idx, result in enumerate(results):
            class_name = result["class_name"]
            confidence = result["confidence"]
            print(f"{idx+1}. {class_name}: {confidence:.4f}")
        
        # Display the image with results
        pil_image = Image.open(args.image_path)
        draw = ImageDraw.Draw(pil_image)
        
        # Try to find a font
        font = None
        try:
            # Try system fonts
            font_size = 20
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
            else:
                # Fall back to default
                font = ImageFont.load_default()
        except Exception as e:
            print(f"Font error: {e}")
            # Continue without font
        
        # Add title
        title = f"Classification Results (Model: {args.model_type})"
        text_color = (255, 255, 255)
        text_bg = (0, 0, 0, 180)
        
        # Draw background for title
        if len(results) > 0:
            # Display top 3 results on the image
            y_position = 10
            
            # Add title with background
            w, h = draw.textsize(title, font=font) if font else (200, 20)
            draw.rectangle([(10, y_position), (10 + w + 10, y_position + h + 10)], fill=text_bg)
            draw.text((15, y_position + 5), title, fill=text_color, font=font)
            y_position += h + 20
            
            # Add each result
            for idx, result in enumerate(results[:3]):  # Show top 3
                class_name = result["class_name"]
                confidence = result["confidence"]
                text = f"{idx+1}. {class_name}: {confidence:.4f}"
                
                w, h = draw.textsize(text, font=font) if font else (200, 20)
                draw.rectangle([(10, y_position), (10 + w + 10, y_position + h + 10)], fill=text_bg)
                draw.text((15, y_position + 5), text, fill=text_color, font=font)
                y_position += h + 15
                
                if idx >= 2:  # Only show top 3
                    break
        
        # Show the image
        plt.figure(figsize=(12, 8))
        plt.imshow(np.array(pil_image))
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    main() 