#!/usr/bin/env python3

import os
import sys
from test_face_detection import detect_faces

def main():
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    # Path to the people.jpg test image
    image_path = os.path.join(project_dir, 'test_images', 'people.jpg')
    
    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Error: Test image not found at {image_path}", file=sys.stderr)
        print("Please place 'people.jpg' in the tAI/test_images/ directory.")
        sys.exit(1)
    
    print(f"Testing face detection with image: {image_path}")
    
    # Run face detection and get the output image path
    output_path = detect_faces(image_path)
    
    # Attempt to display the image if running in an environment with display capability
    try:
        from PIL import Image
        
        print("Opening the annotated image...")
        image = Image.open(output_path)
        image.show()
    except Exception as e:
        print(f"Note: Could not display the image automatically. Please open {output_path} manually.")
    
    print(f"Face detection test completed. Annotated image saved to: {output_path}")

if __name__ == '__main__':
    main() 