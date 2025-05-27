#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Teachable Machine Model Test Script
This script loads a model generated from Teachable Machine and tests it with an image file.
"""

import cv2
import numpy as np
import tensorflow as tf
import argparse
import os
import warnings
import sys

# Filter warnings
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings (0=all, 1=INFO, 2=WARNING, 3=ERROR)

# Fix for TensorFlow model loading issues
# Custom object to fix the 'groups' parameter issue
def _custom_layer_fix():
    try:
        from tensorflow.keras.layers import DepthwiseConv2D
        # Save original from_config method
        original_from_config = DepthwiseConv2D.from_config
        
        # Define new from_config method
        @classmethod
        def patched_from_config(cls, config):
            # Remove groups parameter if present (without warning message)
            if 'groups' in config:
                config.pop('groups')
            return original_from_config(config)
        
        # Apply method patch
        DepthwiseConv2D.from_config = patched_from_config
        print("Model compatibility patch applied.")
    except Exception as e:
        print(f"Failed to apply patch: {e}")

# Parse arguments
parser = argparse.ArgumentParser(description='Teachable Machine Model Test')
parser.add_argument('--model', type=str, default='../models/converted_keras/keras_model.h5',
                    help='Model file path (default: ../models/converted_keras/keras_model.h5)')
parser.add_argument('--labels', type=str, default='../models/converted_keras/labels.txt',
                    help='Labels file path (default: ../models/converted_keras/labels.txt)')
parser.add_argument('--image', type=str, 
                    help='Path to image file for testing (if not specified, camera will be used)')
parser.add_argument('--image_size', type=int, default=224,
                    help='Input image size (default: 224 - Teachable Machine default)')
args = parser.parse_args()

# Apply Teachable Machine model compatibility patch
_custom_layer_fix()

def preprocess_image(img, target_size):
    """Image preprocessing function"""
    # Resize and normalize image
    img_resized = cv2.resize(img, (target_size, target_size))
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    return img_batch

def put_text_with_background(img, text, position, font, font_scale, text_color, thickness):
    """Function to add text with background"""
    # Get text size
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    
    # Create background coordinates
    x, y = position
    background_rect = (x, y - text_h - 5, x + text_w + 10, y + 5)
    
    # Draw background rectangle (semi-transparent)
    overlay = img.copy()
    cv2.rectangle(overlay, (background_rect[0], background_rect[1]), 
                 (background_rect[2], background_rect[3]), (0, 0, 0), -1)
    
    # Apply transparency
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Put text on the background
    cv2.putText(img, text, (x + 5, y - 5), font, font_scale, text_color, thickness)

def load_model_and_labels(model_path, labels_path):
    """Function to load model and labels"""
    # Load label file
    try:
        with open(labels_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        print(f"Labels loaded successfully: {class_names}")
    except Exception as e:
        print(f"Failed to load labels file: {e}")
        print("Using default labels: ['Class1', 'Class2']")
        class_names = ['Class1', 'Class2']

    # Load model
    try:
        # Register custom objects and load model in quiet mode
        with tf.keras.utils.custom_object_scope({}):
            model = tf.keras.models.load_model(model_path, compile=False)
        print(f"Model loaded successfully: {model_path}")
        return model, class_names
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None, class_names

def test_with_image(model, class_names, image_path, image_size):
    """Test model with image file"""
    # Load image
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Cannot load image: {image_path}")
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Failed to load image: {e}")
        return

    # Preprocess image and predict
    processed_img = preprocess_image(img, image_size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prediction = model.predict(processed_img, verbose=0)
    
    # Extract prediction result and confidence
    class_idx = np.argmax(prediction[0])
    confidence = float(prediction[0][class_idx])
    predicted_class = class_names[class_idx]
    
    # Check if the label contains "mask" or "Mask" (case insensitive)
    is_wearing_mask = "mask" in predicted_class.lower() and not ("no" in predicted_class.lower() or "without" in predicted_class.lower())
    
    # Print results
    print(f"\nPrediction result: {predicted_class} (Confidence: {confidence*100:.2f}%)")
    print("\nPrediction probabilities for all classes:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {prediction[0][i]*100:.2f}%")
    
    # Create result image
    result_img = img.copy()
    h, w, _ = result_img.shape
    
    # Set text color based on mask detection
    if is_wearing_mask:
        text_color = (0, 255, 0)  # Green for mask
        border_color = (0, 255, 0)
    else:
        text_color = (255, 0, 0)  # Red for no mask
        border_color = (255, 0, 0)
    
    # Add border to image
    border_thickness = 10
    result_img = cv2.copyMakeBorder(
        result_img, 
        border_thickness, border_thickness, border_thickness, border_thickness, 
        cv2.BORDER_CONSTANT, 
        value=border_color
    )
    
    # Display result on image
    font = cv2.FONT_HERSHEY_SIMPLEX
    result_text = f"{predicted_class}: {confidence*100:.2f}%"
    
    # Add mask status with background
    mask_status = "Mask Detected" if is_wearing_mask else "No Mask Detected"
    
    # Add text with background
    put_text_with_background(result_img, result_text, (30, 60), font, 1, (255, 255, 255), 2)
    put_text_with_background(result_img, mask_status, (30, 100), font, 1, (255, 255, 255), 2)
    
    # Add model info
    put_text_with_background(
        result_img, 
        f"Model: {os.path.basename(model_path)}", 
        (30, h - 30), 
        font, 0.6, (255, 255, 255), 1
    )
    
    # Show result image
    result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)  # Convert to BGR
    cv2.imshow('Prediction Result', result_img)
    print("\nPress any key to close the result window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    """Main function"""
    # Load model and labels
    model, class_names = load_model_and_labels(args.model, args.labels)
    if model is None:
        return
    
    # Test with image
    if args.image:
        test_with_image(model, class_names, args.image, args.image_size)
    else:
        print("Please specify an image path (--image option).")
        print("Example: python test_model.py --image test.jpg")

if __name__ == "__main__":
    main() 