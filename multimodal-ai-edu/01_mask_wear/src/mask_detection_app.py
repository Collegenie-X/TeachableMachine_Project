#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mask Detection Application
This application detects mask wearing through real-time webcam feed.
Supports models (.h5) and labels (.txt) created from Teachable Machine.
"""

import cv2
import numpy as np
import tensorflow as tf
import time
import os
import argparse
import warnings

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

# Apply Teachable Machine model compatibility patch
_custom_layer_fix()

# Parse arguments
parser = argparse.ArgumentParser(description='Mask Detection Application')
parser.add_argument('--model', type=str, default='../models/converted_keras/keras_model.h5',
                    help='Model file path (default: ../models/converted_keras/keras_model.h5)')
parser.add_argument('--labels', type=str, default='../models/converted_keras/labels.txt',
                    help='Labels file path (default: ../models/converted_keras/labels.txt)')
parser.add_argument('--camera', type=int, default=0,
                    help='Camera device number (default: 0)')
parser.add_argument('--image_size', type=int, default=224,
                    help='Input image size (default: 224 - Teachable Machine default)')
parser.add_argument('--quiet', action='store_true',
                    help='Run with minimal output')
args = parser.parse_args()

model_path = args.model
labels_path = args.labels
verbose = not args.quiet

if verbose:
    print(f"Model path: {model_path}")
    print(f"Labels path: {labels_path}")

# Load Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load label file
try:
    with open(labels_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    if verbose:
        print(f"Labels loaded successfully: {class_names}")
except Exception as e:
    if verbose:
        print(f"Failed to load labels file: {e}")
        print("Using default labels: ['Mask', 'No Mask']")
    class_names = ['Mask', 'No Mask']

# Load model
try:
    # Register custom objects and load model in quiet mode
    with tf.keras.utils.custom_object_scope({}):
        # Load model with compile=False to avoid warning messages
        model = tf.keras.models.load_model(model_path, compile=False)
    if verbose:
        print(f"Model loaded successfully: {model_path}")
        # Show model summary (not in quiet mode)
        if not args.quiet:
            model.summary()
except Exception as e:
    print(f"Failed to load model: {e}")
    print("Trained model required. Download from Teachable Machine and save to models folder.")
    exit(1)

# Set up webcam
cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    print(f"Cannot open camera {args.camera}.")
    exit(1)

# Image size setting
IMG_SIZE = args.image_size

# Data structure for tracking repeat offenders
violation_counter = {}
VIOLATION_THRESHOLD = 5  # Number of violations to consider as repeat offender

# Store detected faces to reduce flickering
previous_faces = []
face_stability_frames = 10  # Number of frames to keep face detection stable

# Image preprocessing function
def preprocess_image(img):
    # Resize and normalize image
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Function to add text with background
def put_text_with_background(img, text, position, font, font_scale, text_color, thickness):
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

# Main loop
print("Mask detection system started. Press 'q' to quit.")
frame_count = 0
last_violation_check = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive frame from camera.")
        break
    
    # Resize display
    frame = cv2.resize(frame, (640, 480))
    
    # Face detection (process every 5 frames for better performance)
    frame_count += 1
    current_faces = []
    
    # Only run face detection every 5 frames
    if frame_count % 5 == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
        
        if len(faces) > 0:
            previous_faces = faces
            current_faces = faces
        else:
            # Keep using previous faces if no new faces detected
            if len(previous_faces) > 0:
                current_faces = previous_faces
    else:
        # Use the previous_faces for stability between detection frames
        if len(previous_faces) > 0:
            current_faces = previous_faces
    
    # Predict mask wearing for each detected face
    current_time = time.time()
    for i, (x, y, w, h) in enumerate(current_faces):
        # Extract face region
        face_id = f"face_{i}"
        face_img = frame[y:y+h, x:x+w]
        
        try:
            # Preprocess for prediction
            processed_face = preprocess_image(face_img)
            
            # Predict mask wearing (suppress warnings)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prediction = model.predict(processed_face, verbose=0)
            
            # Extract prediction result and confidence
            class_idx = np.argmax(prediction[0])
            confidence = float(prediction[0][class_idx])
            label = class_names[class_idx]
            
            # Check if the label contains "mask" or "Mask" (case insensitive)
            is_wearing_mask = "mask" in label.lower() and not ("no" in label.lower() or "without" in label.lower())
            
            # Display result
            if is_wearing_mask:  # Mask detected
                color = (0, 255, 0)  # Green
                result_text = f"{label}: {confidence*100:.2f}%"
                # Reset violation counter
                if face_id in violation_counter:
                    violation_counter[face_id] = 0
            else:  # No mask or other class
                color = (0, 0, 255)  # Red
                result_text = f"{label}: {confidence*100:.2f}%"
                
                # Increase violation counter
                if face_id in violation_counter:
                    violation_counter[face_id] += 1
                else:
                    violation_counter[face_id] = 1
                
                # Warning message with background
                put_text_with_background(
                    frame, "Please wear a mask!", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )
                
                # Repeat offender check
                if violation_counter[face_id] >= VIOLATION_THRESHOLD:
                    put_text_with_background(
                        frame, "Repeat offender!", (x, y-40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                    )
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Display prediction result text with background
            put_text_with_background(
                frame, result_text, (x, y+h+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
                
        except Exception as e:
            if verbose:
                print(f"Prediction error: {e}")
    
    # Remove old face IDs (every 30 seconds)
    if current_time - last_violation_check > 30:
        violation_counter = {}
        last_violation_check = current_time
    
    # Display system info with background
    put_text_with_background(
        frame, f"Model: {os.path.basename(model_path)}", (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
    )
    put_text_with_background(
        frame, f"Resolution: {IMG_SIZE}x{IMG_SIZE}", (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
    )
    
    # Instruction message with background
    put_text_with_background(
        frame, "Press 'q' to quit", (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
    )
    
    # Show result
    cv2.imshow('Mask Detection System', frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Mask detection system terminated") 