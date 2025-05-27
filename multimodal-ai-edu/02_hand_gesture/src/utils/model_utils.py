#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model Utilities Module
Contains utilities for loading and processing models in hand gesture recognition applications.
"""

import tensorflow as tf
import os
import warnings

# Filter warnings
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings (0=all, 1=INFO, 2=WARNING, 3=ERROR)

def patch_depthwise_conv():
    """
    Apply a compatibility patch for DepthwiseConv2D layer in TensorFlow.
    This resolves the 'groups' parameter issue with Teachable Machine models.
    
    Args:
        None
        
    Returns:
        bool: True if patch was successfully applied, False otherwise
    """
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
        return True
    except Exception as e:
        print(f"Failed to apply patch: {e}")
        return False

def clean_label(label):
    """
    Clean label text by removing number prefixes commonly found in Teachable Machine labels.
    
    Args:
        label (str): Original label from the model
        
    Returns:
        str: Cleaned label without prefix numbers
    """
    # Remove common prefixes like "0 ", "1 ", etc.
    if " " in label and label[0].isdigit():
        parts = label.split(" ", 1)
        if len(parts) > 1:
            return parts[1]
    return label

def load_model_and_labels(model_path, labels_path):
    """
    Load model and labels from specified paths.
    
    Args:
        model_path (str): Path to the Teachable Machine model file (.h5)
        labels_path (str): Path to the labels file (.txt)
        
    Returns:
        tuple: (model, class_names) - Loaded model and list of class names
              If model fails to load, returns (None, class_names)
    """
    # Load label file
    try:
        with open(labels_path, 'r') as f:
            class_names = [clean_label(line.strip()) for line in f.readlines()]
        print(f"Labels loaded successfully: {class_names}")
    except Exception as e:
        print(f"Failed to load labels file: {e}")
        print("Using default labels")
        if "rps" in model_path.lower():
            class_names = ['Rock', 'Paper', 'Scissors']
        else:
            class_names = ['Class1', 'Class2', 'Class3']

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

def get_prediction(model, preprocessed_image):
    """
    Get prediction from model for a preprocessed image.
    
    Args:
        model (tf.keras.Model): Loaded TensorFlow model
        preprocessed_image (numpy.ndarray): Preprocessed image tensor
        
    Returns:
        tuple: (predicted_class_index, confidence, prediction_array)
               predicted_class_index is the index of the class with highest confidence
               confidence is the confidence score for the predicted class
               prediction_array is the raw prediction array from the model
    """
    # Get prediction from model
    prediction = model.predict(preprocessed_image, verbose=0)
    
    # Get the predicted class
    class_idx = prediction[0].argmax()
    confidence = float(prediction[0][class_idx])
    
    return class_idx, confidence, prediction[0] 