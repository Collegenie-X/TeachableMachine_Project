#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Image processing utility module

This module provides image processing functions for the waste sorting game.
"""

import cv2
import numpy as np

def preprocess_image(frame, image_size):
    """
    Process an image for model prediction.
    
    Args:
        frame (numpy.ndarray): The input image frame
        image_size (int): Target image size
        
    Returns:
        numpy.ndarray: Processed image ready for model input
    """
    # Resize to the target size
    img = cv2.resize(frame, (image_size, image_size))
    
    # Convert to RGB (TensorFlow models typically expect RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values
    img = img / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def put_text_with_background(img, text, position, font, font_scale, color, thickness):
    """
    Puts text on image with a semi-transparent background for better readability.
    
    Args:
        img (numpy.ndarray): Input image
        text (str): Text to display
        position (tuple): Position (x, y) to place the text
        font: Font type
        font_scale (float): Font scale factor
        color (tuple): Text color (BGR)
        thickness (int): Line thickness
    """
    # Get text size
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    
    # Calculate background rectangle dimensions
    x, y = position
    padding = 5
    
    # Create semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (x - padding, y - text_h - padding), 
                 (x + text_w + padding, y + padding), (0, 0, 0), -1)
    
    # Apply transparency
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Put text
    cv2.putText(img, text, position, font, font_scale, color, thickness)

def center_text(text, frame_width, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1.0):
    """
    Calculate the x position to center text in the frame.
    
    Args:
        text (str): Text to center
        frame_width (int): Width of the frame
        font: Font type
        font_scale (float): Font scale factor
        
    Returns:
        int: x position for centered text
    """
    text_size = cv2.getTextSize(text, font, font_scale, 2)[0]
    return int((frame_width - text_size[0]) / 2)

def create_transparent_overlay(frame, rect, color, alpha):
    """
    Creates a semi-transparent overlay on part of the frame.
    
    Args:
        frame (numpy.ndarray): Input frame
        rect (tuple): Rectangle coordinates (x1, y1, x2, y2)
        color (tuple): Overlay color (BGR)
        alpha (float): Transparency factor (0-1)
    """
    x1, y1, x2, y2 = rect
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def draw_progress_bar(frame, position, width, height, progress, color_bg=(100, 100, 100), color_fg=(0, 255, 0)):
    """
    Draw a progress bar on the frame.
    
    Args:
        frame (numpy.ndarray): Input frame
        position (tuple): Top-left position (x, y)
        width (int): Bar width
        height (int): Bar height
        progress (float): Progress value (0-1)
        color_bg (tuple): Background color (BGR)
        color_fg (tuple): Foreground color (BGR)
    """
    x, y = position
    filled_width = int(width * min(1.0, max(0.0, progress)))
    
    # Draw background
    cv2.rectangle(frame, (x, y), (x + width, y + height), color_bg, -1)
    
    # Draw filled portion
    if filled_width > 0:
        cv2.rectangle(frame, (x, y), (x + filled_width, y + height), color_fg, -1)
    
    # Draw border
    cv2.rectangle(frame, (x, y), (x + width, y + height), (200, 200, 200), 1) 