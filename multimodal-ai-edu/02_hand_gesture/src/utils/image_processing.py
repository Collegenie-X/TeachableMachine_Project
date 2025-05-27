#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Image Processing Utilities Module
Contains utilities for image processing in hand gesture recognition applications.
"""

import cv2
import numpy as np

def preprocess_image(img, target_size=224):
    """
    Preprocess an image for model prediction.
    
    Args:
        img (numpy.ndarray): Input image in BGR format from OpenCV
        target_size (int): Target size for model input (width and height)
        
    Returns:
        numpy.ndarray: Processed image as a normalized tensor with shape (1, target_size, target_size, 3)
    """
    # Resize and normalize image
    img = cv2.resize(img, (target_size, target_size))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def put_text_with_background(img, text, position, font, font_scale, text_color, thickness):
    """
    Add text with semi-transparent background to make it more readable.
    
    Args:
        img (numpy.ndarray): Image to add text to
        text (str): Text to display
        position (tuple): Position (x, y) for text
        font: Font type from cv2.FONT_*
        font_scale (float): Font scale
        text_color (tuple): RGB color for text as (R, G, B)
        thickness (int): Line thickness for text
        
    Returns:
        None: Modifies the input image directly
    """
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

def center_text(text, frame_width, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1.0, thickness=2):
    """
    Calculate the x-coordinate to center text on the frame.
    
    Args:
        text (str): Text to be centered
        frame_width (int): Width of the frame
        font: Font type from cv2.FONT_*
        font_scale (float): Font scale
        thickness (int): Line thickness for text
        
    Returns:
        int: x-coordinate to position the text centered on the frame
    """
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = frame_width // 2 - text_size[0] // 2
    return text_x

def create_transparent_overlay(frame, rect, color, alpha=0.7):
    """
    Create a semi-transparent overlay on a specified rectangle area of the frame.
    
    Args:
        frame (numpy.ndarray): Input frame
        rect (tuple): Rectangle coordinates as (x1, y1, x2, y2)
        color (tuple): BGR color of the overlay
        alpha (float): Transparency factor (0.0 to 1.0)
        
    Returns:
        None: Modifies the input frame directly
    """
    # Create a copy for the overlay
    overlay = frame.copy()
    
    # Draw filled rectangle
    cv2.rectangle(overlay, (rect[0], rect[1]), (rect[2], rect[3]), color, -1)
    
    # Apply transparency
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame) 