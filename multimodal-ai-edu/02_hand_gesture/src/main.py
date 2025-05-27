#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hand Gesture Recognition System - Main Application
This is the main entry point for the hand gesture recognition applications.
"""

import argparse
from utils.model_utils import patch_depthwise_conv, load_model_and_labels
from rock_paper_scissors import RockPaperScissorsGame
from sign_language import SignLanguageTranslator

def main():
    """
    Main function to parse arguments and run the application.
    
    Args:
        None
        
    Returns:
        None: Runs the selected application
    """
    # Apply TensorFlow compatibility patch
    patch_depthwise_conv()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Hand Gesture Recognition System')
    parser.add_argument('--mode', type=str, default='rps', choices=['rps', 'sign'],
                        help='Application mode: rps (Rock Paper Scissors) or sign (Sign Language)')
    parser.add_argument('--model', type=str, default='../models/converted_keras/keras_model.h5',
                        help='Model file path')
    parser.add_argument('--labels', type=str, default='../models/converted_keras/labels.txt',
                        help='Labels file path')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device number (default: 0)')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image size (default: 224)')
    args = parser.parse_args()
    
    # Load model and labels
    model, class_names = load_model_and_labels(args.model, args.labels)
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Start the selected application
    if args.mode == 'rps':
        print("가위바위보 게임을 시작합니다... (ESC 키로 종료)")
        game = RockPaperScissorsGame(model, class_names, args.camera, args.image_size)
        game.start()
    else:
        print("수어 번역기를 시작합니다... (ESC 키로 종료)")
        translator = SignLanguageTranslator(model, class_names, args.camera, args.image_size)
        translator.start()

if __name__ == "__main__":
    main() 