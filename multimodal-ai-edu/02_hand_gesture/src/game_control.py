#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DEPRECATED - Use modular version instead
This monolithic version has been refactored into smaller, more maintainable modules.

Please use the following files instead:
- main.py: Main entry point for the application
- rock_paper_scissors.py: Rock-Paper-Scissors game implementation
- sign_language.py: Sign Language Translator implementation
- utils/model_utils.py: Model loading and prediction utilities
- utils/image_processing.py: Image processing utilities

For example, to run the application:
python main.py --mode rps  # for Rock-Paper-Scissors game
python main.py --mode sign  # for Sign Language Translator
"""

import cv2
import numpy as np
import tensorflow as tf
import time
import os
import argparse
import warnings
import random
from datetime import datetime

# Filter warnings
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings (0=all, 1=INFO, 2=WARNING, 3=ERROR)

# Fix for TensorFlow model loading issues
def patch_depthwise_conv():
    """
    Apply a compatibility patch for DepthwiseConv2D layer in TensorFlow.
    This resolves the 'groups' parameter issue with Teachable Machine models.
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
    except Exception as e:
        print(f"Failed to apply patch: {e}")

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

def preprocess_image(img, target_size=224):
    """
    Preprocess image for model prediction.
    
    Args:
        img (numpy.ndarray): Input image
        target_size (int): Target size for model input
        
    Returns:
        numpy.ndarray: Processed image ready for model prediction
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
        font: Font type
        font_scale (float): Font scale
        text_color (tuple): RGB color for text
        thickness (int): Line thickness for text
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

class RockPaperScissorsGame:
    """
    Rock Paper Scissors game using hand gesture recognition.
    """
    def __init__(self, model, class_names, camera_id=0, image_size=224):
        """
        Initialize the Rock Paper Scissors game.
        
        Args:
            model: TensorFlow model for gesture recognition
            class_names (list): List of class names (Rock, Paper, Scissors)
            camera_id (int): Camera device ID
            image_size (int): Input image size for the model
        """
        self.model = model
        self.class_names = class_names
        self.camera_id = camera_id
        self.image_size = image_size
        self.cap = None
        
        # Game state variables
        self.score_player = 0
        self.score_computer = 0
        self.last_prediction = None
        self.computer_choice = None
        self.game_result = None
        self.game_state = "WAITING"  # WAITING, START_MESSAGE, PLAYING, RESULT
        self.stability_counter = 0
        self.required_stable_frames = 8
        self.last_stable_gesture = None
        self.show_start_message = False
        self.start_message_time = 0
        self.round_start_time = 0
        self.round_timeout = 3.0  # 3 seconds to make a gesture
        self.show_result = False
        self.result_start_time = 0
        self.result_display_duration = 2.0  # Show result for 2 seconds
        
        # Map detected gestures to standard rock-paper-scissors terms
        self.gesture_mapping = self._create_gesture_mapping(class_names)
        
        # Game rules: key beats value
        self.rules = {
            "Rock": "Scissors",
            "Paper": "Rock",
            "Scissors": "Paper"
        }
    
    def _create_gesture_mapping(self, class_names):
        """
        Create a mapping between detected classes and standard game terms.
        This allows the game to work with any label naming convention.
        
        Args:
            class_names (list): List of class names from the model
            
        Returns:
            dict: Mapping from detected class names to standard game terms
        """
        mapping = {}
        
        # Try to automatically map class names to rock, paper, scissors
        for name in class_names:
            name_lower = name.lower()
            if 'rock' in name_lower or 'stone' in name_lower or 'fist' in name_lower:
                mapping[name] = "Rock"
            elif 'paper' in name_lower or 'flat' in name_lower or 'palm' in name_lower:
                mapping[name] = "Paper"
            elif 'scissors' in name_lower or 'scissor' in name_lower or 'victory' in name_lower or 'peace' in name_lower:
                mapping[name] = "Scissors"
            else:
                # For unknown gestures, just use as is - they won't match game rules
                mapping[name] = name
        
        # If we couldn't identify standard gestures, create a basic mapping
        if "Rock" not in mapping.values() and len(class_names) >= 3:
            print("Warning: Could not identify standard rock-paper-scissors gestures in labels.")
            print("Using first three labels as Rock, Paper, and Scissors.")
            for i, name in enumerate(class_names[:3]):
                if i == 0:
                    mapping[name] = "Rock"
                elif i == 1:
                    mapping[name] = "Paper"
                elif i == 2:
                    mapping[name] = "Scissors"
        
        print(f"Gesture mapping created: {mapping}")
        return mapping
    
    def start(self):
        """
        Start the camera and game loop.
        """
        # Initialize camera
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"Cannot open camera {self.camera_id}")
            return
        
        print("Rock Paper Scissors Game started! Press ESC to quit.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Cannot receive frame from camera.")
                break
            
            # Flip the frame horizontally for a more intuitive mirror view
            frame = cv2.flip(frame, 1)
            
            # Process the frame for the game
            self._process_frame(frame)
            
            # Display the frame
            cv2.imshow("Rock Paper Scissors Game", frame)
            
            # Check for global key presses
            key = cv2.waitKey(1) & 0xFF
            
            # Exit on ESC key (27)
            if key == 27:
                break
                
            # Restart on space key if in result state
            if key == 32 and (self.game_state == "RESULT" or self.game_state == "WAITING"):
                self.game_state = "START_MESSAGE"
                self.stability_counter = 0
                self.last_stable_gesture = None
                self.start_message_time = time.time()
        
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()
        print("Game terminated")
    
    def _process_frame(self, frame):
        """
        Process each frame for the game.
        
        Args:
            frame (numpy.ndarray): Current camera frame
        """
        # Get frame dimensions
        h, w, _ = frame.shape
        
        # Display game information
        self._display_game_info(frame, w, h)
        
        # Process based on game state
        if self.game_state == "WAITING":
            self._handle_waiting_state(frame)
        elif self.game_state == "START_MESSAGE":
            self._handle_start_message_state(frame)
        elif self.game_state == "PLAYING":
            self._handle_playing_state(frame)
        elif self.game_state == "RESULT":
            if self.show_result:
                self._display_result(frame)
            else:
                # If not showing result animation, wait for space to restart
                restart_text = "Press SPACE to play again"
                text_size, _ = cv2.getTextSize(restart_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                text_x = w // 2 - text_size[0] // 2
                put_text_with_background(frame, restart_text, 
                                     (text_x, h // 2 + 120), 
                                     cv2.FONT_HERSHEY_SIMPLEX, 
                                     1, (0, 255, 255), 2)
    
    def _display_game_info(self, frame, width, height):
        """
        Display game information on the frame.
        
        Args:
            frame (numpy.ndarray): Current frame
            width (int): Frame width
            height (int): Frame height
        """
        # Display score
        score_text = f"Player: {self.score_player} | Computer: {self.score_computer}"
        put_text_with_background(frame, score_text, (20, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display instructions based on game state
        if self.game_state == "WAITING":
            instruction = "Press SPACE to start the game"
        elif self.game_state == "START_MESSAGE":
            instruction = "Get ready!"
        elif self.game_state == "PLAYING":
            # Calculate remaining time
            current_time = time.time()
            elapsed_time = current_time - self.round_start_time
            remaining_time = max(0, self.round_timeout - elapsed_time)
            
            # Color based on urgency
            time_color = (0, 255, 0) if remaining_time > 2.0 else \
                        (0, 165, 255) if remaining_time > 1.0 else \
                        (0, 0, 255)
            
            instruction = f"Show your gesture! Time: {remaining_time:.1f}s"
            put_text_with_background(frame, instruction, (20, height - 20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, time_color, 2)
            return  # Skip the rest of instructions
        elif self.game_state == "RESULT":
            instruction = "Press SPACE to play again"
        
        put_text_with_background(frame, instruction, (20, height - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display exit instruction
        exit_text = "Press ESC to quit"
        put_text_with_background(frame, exit_text, (width - 150, height - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display current prediction if available with more detail
        if self.last_prediction and self.game_state != "RESULT":
            # Get the mapped standard gesture name
            standard_gesture = self.gesture_mapping.get(self.last_prediction, self.last_prediction)
            
            # Display raw detection result
            raw_text = f"Detected: {self.last_prediction}"
            put_text_with_background(frame, raw_text, 
                                  (width - 350, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.7, (255, 255, 255), 2)
            
            # Display mapped gesture
            mapped_text = f"Mapped to: {standard_gesture}"
            put_text_with_background(frame, mapped_text, 
                                  (width - 350, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.7, (255, 255, 255), 2)
            
            # Display stability counter when in playing state
            if self.game_state == "PLAYING":
                stability_text = f"Stability: {self.stability_counter}/{self.required_stable_frames}"
                put_text_with_background(frame, stability_text, 
                                      (width - 350, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.7, (255, 255, 255), 2)
    
    def _handle_waiting_state(self, frame):
        """
        Handle the waiting state logic.
        
        Args:
            frame (numpy.ndarray): Current frame
        """
        h, w, _ = frame.shape
        
        # Process the frame and get prediction for display only
        processed_frame = preprocess_image(frame, self.image_size)
        prediction = self.model.predict(processed_frame, verbose=0)
        
        # Get the predicted class
        class_idx = np.argmax(prediction[0])
        predicted_class = self.class_names[class_idx]
        
        # Update last prediction
        self.last_prediction = predicted_class
        
        # Display instructions to start game
        start_text = "Press SPACE to start Rock Paper Scissors Game!"
        text_size, _ = cv2.getTextSize(start_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        text_x = w // 2 - text_size[0] // 2
        
        put_text_with_background(frame, start_text, 
                               (text_x, h // 2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 
                               1.0, (0, 255, 255), 2)
    
    def _handle_start_message_state(self, frame):
        """
        Handle the start message state before the game starts.
        
        Args:
            frame (numpy.ndarray): Current frame
        """
        h, w, _ = frame.shape
        
        # Display start message in center of screen
        start_message = "Rock Paper Scissors Game Start!"
        text_size, _ = cv2.getTextSize(start_message, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
        text_x = w // 2 - text_size[0] // 2
        
        put_text_with_background(frame, start_message, 
                               (text_x, h // 2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 
                               1.2, (0, 255, 255), 2)
        
        # After 2 seconds, move to playing state
        current_time = time.time()
        if current_time - self.start_message_time >= 2:
            self.game_state = "PLAYING"
            self.round_start_time = time.time()
            self.stability_counter = 0
            self.last_stable_gesture = None
    
    def _handle_playing_state(self, frame):
        """
        Handle the playing state logic with time limit.
        
        Args:
            frame (numpy.ndarray): Current frame
        """
        # Process the frame and get prediction
        processed_frame = preprocess_image(frame, self.image_size)
        prediction = self.model.predict(processed_frame, verbose=0)
        
        # Get the predicted class
        class_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][class_idx])
        predicted_class = self.class_names[class_idx]
        
        # Update last prediction
        self.last_prediction = predicted_class
        
        # Check for stable gesture
        if self.last_stable_gesture == predicted_class:
            self.stability_counter += 1
        else:
            self.stability_counter = 0
            self.last_stable_gesture = predicted_class
        
        # Check if time is up
        current_time = time.time()
        elapsed_time = current_time - self.round_start_time
        
        # If stable gesture is detected for required frames, determine result
        if self.stability_counter >= self.required_stable_frames:
            self._determine_game_result(False)  # Player made a choice
            self.show_result = True
            self.result_start_time = time.time()
            self.game_state = "RESULT"
            return
            
        # If time is up and no stable gesture, player loses
        if elapsed_time >= self.round_timeout:
            self._determine_game_result(True)  # Player didn't make a choice
            self.show_result = True
            self.result_start_time = time.time()
            self.game_state = "RESULT"
    
    def _determine_game_result(self, timeout=False):
        """
        Determine the game result based on player and computer choices.
        
        Args:
            timeout (bool): Whether the player timed out without making a choice
        """
        # Generate computer's choice from standard terms
        standard_choices = ["Rock", "Paper", "Scissors"]
        self.computer_choice = random.choice(standard_choices)
        
        if timeout:
            # Player didn't make a choice in time
            self.game_result = "TIMEOUT"
            self.score_computer += 1
            return
            
        # Get player's final choice and map it to standard term
        player_raw_choice = self.last_stable_gesture
        player_choice = self.gesture_mapping.get(player_raw_choice, player_raw_choice)
        
        # Determine winner
        if player_choice == self.computer_choice:
            self.game_result = "TIE"
        elif player_choice in self.rules and self.rules[player_choice] == self.computer_choice:
            self.game_result = "WIN"
            self.score_player += 1
        else:
            self.game_result = "LOSE"
            self.score_computer += 1
    
    def _display_result(self, frame):
        """
        Display the game result with large text in center of screen.
        
        Args:
            frame (numpy.ndarray): Current frame
        """
        h, w, _ = frame.shape
        
        # Check if result animation should end
        current_time = time.time()
        if current_time - self.result_start_time >= self.result_display_duration:
            self.show_result = False
            return
            
        # Get the mapped player choice for display
        player_choice_text = ""
        
        if self.game_result == "TIMEOUT":
            result_text = "TIME OUT!"
            result_color = (0, 0, 255)  # Red
            player_choice_text = "You didn't make a choice in time!"
        else:
            player_raw_choice = self.last_stable_gesture
            player_choice = self.gesture_mapping.get(player_raw_choice, player_raw_choice)
            player_choice_text = f"You chose: {player_choice}"
            
            # Determine result text and color
            if self.game_result == "WIN":
                result_text = "YOU WIN!"
                result_color = (0, 255, 0)  # Green
            elif self.game_result == "LOSE":
                result_text = "YOU LOSE!"
                result_color = (0, 0, 255)  # Red
            else:  # TIE
                result_text = "TIE GAME!"
                result_color = (255, 255, 255)  # White
        
        # Create background for result
        overlay = frame.copy()
        bg_rect = (w // 4, h // 4, 3 * w // 4, 3 * h // 4)
        bg_color = (0, 100, 0) if self.game_result == "WIN" else (100, 0, 0) if self.game_result == "LOSE" or self.game_result == "TIMEOUT" else (70, 70, 70)
        cv2.rectangle(overlay, (bg_rect[0], bg_rect[1]), (bg_rect[2], bg_rect[3]), bg_color, -1)
        
        # Apply transparency
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Display result text in large font
        text_size, _ = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)
        text_x = w // 2 - text_size[0] // 2
        cv2.putText(frame, result_text, (text_x, h // 2 - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, result_color, 4)
        
        # Display player choice
        if player_choice_text:
            text_size, _ = cv2.getTextSize(player_choice_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            text_x = w // 2 - text_size[0] // 2
            cv2.putText(frame, player_choice_text, (text_x, h // 2 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Display computer choice
        comp_text = f"Computer chose: {self.computer_choice}"
        text_size, _ = cv2.getTextSize(comp_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        text_x = w // 2 - text_size[0] // 2
        cv2.putText(frame, comp_text, (text_x, h // 2 + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

class SignLanguageTranslator:
    """
    Sign Language Translator using hand gesture recognition.
    """
    def __init__(self, model, class_names, camera_id=0, image_size=224):
        """
        Initialize the Sign Language Translator.
        
        Args:
            model: TensorFlow model for sign language recognition
            class_names (list): List of sign language gestures
            camera_id (int): Camera device ID
            image_size (int): Input image size for the model
        """
        self.model = model
        self.class_names = class_names
        self.camera_id = camera_id
        self.image_size = image_size
        self.cap = None
        
        # Translator state variables
        self.current_sign = None
        self.sign_stability = 0
        self.required_stability = 8
        self.challenge_sign = None
        self.challenge_start_time = None
        self.challenge_timeout = 5  # Changed to 5 seconds
        self.challenge_active = False
        self.challenge_result = None
        self.challenge_success_count = 0
        self.challenge_attempt_count = 0
        self.current_score = 0  # New score tracking
        self.max_score = 100
        self.show_challenge_prompt = True
        self.challenge_prompt_time = 0
        self.random_words = []  # Store random words for display
        self.current_word_index = 0  # Index of current challenge word
        self.show_result = False  # Flag to show result
        self.result_start_time = 0  # Time when result started showing
        self.result_display_duration = 2  # Show result for 2 seconds
    
    def start(self):
        """
        Start the camera and translator loop.
        """
        # Initialize camera
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"Cannot open camera {self.camera_id}")
            return
        
        # Generate initial random words from class names
        self._generate_random_words()
        
        print("Sign Language Translator started! Press ESC to quit, 'c' to start a challenge.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Cannot receive frame from camera.")
                break
            
            # Flip the frame horizontally for a more intuitive mirror view
            frame = cv2.flip(frame, 1)
            
            # Process the frame for sign language translation
            self._process_frame(frame)
            
            # Display the frame
            cv2.imshow("Sign Language Translator", frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
            elif key == ord('c') and not self.challenge_active and not self.show_challenge_prompt and not self.show_result:
                self.show_challenge_prompt = True
                self.challenge_prompt_time = time.time()
            elif key == ord('c') and self.show_challenge_prompt:
                self.show_challenge_prompt = False
                self._start_challenge()
        
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()
        print("Translator terminated")
    
    def _generate_random_words(self):
        """Generate a randomized list of words from class names for challenges"""
        # Create a copy of class names and shuffle it
        self.random_words = self.class_names.copy()
        random.shuffle(self.random_words)
        self.current_word_index = 0
    
    def _process_frame(self, frame):
        """
        Process each frame for sign language translation.
        
        Args:
            frame (numpy.ndarray): Current camera frame
        """
        # Get frame dimensions
        h, w, _ = frame.shape
        
        # Process the frame and get prediction
        processed_frame = preprocess_image(frame, self.image_size)
        prediction = self.model.predict(processed_frame, verbose=0)
        
        # Get the predicted class
        class_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][class_idx])
        predicted_class = self.class_names[class_idx]
        
        # Update sign stability
        if self.current_sign == predicted_class:
            self.sign_stability += 1
        else:
            self.sign_stability = 0
            self.current_sign = predicted_class
        
        # Display current sign with confidence
        confidence_text = f"{predicted_class} ({confidence*100:.1f}%)"
        put_text_with_background(frame, confidence_text, (20, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display stable sign if stability threshold is reached
        if self.sign_stability >= self.required_stability:
            stable_text = f"Stable Sign: {self.current_sign}"
            put_text_with_background(frame, stable_text, (20, 70), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display random words in the top center
        self._display_random_words(frame, w)
        
        # Check if result should be displayed
        if self.show_result:
            current_time = time.time()
            if current_time - self.result_start_time >= self.result_display_duration:
                self.show_result = False
                # If we had a successful challenge, immediately start next one
                if "SUCCESS" in self.challenge_result:
                    self._start_challenge()
            else:
                self._display_result(frame)
                return  # Skip other displays while showing result
        
        # Handle challenge prompt if active
        if self.show_challenge_prompt:
            self._handle_challenge_prompt(frame)
        # Handle challenge mode if active
        elif self.challenge_active:
            self._handle_challenge(frame)
        else:
            # Display instructions for starting challenge (larger and centered)
            instruction = "Press 'c' to start a sign language challenge"
            text_size, _ = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            text_x = w // 2 - text_size[0] // 2
            put_text_with_background(frame, instruction, (text_x, h // 2), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            
            # Display exit instruction
            exit_text = "Press ESC to quit"
            put_text_with_background(frame, exit_text, (w - 150, h - 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display score and statistics
        score_text = f"Score: {self.current_score}/{self.max_score}"
        stats_text = f"Challenges: {self.challenge_success_count}/{self.challenge_attempt_count}"
        
        put_text_with_background(frame, score_text, (w - 250, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        put_text_with_background(frame, stats_text, (w - 250, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def _display_result(self, frame):
        """Display the challenge result in large text at center of screen"""
        h, w, _ = frame.shape
        
        # Determine result text and color
        is_success = "SUCCESS" in self.challenge_result
        result_text = "CORRECT!" if is_success else "WRONG!"
        result_color = (0, 255, 0) if is_success else (0, 0, 255)
        
        # Display large result text
        text_size, _ = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 3.0, 5)
        text_x = w // 2 - text_size[0] // 2
        text_y = h // 2
        
        # Create background
        overlay = frame.copy()
        bg_rect = (text_x - 40, text_y - text_size[1] - 20, 
                  text_x + text_size[0] + 40, text_y + 40)
        bg_color = (0, 100, 0) if is_success else (100, 0, 0)
        cv2.rectangle(overlay, (bg_rect[0], bg_rect[1]), 
                     (bg_rect[2], bg_rect[3]), bg_color, -1)
        
        # Apply transparency
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw text
        cv2.putText(frame, result_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 5)
        
        # Display additional information
        if is_success:
            # Extract points information
            try:
                points_text = self.challenge_result.split("!")[1].strip()  # Extract "+XX points" part
                # Calculate and display current score and progress
                score_info = f"{points_text} ({self.current_score}/{self.max_score})"
                
                # Center the score text
                score_size, _ = cv2.getTextSize(score_info, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
                score_x = w // 2 - score_size[0] // 2
                
                cv2.putText(frame, score_info, (score_x, text_y + 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            except:
                # Fallback if there's an error in text processing
                cv2.putText(frame, f"Score: {self.current_score}", (text_x, text_y + 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    def _display_random_words(self, frame, width):
        """Display random words in top center of frame"""
        if not self.random_words:
            return
            
        # Display 5 words (or fewer if not enough) centered at top
        num_words = min(5, len(self.random_words))
        total_width = 0
        word_heights = []
        
        # First calculate total width needed
        for i in range(num_words):
            idx = (self.current_word_index + i) % len(self.random_words)
            word = self.random_words[idx]
            text_size, _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            total_width += text_size[0] + 20  # Add spacing
            word_heights.append(text_size[1])
        
        # Calculate starting x position to center all words
        start_x = max(20, (width - total_width) // 2)
        max_height = max(word_heights) if word_heights else 20
        y_pos = max_height + 20
        
        current_x = start_x
        for i in range(num_words):
            idx = (self.current_word_index + i) % len(self.random_words)
            word = self.random_words[idx]
            
            # Highlight current challenge word
            text_color = (0, 255, 255) if self.challenge_active and word == self.challenge_sign else (255, 255, 255)
            bg_color = (0, 0, 128) if self.challenge_active and word == self.challenge_sign else (0, 0, 0)
            
            text_size, _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            # Create background with specified color
            overlay = frame.copy()
            bg_rect = (current_x - 5, y_pos - text_size[1] - 5, 
                      current_x + text_size[0] + 5, y_pos + 5)
            cv2.rectangle(overlay, (bg_rect[0], bg_rect[1]), 
                         (bg_rect[2], bg_rect[3]), bg_color, -1)
            
            # Apply transparency
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # Add text
            cv2.putText(frame, word, (current_x, y_pos - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            
            current_x += text_size[0] + 20  # Move to next position
    
    def _handle_challenge_prompt(self, frame):
        """
        Handle the challenge prompt before starting the challenge.
        
        Args:
            frame (numpy.ndarray): Current frame
        """
        h, w, _ = frame.shape
        
        # Calculate elapsed time
        current_time = time.time()
        elapsed_time = current_time - self.challenge_prompt_time
        
        # If shown for 2 seconds, start the challenge
        if elapsed_time >= 2:
            self.show_challenge_prompt = False
            self._start_challenge()
            return
            
        # Display large challenge prompt message
        prompt_text = "Get Ready for Sign Language Challenge!"
        text_size, _ = cv2.getTextSize(prompt_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
        text_x = w // 2 - text_size[0] // 2
        
        # Create background
        overlay = frame.copy()
        bg_rect = (text_x - 20, h // 2 - 50, text_x + text_size[0] + 20, h // 2 + 50)
        cv2.rectangle(overlay, (bg_rect[0], bg_rect[1]), (bg_rect[2], bg_rect[3]), (0, 0, 128), -1)
        
        # Apply transparency
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        put_text_with_background(frame, prompt_text, (text_x, h // 2), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
    def _start_challenge(self):
        """
        Start a new sign language challenge.
        """
        # Move to next word in the random list
        if self.current_word_index >= len(self.random_words) - 1:
            # If we've used all words, regenerate the list
            self._generate_random_words()
        else:
            self.current_word_index += 1
        
        # Select the challenge sign from the current position in random words
        self.challenge_sign = self.random_words[self.current_word_index]
        self.challenge_start_time = time.time()
        self.challenge_active = True
        self.challenge_result = None
        self.challenge_attempt_count += 1
    
    def _handle_challenge(self, frame):
        """
        Handle the sign language challenge logic.
        
        Args:
            frame (numpy.ndarray): Current frame
        """
        h, w, _ = frame.shape
        current_time = time.time()
        elapsed_time = current_time - self.challenge_start_time
        remaining_time = max(0, self.challenge_timeout - elapsed_time)
        
        # Display challenge information
        challenge_text = f"Challenge: Show '{self.challenge_sign}' sign"
        
        # Center the challenge text
        text_size, _ = cv2.getTextSize(challenge_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        text_x = w // 2 - text_size[0] // 2
        
        put_text_with_background(frame, challenge_text, (text_x, h - 80), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
        
        # Display remaining time with colored background based on urgency
        time_color = (0, 255, 0) if remaining_time > 3.0 else \
                    (0, 165, 255) if remaining_time > 1.5 else \
                    (0, 0, 255)
                    
        time_text = f"Time: {remaining_time:.1f}s"
        
        # Center the time text
        text_size, _ = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        text_x = w // 2 - text_size[0] // 2
        
        put_text_with_background(frame, time_text, (text_x, h - 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, time_color, 2)
        
        # Calculate and display accuracy percentage
        if self.current_sign == self.challenge_sign and self.sign_stability > 0:
            # Find the prediction confidence for the current sign
            processed_frame = preprocess_image(frame, self.image_size)
            prediction = self.model.predict(processed_frame, verbose=0)
            
            # Get confidence for the challenge sign class
            try:
                sign_idx = self.class_names.index(self.challenge_sign)
                confidence = float(prediction[0][sign_idx])
                
                # Calculate accuracy based on confidence and stability
                # This makes accuracy increase gradually with stability
                stability_factor = min(1.0, self.sign_stability / self.required_stability)
                accuracy = confidence * 100 * stability_factor
                
                accuracy_text = f"Accuracy: {accuracy:.1f}%"
                accuracy_color = (0, 255, 0) if accuracy > 80 else \
                                (0, 165, 255) if accuracy > 50 else \
                                (0, 0, 255)
                
                put_text_with_background(frame, accuracy_text, (w - 250, 90), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, accuracy_color, 2)
            except ValueError:
                # Handle case where challenge sign is not in class names
                pass
        
        # Check for challenge completion or timeout
        if self.sign_stability >= self.required_stability and self.current_sign == self.challenge_sign:
            # Calculate score based on remaining time (상대평가)
            # 5초 중 남은 시간의 비율에 따라 40점까지 획득
            time_ratio = remaining_time / self.challenge_timeout  # 0.0 ~ 1.0 사이의 값
            score_earned = int(40 * time_ratio)  # 최대 40점
            
            # 최소 점수 보장 (빨리 맞춰도 최소 15점)
            if score_earned < 15:
                score_earned = 15
                
            self.current_score = min(self.max_score, self.current_score + score_earned)
            
            self.challenge_result = f"SUCCESS! +{score_earned} points"
            self.challenge_success_count += 1
            self.challenge_active = False
            
            # Set up to show result
            self.show_result = True
            self.result_start_time = time.time()
        
        elif remaining_time <= 0:
            self.challenge_result = "FAILED! Try again"
            self.challenge_active = False
            
            # Set up to show result
            self.show_result = True
            self.result_start_time = time.time()

def main():
    """
    Main function to parse arguments and run the application.
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