#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sign Language Translator Module
Implements a sign language translator using hand gesture recognition.
"""

import cv2
import numpy as np
import time
import random
from utils.image_processing import preprocess_image, put_text_with_background, center_text, create_transparent_overlay
from utils.model_utils import get_prediction

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
        self.challenge_timeout = 5  # 5 seconds
        self.challenge_active = False
        self.challenge_result = None
        self.challenge_success_count = 0
        self.challenge_attempt_count = 0
        self.current_score = 0  # Score tracking
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
        
        Args:
            None
            
        Returns:
            None
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
        """
        Generate a randomized list of words from class names for challenges.
        
        Args:
            None
            
        Returns:
            None: Updates the random_words list
        """
        # Create a copy of class names and shuffle it
        self.random_words = self.class_names.copy()
        random.shuffle(self.random_words)
        self.current_word_index = 0
    
    def _process_frame(self, frame):
        """
        Process each frame for sign language translation.
        
        Args:
            frame (numpy.ndarray): Current camera frame
            
        Returns:
            None: Updates the frame with translation information
        """
        # Get frame dimensions
        h, w, _ = frame.shape
        
        # Process the frame and get prediction
        processed_frame = preprocess_image(frame, self.image_size)
        class_idx, confidence, _ = get_prediction(self.model, processed_frame)
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
            text_x = center_text(instruction, w, font_scale=1.0)
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
        """
        Display the challenge result in large text at center of screen.
        
        Args:
            frame (numpy.ndarray): Current frame
            
        Returns:
            None: Updates the frame with result information
        """
        h, w, _ = frame.shape
        
        # Determine result text and color
        is_success = "SUCCESS" in self.challenge_result
        result_text = "CORRECT!" if is_success else "WRONG!"
        result_color = (0, 255, 0) if is_success else (0, 0, 255)
        
        # Create background
        bg_rect = (w // 4, h // 4, 3 * w // 4, 3 * h // 4)
        bg_color = (0, 100, 0) if is_success else (100, 0, 0)
        create_transparent_overlay(frame, bg_rect, bg_color, 0.7)
        
        # Display large result text
        text_size, _ = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 3.0, 5)
        text_x = w // 2 - text_size[0] // 2
        text_y = h // 2
        
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
        """
        Display random words in top center of frame.
        
        Args:
            frame (numpy.ndarray): Current frame
            width (int): Frame width
            
        Returns:
            None: Updates the frame with random words
        """
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
            
        Returns:
            None: Updates the frame with challenge prompt
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
        text_x = center_text(prompt_text, w, font_scale=1.2)
        
        # Create background
        bg_rect = (w // 4, h // 3, 3 * w // 4, 2 * h // 3)
        create_transparent_overlay(frame, bg_rect, (0, 0, 128), 0.7)
        
        put_text_with_background(frame, prompt_text, (text_x, h // 2), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
    def _start_challenge(self):
        """
        Start a new sign language challenge.
        
        Args:
            None
            
        Returns:
            None: Updates challenge state
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
            
        Returns:
            None: Updates challenge state based on player performance
        """
        h, w, _ = frame.shape
        current_time = time.time()
        elapsed_time = current_time - self.challenge_start_time
        remaining_time = max(0, self.challenge_timeout - elapsed_time)
        
        # Display challenge information
        challenge_text = f"Challenge: Show '{self.challenge_sign}' sign"
        
        # Center the challenge text
        text_x = center_text(challenge_text, w, font_scale=1.0)
        
        put_text_with_background(frame, challenge_text, (text_x, h - 80), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
        
        # Display remaining time with colored background based on urgency
        time_color = (0, 255, 0) if remaining_time > 3.0 else \
                    (0, 165, 255) if remaining_time > 1.5 else \
                    (0, 0, 255)
                    
        time_text = f"Time: {remaining_time:.1f}s"
        
        # Center the time text
        text_x = center_text(time_text, w, font_scale=0.9)
        
        put_text_with_background(frame, time_text, (text_x, h - 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, time_color, 2)
        
        # Calculate and display accuracy percentage
        if self.current_sign == self.challenge_sign and self.sign_stability > 0:
            # Find the prediction confidence for the current sign
            processed_frame = preprocess_image(frame, self.image_size)
            _, _, prediction = get_prediction(self.model, processed_frame)
            
            # Get confidence for the challenge sign class
            try:
                sign_idx = self.class_names.index(self.challenge_sign)
                confidence = float(prediction[sign_idx])
                
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