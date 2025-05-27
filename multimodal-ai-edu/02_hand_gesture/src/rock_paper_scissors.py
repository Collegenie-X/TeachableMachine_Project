#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rock Paper Scissors Game Module
Implements a rock-paper-scissors game using hand gesture recognition.
"""

import cv2
import numpy as np
import time
import random
from utils.image_processing import preprocess_image, put_text_with_background, center_text, create_transparent_overlay
from utils.model_utils import get_prediction

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
        self.current_confidence = 0.0  # 현재 인식률 저장 변수 추가
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
            
        Returns:
            None: Processes the frame and updates game state
        """
        # Get frame dimensions
        h, w, _ = frame.shape
        
        # 항상 최신 예측 결과 얻기
        processed_frame = preprocess_image(frame, self.image_size)
        class_idx, confidence, prediction = get_prediction(self.model, processed_frame)
        predicted_class = self.class_names[class_idx]
        
        # 모든 게임 상태에서 사용할 인식 정보 업데이트
        if self.game_state != "RESULT":  # 결과 화면일 때는 마지막 안정적 인식 유지
            self.last_prediction = predicted_class
            self.current_confidence = float(prediction[class_idx]) * 100
        
        # 항상 오른쪽 상단에 인식 정보 표시
        self._display_recognition_info(frame, w)
        
        # Display game information
        self._display_game_info(frame, w, h)
        
        # Process based on game state
        if self.game_state == "WAITING":
            self._handle_waiting_state(frame)
        elif self.game_state == "START_MESSAGE":
            self._handle_start_message_state(frame)
        elif self.game_state == "PLAYING":
            self._handle_playing_state(frame, class_idx, confidence)
        elif self.game_state == "RESULT":
            if self.show_result:
                self._display_result(frame)
            else:
                # If not showing result animation, wait for space to restart
                restart_text = "Press SPACE to play again"
                text_x = center_text(restart_text, w)
                put_text_with_background(frame, restart_text, 
                                     (text_x, h // 2 + 120), 
                                     cv2.FONT_HERSHEY_SIMPLEX, 
                                     1, (0, 255, 255), 2)
    
    def _display_recognition_info(self, frame, width):
        """
        항상 오른쪽 상단에 인식된 라벨과 인식률을 표시합니다.
        
        Args:
            frame (numpy.ndarray): 현재 프레임
            width (int): 프레임 너비
            
        Returns:
            None: 프레임에 인식 정보를 추가합니다
        """
        if not hasattr(self, 'current_confidence'):
            self.current_confidence = 0.0
            
        if self.last_prediction:
            # 인식된 제스처가 유효한지 확인
            standard_gesture = self.gesture_mapping.get(self.last_prediction, self.last_prediction)
            is_valid_gesture = standard_gesture in ["Rock", "Paper", "Scissors"]
            RECT_WIDTH = 240
            
            # 검정색 배경 영역 생성
            overlay = frame.copy()
            cv2.rectangle(overlay, (width - RECT_WIDTH, 10), (width - 10, 130), (0, 0, 0), -1)
            alpha = 0.8
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # 라벨 이름 표시
            conf_text = f"{self.current_confidence:.1f}%"
            label_text = f"{self.last_prediction}"
            cv2.putText(frame, label_text+"(" + conf_text + ")", (width - RECT_WIDTH+10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 인식률 바 표시
            bar_width = 200
            filled_width = int(bar_width * self.current_confidence / 100)
            cv2.rectangle(frame, (width - RECT_WIDTH+10, 70), (width - RECT_WIDTH+10 + bar_width, 90), (70, 70, 70), -1)  # 어두운 회색 배경
            cv2.rectangle(frame, (width - RECT_WIDTH+10, 70), (width - RECT_WIDTH+10 + filled_width, 90), (255, 255, 255), -1)  # 흰색 채워진 부분
            
            
            
            # 유효성 상태 표시
            validity_text = "VALID" if is_valid_gesture else "INVALID"
            cv2.putText(frame, validity_text, (width - RECT_WIDTH+10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def _display_game_info(self, frame, width, height):
        """
        Display game information on the frame.
        
        Args:
            frame (numpy.ndarray): Current frame
            width (int): Frame width
            height (int): Frame height
            
        Returns:
            None: Updates the frame with game information
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
                        (255, 255, 255)  # Changed from red to white
            
            instruction = f"Show Rock/Paper/Scissors! Time: {remaining_time:.1f}s"
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
        
        # 인식 정보를 모든 상태에서 표시하기 위해 이 부분은 제거 (새로운 _display_recognition_info 메서드로 대체)
        
    def _handle_waiting_state(self, frame):
        """
        Handle the waiting state logic.
        
        Args:
            frame (numpy.ndarray): Current frame
            
        Returns:
            None: Updates game state based on waiting state logic
        """
        h, w, _ = frame.shape
        
        # 처리 및 예측은 _process_frame에서 이미 수행됨
        
        # Display instructions to start game
        start_text = "Press SPACE to start Rock Paper Scissors Game!"
        text_x = center_text(start_text, w)
        
        put_text_with_background(frame, start_text, 
                               (text_x, h // 2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 
                               1.0, (0, 255, 255), 2)
    
    def _handle_start_message_state(self, frame):
        """
        Handle the start message state before the game starts.
        
        Args:
            frame (numpy.ndarray): Current frame
            
        Returns:
            None: Updates game state based on start message state logic
        """
        h, w, _ = frame.shape
        
        # Display start message in center of screen
        start_message = "Rock Paper Scissors Game Start!"
        text_x = center_text(start_message, w, font_scale=1.2)
        
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
    
    def _handle_playing_state(self, frame, class_idx, confidence):
        """
        Handle the playing state logic with time limit.
        
        Args:
            frame (numpy.ndarray): Current frame
            class_idx (int): 인식된 클래스 인덱스
            confidence (float): 인식 확률
            
        Returns:
            None: Updates game state based on playing state logic
        """
        # 클래스 및 예측은 이미 _process_frame에서 수행됨
        predicted_class = self.last_prediction
        
        # Check mapped gesture
        mapped_gesture = self.gesture_mapping.get(predicted_class, predicted_class)
        
        # Check if it's one of Rock, Paper, Scissors
        is_valid_gesture = mapped_gesture in ["Rock", "Paper", "Scissors"]
        
        # Calculate stability - only increase for valid gestures
        if self.last_stable_gesture == predicted_class and is_valid_gesture:
            self.stability_counter += 1
        else:
            self.stability_counter = 0
            self.last_stable_gesture = predicted_class if is_valid_gesture else None
        
        # Display additional information - whether the gesture is valid
        h, w, _ = frame.shape
        if not is_valid_gesture and self.last_prediction:
            warning_text = f"'{self.last_prediction}' is not Rock/Paper/Scissors!"
            text_x = center_text(warning_text, w)
            put_text_with_background(frame, warning_text, 
                                   (text_x, h // 2 + 150), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.8, (255, 255, 255), 2)  # Changed from red to white
        
        # 안정성 카운터 표시
        if is_valid_gesture:
            stability_text = f"Stability: {self.stability_counter}/{self.required_stable_frames}"
            put_text_with_background(frame, stability_text, 
                                  (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.7, (0, 255, 0), 2)
        
        # Check if time is up
        current_time = time.time()
        elapsed_time = current_time - self.round_start_time
        
        # If stable gesture is detected for required frames, determine result
        if self.stability_counter >= self.required_stable_frames:
            # Double-check if it's a valid gesture
            mapped_gesture = self.gesture_mapping.get(self.last_stable_gesture, self.last_stable_gesture)
            if mapped_gesture in ["Rock", "Paper", "Scissors"]:
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
            
        Returns:
            None: Updates game result and scores
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
        
        # 안전 검사: 매핑된 제스처가 Rock, Paper, Scissors 중 하나인지 확인
        if player_choice not in ["Rock", "Paper", "Scissors"]:
            # 이 경우는 플레이어가 유효하지 않은 제스처를 취한 것으로 간주
            self.game_result = "INVALID"
            self.score_computer += 1
            return
        
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
            
        Returns:
            None: Updates frame with game result
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
            result_color = (255, 255, 255)  # Changed from red to white
            player_choice_text = "You didn't make a choice in time!"
        elif self.game_result == "INVALID":
            result_text = "INVALID GESTURE!"
            result_color = (255, 255, 255)  # Changed from red to white
            player_choice_text = "No valid Rock/Paper/Scissors detected!"
        else:
            player_raw_choice = self.last_stable_gesture
            player_choice = self.gesture_mapping.get(player_raw_choice, player_raw_choice)
            player_choice_text = f"You chose: {player_choice}"
            
            # Determine result text and color
            if self.game_result == "WIN":
                result_text = "YOU WIN!"
                result_color = (0, 255, 0)  # Keep green
            elif self.game_result == "LOSE":
                result_text = "YOU LOSE!"
                result_color = (255, 255, 255)  # Changed from red to white
            else:  # TIE
                result_text = "TIE GAME!"
                result_color = (255, 255, 255)  # Keep white
        
        # Create background for result
        bg_rect = (w // 4, h // 4, 3 * w // 4, 3 * h // 4)
        bg_color = (0, 100, 0) if self.game_result == "WIN" else \
                  (100, 0, 0) if self.game_result == "LOSE" or self.game_result == "TIMEOUT" else \
                  (70, 70, 70)
        create_transparent_overlay(frame, bg_rect, bg_color, 0.7)
        
        # Display result text in large font
        text_size, _ = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)
        text_x = w // 2 - text_size[0] // 2
        cv2.putText(frame, result_text, (text_x, h // 2 - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, result_color, 4)
        
        # Display player choice
        if player_choice_text:
            text_x = center_text(player_choice_text, w)
            cv2.putText(frame, player_choice_text, (text_x, h // 2 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Display computer choice
        comp_text = f"Computer chose: {self.computer_choice}"
        text_x = center_text(comp_text, w)
        cv2.putText(frame, comp_text, (text_x, h // 2 + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2) 