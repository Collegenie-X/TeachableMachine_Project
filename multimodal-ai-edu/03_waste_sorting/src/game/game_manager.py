#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
게임 관리자 모듈

쓰레기 분류 게임의 핵심 로직과 상태 관리를 담당하는 클래스를 제공합니다.
"""

import cv2
import time
import numpy as np

from utils.image_processing import preprocess_image
from utils.model_utils import get_prediction
from game.state_handlers import (
    handle_intro_state,
    handle_playing_state,
    handle_waiting_state,
    handle_guide_state,
    handle_game_end_state
)

class WasteSortingGameManager:
    """
    쓰레기 분류 게임 관리자 클래스
    
    게임의 상태와 데이터를 관리하고 게임 루프를 실행합니다.
    """
    
    def __init__(self, model, class_names, recycling_values, disposal_methods, camera_id=0, image_size=224):
        """
        쓰레기 분류 게임 관리자 초기화
        
        Args:
            model: TensorFlow 모델
            class_names (list): 클래스 이름 목록
            recycling_values (list): 각 쓰레기 유형별 재활용 금액
            disposal_methods (list): 각 쓰레기 유형별 처리 방법
            camera_id (int): 카메라 장치 ID
            image_size (int): 모델 입력 이미지 크기
        """
        # 중요: total_targets를 먼저 정의 (다른 메소드에서 참조하기 때문)
        self.total_targets = 5  # 한 게임당 분류할 쓰레기 수
        
        self.model = model
        self.class_names = class_names
        self.recycling_values = recycling_values
        self.disposal_methods = disposal_methods
        self.disposal_dict = {name: method for name, method in zip(class_names, disposal_methods) if method}
        self.camera_id = camera_id
        self.image_size = image_size
        self.cap = None
        
        # 게임 상태 변수
        self.game_state = "INTRO"  # INTRO, PLAYING, WAITING, GUIDE, GAME_END
        self.total_saved_money = 0  # 총 절약된 금액
        
        # 안정성 추적 변수
        self.stability_counter = 0
        self.required_stable_frames = 10
        self.last_stable_prediction = None
        self.saved_predictions = []  # 저장된 예측 기록
        self.waiting_for_next = False  # 스페이스바 입력 대기 상태
        
        # 타이머 변수
        self.intro_start_time = time.time()
        self.last_prediction_time = 0
        self.prediction_cooldown = 3.0  # 같은 예측을 반복하지 않기 위한 쿨다운 시간(초)
        
        # 타이밍 관련 변수 추가
        self.result_delay_time = 3.0    # 스페이스바 입력 후 결과 표시까지의 지연 시간(초)
        self.result_display_time = 2.0  # 결과 표시 시간(초)
        self.space_pressed_time = 0      # 스페이스바가 눌린 시간
        self.result_start_time = 0       # 결과 표시 시작 시간
        
        # 안내 화면 변수
        self.current_guide_item = None
        self.guide_start_time = 0
        
        # 현재 인식된 아이템 정보
        self.current_item = None
        self.current_value = 0
        
        # 기타 변수
        self.last_prediction = None
        self.current_confidence = 0.0
        
        # 스페이스바 처리 상태 변수
        self.space_pressed = False      # 스페이스바가 눌렸는지 여부
        self.processing_result = False  # 결과 처리 중인지 여부
        
        # 게임 종료 변수
        self.game_completed = False     # 게임 종료 여부
        self.game_end_time = 0          # 게임 종료 시간
    
    def start(self):
        """
        게임을 시작합니다. 카메라를 초기화하고 게임 루프를 실행합니다.
        """
        # 카메라 초기화
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"Cannot open camera {self.camera_id}.")
            return
        
        print("Waste recognition started! Press ESC to exit.")
        
        try:
            self._game_loop()
        finally:
            # 리소스 해제
            self.cap.release()
            cv2.destroyAllWindows()
            print("Recognition ended.")
    
    def _game_loop(self):
        """
        게임의 메인 루프를 실행합니다.
        카메라 프레임을 처리하고 게임 상태에 따라 화면을 업데이트합니다.
        """
        while True:
            # 프레임 가져오기
            ret, frame = self.cap.read()
            if not ret:
                print("Cannot receive frame from camera.")
                break
            
            # 거울 모드로 프레임 반전
            frame = cv2.flip(frame, 1)
            
            # 현재 프레임 처리
            processed_frame = preprocess_image(frame, self.image_size)
            
            # 예측 수행
            class_idx = self._process_prediction(processed_frame)
            
            # 현재 시간 가져오기
            current_time = time.time()
            
            # 타이밍 업데이트 및 상태 전환 처리
            self._update_timing_and_state(current_time)
            
            # 게임 상태에 따라 프레임 처리
            frame = self._handle_current_state(frame, class_idx)
            
            # 화면에 프레임 표시
            cv2.imshow("Waste Recognition", frame)
            
            # 키 입력 처리
            if self._handle_key_input():
                break  # ESC 키가 눌리면 종료
    
    def _process_prediction(self, processed_frame):
        """
        이미지에 대한 예측을 처리합니다.
        
        Args:
            processed_frame (numpy.ndarray): 전처리된 프레임
            
        Returns:
            int: 인식된 클래스 인덱스
        """
        try:
            class_idx, confidence, prediction = get_prediction(self.model, processed_frame)
            
            # 인식 결과 업데이트 (안전하게 인덱스 확인)
            if 0 <= class_idx < len(self.class_names):
                self.last_prediction = self.class_names[class_idx]
            else:
                # 인덱스가 범위를 벗어나면 첫 번째 클래스 사용
                print(f"Warning: Class index {class_idx} out of range (max: {len(self.class_names)-1})")
                class_idx = 0
                self.last_prediction = self.class_names[0]
            
            self.current_confidence = confidence * 100
            return class_idx
            
        except Exception as e:
            print(f"Prediction error: {e}")
            # 오류 발생 시 기본값 설정
            self.last_prediction = self.class_names[0] if self.class_names else "Unknown"
            self.current_confidence = 50.0
            return 0
    
    def _update_timing_and_state(self, current_time):
        """
        시간 기반 상태 업데이트를 처리합니다.
        
        Args:
            current_time (float): 현재 시간
        """
        # 스페이스바가 눌렸고 지연 시간이 지났는지 확인
        if self.space_pressed and not self.processing_result:
            if current_time - self.space_pressed_time >= self.result_delay_time:
                # 결과 처리 시작
                self.processing_result = True
                self.result_start_time = current_time
                
                # 상태 변경 (INTRO → PLAYING, WAITING → PLAYING)
                if self.game_state == "INTRO":
                    self.game_state = "PLAYING"
                    self.waiting_for_next = False
                    print("Game started. Show a waste item to the camera.")
                elif self.game_state == "WAITING":
                    self.game_state = "PLAYING"
                    self.waiting_for_next = False
                    print("Ready for next item. Show a waste item to the camera.")
                elif self.game_state == "GUIDE":
                    if self.waiting_for_next:
                        self.game_state = "WAITING"
                    else:
                        self.game_state = "PLAYING"
        
        # 결과 처리 중이고 표시 시간이 지났는지 확인
        if self.processing_result:
            if current_time - self.result_start_time >= self.result_display_time:
                # 결과 처리 완료
                self.processing_result = False
                self.space_pressed = False
    
    def _handle_current_state(self, frame, class_idx):
        """
        현재 게임 상태에 따라 프레임을 처리합니다.
        
        Args:
            frame (numpy.ndarray): 현재 프레임
            class_idx (int): 인식된 클래스 인덱스
            
        Returns:
            numpy.ndarray: 처리된 프레임
        """
        if self.game_state == "INTRO":
            return handle_intro_state(self, frame)
            
        elif self.game_state == "PLAYING":
            processed_frame, state_changed = handle_playing_state(self, frame, class_idx)
            return processed_frame
            
        elif self.game_state == "WAITING":
            return handle_waiting_state(self, frame)
            
        elif self.game_state == "GUIDE":
            return handle_guide_state(self, frame)
            
        elif self.game_state == "GAME_END":
            return handle_game_end_state(self, frame)
            
        return frame
    
    def _handle_key_input(self):
        """
        키보드 입력을 처리합니다.
        
        Returns:
            bool: ESC 키가 눌렸으면 True, 아니면 False
        """
        key = cv2.waitKey(1) & 0xFF
        
        # ESC 키로 종료
        if key == 27:
            return True
        
        # 스페이스바 처리
        if key == 32 and not self.space_pressed:
            if self.game_state == "INTRO":
                self.space_pressed = True
                self.space_pressed_time = time.time()
                print(f"Processing will begin in {self.result_delay_time} seconds...")
            
            elif self.game_state == "WAITING":
                # 이미 수집한 예측이 목표 개수에 도달했는지 확인
                if len(self.saved_predictions) >= self.total_targets:
                    # 게임 종료 상태로 전환
                    self.game_state = "GAME_END"
                    self.game_completed = True
                    self.game_end_time = time.time()
                    print(f"Game completed! Total score: {self.total_saved_money} KRW")
                else:
                    # 다음 아이템으로 진행
                    self.space_pressed = True
                    self.space_pressed_time = time.time()
                    print(f"Processing will begin in {self.result_delay_time} seconds...")
            
            elif self.game_state == "GUIDE":
                self.space_pressed = True
                self.space_pressed_time = time.time()
                print(f"Processing will begin in {self.result_delay_time} seconds...")
            
            elif self.game_state == "GAME_END":
                # 게임 리셋 및 재시작
                self._reset_game()
                self.game_state = "INTRO"
                self.intro_start_time = time.time()
                print("Game reset. Press SPACE to start a new game.")
        
        # G 키를 눌러 가이드 보기
        if key == ord('g') or key == ord('G'):
            if (self.game_state == "PLAYING" or self.game_state == "WAITING") and self.current_item:
                self.game_state = "GUIDE"
                self.guide_start_time = time.time()
                self.current_guide_item = self.current_item
        
        # R 키를 눌러 저장된 금액 초기화
        if key == ord('r') or key == ord('R'):
            self.total_saved_money = 0
            print("Total saved money has been reset.")
            
        return False
        
    def _reset_game(self):
        """
        게임을 초기 상태로 리셋합니다.
        """
        # 점수 관련 변수 초기화
        self.total_saved_money = 0
        
        # 예측 관련 변수 초기화
        self.saved_predictions = []
        self.stability_counter = 0
        self.last_stable_prediction = None
        self.current_item = None
        self.current_value = 0
        
        # 상태 변수 초기화
        self.waiting_for_next = False
        self.space_pressed = False
        self.processing_result = False
        self.game_completed = False 