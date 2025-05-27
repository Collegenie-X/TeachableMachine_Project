#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
게임 상태 처리 모듈

게임의 다양한 상태(인트로, 플레이, 대기, 가이드)를 처리하는 함수들을 제공합니다.
"""

import cv2
import time
import numpy as np
from utils.image_processing import (
    put_text_with_background, 
    center_text, 
    create_transparent_overlay,
    draw_progress_bar
)
from utils.model_utils import get_waste_info

def display_realtime_recognition(game_instance, frame):
    """
    항상 상단 오른쪽에 실시간 인식 정보를 표시합니다.
    
    Args:
        game_instance: 게임 인스턴스
        frame (numpy.ndarray): 현재 프레임
    """
    h, w, _ = frame.shape
    
    if game_instance.last_prediction:
        # 텍스트 생성
        recognition_text = f"Label: {game_instance.last_prediction} ({game_instance.current_confidence:.1f}%)"
        
        # 텍스트 크기 계산하여 오른쪽 정렬
        text_size = cv2.getTextSize(recognition_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = w - text_size[0] - 20  # 오른쪽 여백 20px
        
        # 배경과 함께 텍스트 표시
        put_text_with_background(frame, recognition_text, (text_x, 30), 
                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

def handle_intro_state(game_instance, frame):
    """
    인트로 화면 상태를 처리합니다.
    
    Args:
        game_instance: 게임 인스턴스
        frame (numpy.ndarray): 현재 프레임
        
    Returns:
        numpy.ndarray: 처리된 프레임
    """
    h, w, _ = frame.shape
    
    # 반투명 오버레이 생성
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # 상단 오른쪽에 인식된 라벨과 인식률 표시
    display_realtime_recognition(game_instance, frame)
    
    # 게임 제목
    title_text = "Waste Recognition System"
    title_x = center_text(title_text, w, font_scale=1.5)
    put_text_with_background(frame, title_text, (title_x, h // 4), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    
    # 현재 시간
    current_time = time.time()
    
    # 스페이스바가 눌린 경우 카운트다운 표시
    if game_instance.space_pressed and not game_instance.processing_result:
        # 남은 시간 계산
        remaining_time = game_instance.result_delay_time - (current_time - game_instance.space_pressed_time)
        if remaining_time > 0:
            countdown_text = f"Starting in: {remaining_time:.1f}s"
            countdown_x = center_text(countdown_text, w, font_scale=1.2)
            put_text_with_background(frame, countdown_text, (countdown_x, h // 2), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        return frame
    
    # 일반 인트로 화면 표시 (스페이스바가 눌리지 않은 경우)
    # 게임 설명
    instruction1 = "Show waste items to camera for recognition"
    instruction2 = "Learn about recycling value and proper disposal methods"
    instruction3 = "Press SPACE to start"
    
    instr1_x = center_text(instruction1, w)
    instr2_x = center_text(instruction2, w)
    instr3_x = center_text(instruction3, w, font_scale=1.2)
    
    put_text_with_background(frame, instruction1, (instr1_x, h // 2 - 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    put_text_with_background(frame, instruction2, (instr2_x, h // 2 - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # 키 설명
    key_guide1 = "Press 'G' to see recycling guide for identified item"
    key_guide2 = "Press 'R' to reset saved money counter"
    key_guide3 = "Press 'ESC' to exit"
    
    guide1_x = center_text(key_guide1, w)
    guide2_x = center_text(key_guide2, w)
    guide3_x = center_text(key_guide3, w)
    
    put_text_with_background(frame, key_guide1, (guide1_x, h // 2 + 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    put_text_with_background(frame, key_guide2, (guide2_x, h // 2 + 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    put_text_with_background(frame, key_guide3, (guide3_x, h // 2 + 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    # 재활용 금액 안내
    money_instruction = "Save money by recycling correctly!"
    money_x = center_text(money_instruction, w)
    put_text_with_background(frame, money_instruction, (money_x, h // 2 + 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # 깜박이는 시작 메시지
    elapsed_time = time.time() - game_instance.intro_start_time
    if int(elapsed_time * 2) % 2 == 0:  # 0.5초마다 깜박임
        put_text_with_background(frame, instruction3, (instr3_x, h - 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
    
    return frame

def handle_playing_state(game_instance, frame, class_idx):
    """
    게임 플레이 상태를 처리합니다.
    
    Args:
        game_instance: 게임 인스턴스
        frame (numpy.ndarray): 현재 프레임
        class_idx (int): 인식된 클래스 인덱스
        
    Returns:
        tuple: (처리된 프레임, 상태 변경 여부)
    """
    h, w, _ = frame.shape
    
    # 상단 오른쪽에 인식된 라벨과 인식률 표시
    display_realtime_recognition(game_instance, frame)
    
    # 현재 아이템 금액 표시
    current_value = 0
    if 0 <= class_idx < len(game_instance.recycling_values):
        current_value = game_instance.recycling_values[class_idx]
    
    # 양수/음수에 따라 다른 텍스트와 색상 표시
    money_color = (0, 255, 0) if current_value >= 0 else (0, 0, 255)
    money_text = ""
    if current_value > 0:
        money_text = f"Recycling value: Save {current_value} KRW"
    elif current_value < 0:
        money_text = f"Recycling cost: Pay {abs(current_value)} KRW"
    else:
        money_text = "Recycling value: 0 KRW"
        
    money_x = center_text(money_text, w)
    put_text_with_background(frame, money_text, (money_x, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, money_color, 2)
    
    # 처리 방법 표시 (있는 경우)
    if 0 <= class_idx < len(game_instance.disposal_methods) and game_instance.disposal_methods[class_idx]:
        disposal_text = f"Disposal: {game_instance.disposal_methods[class_idx]}"
        disposal_x = center_text(disposal_text, w)
        put_text_with_background(frame, disposal_text, (disposal_x, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # 왼쪽 상단에 누적 점수 표시 (크고 눈에 띄게)
    display_score(game_instance, frame)
    
    # 안정성 확인 - 현재 예측이 충분히 안정적인지 확인
    if game_instance.last_prediction == game_instance.last_stable_prediction:
        game_instance.stability_counter += 1
    else:
        game_instance.stability_counter = 0
        game_instance.last_stable_prediction = game_instance.last_prediction
    
    # 안정성 게이지 표시
    if game_instance.stability_counter > 0:
        stability_progress = min(1.0, game_instance.stability_counter / game_instance.required_stable_frames)
        draw_progress_bar(frame, (w - 170, 30), 150, 15, stability_progress, 
                        color_bg=(50, 50, 50), color_fg=(0, 255, 0))
        
        stability_text = f"Stability: {game_instance.stability_counter}/{game_instance.required_stable_frames}"
        put_text_with_background(frame, stability_text, (w - 170, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # 안정적인 예측이 있을 때 해당 쓰레기 인식 및 WAITING 상태로 전환
    if game_instance.stability_counter >= game_instance.required_stable_frames:
        # 현재 아이템 및 가치 저장
        game_instance.current_item = game_instance.last_stable_prediction
        game_instance.current_value = current_value
        
        # 누적 금액에 추가
        game_instance.total_saved_money += current_value
        
        # 상태 전환
        game_instance.game_state = "WAITING"
        game_instance.waiting_for_next = True
        
        # 인식 성공 메시지
        success_text = f"Recognized {game_instance.current_item}! "
        success_text += f"{'Saved' if current_value >= 0 else 'Cost'}: {abs(current_value)} KRW"
        print(success_text)
        
        # 저장된 예측 목록에 추가 (중복 인식 방지)
        game_instance.saved_predictions.append(game_instance.current_item)
        
        return frame, True
    
    # 가이드 키 안내 메시지
    guide_text = "Press 'G' for recycling guide"
    put_text_with_background(frame, guide_text, (20, h - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
    
    # 초기화 키 안내 메시지
    reset_text = "Press 'R' to reset counter"
    put_text_with_background(frame, reset_text, (w - 220, h - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
    
    return frame, False

def handle_waiting_state(game_instance, frame):
    """
    인식 후 대기 상태를 처리합니다.
    
    Args:
        game_instance: 게임 인스턴스
        frame (numpy.ndarray): 현재 프레임
        
    Returns:
        numpy.ndarray: 처리된 프레임
    """
    h, w, _ = frame.shape
    
    # 반투명 오버레이 생성
    overlay = frame.copy()
    bg_color = (0, 100, 0) if game_instance.current_value >= 0 else (100, 0, 0)
    cv2.rectangle(overlay, (0, 0), (w, h), bg_color, -1)
    alpha = 0.3
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # 왼쪽 상단에 누적 점수 표시
    display_score(game_instance, frame)
    
    # 상단 오른쪽에 인식된 라벨과 인식률 표시
    display_realtime_recognition(game_instance, frame)
    
    # 현재 시간
    current_time = time.time()
    
    if game_instance.current_item:
        recognition_text = f"Label: {game_instance.current_item} ({game_instance.current_confidence:.1f}%)"
        text_size = cv2.getTextSize(recognition_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = w - text_size[0] - 20  # 오른쪽 여백 20px
        put_text_with_background(frame, recognition_text, (text_x, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # 인식 결과 표시
    result_text = f"Recognized: {game_instance.current_item}"
    result_x = center_text(result_text, w, font_scale=1.2)
    put_text_with_background(frame, result_text, (result_x, h // 3), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    # 재활용 가치 표시
    value_color = (0, 255, 0) if game_instance.current_value >= 0 else (0, 0, 255)
    value_text = ""
    if game_instance.current_value > 0:
        value_text = f"Value: Saved {game_instance.current_value} KRW"
    elif game_instance.current_value < 0:
        value_text = f"Value: Cost {abs(game_instance.current_value)} KRW"
    else:
        value_text = "Value: 0 KRW"
    
    value_x = center_text(value_text, w, font_scale=1.0)
    put_text_with_background(frame, value_text, (value_x, h // 3 + 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, value_color, 2)
    
    # 처리 방법 표시
    if game_instance.current_item in game_instance.class_names:
        idx = game_instance.class_names.index(game_instance.current_item)
        if idx < len(game_instance.disposal_methods) and game_instance.disposal_methods[idx]:
            disposal_text = f"How to dispose: {game_instance.disposal_methods[idx]}"
            disposal_x = center_text(disposal_text, w)
            put_text_with_background(frame, disposal_text, (disposal_x, h // 3 + 120), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    # 타이머 표시 - 스페이스바가 눌린 경우
    if game_instance.space_pressed and not game_instance.processing_result:
        # 남은 시간 계산
        remaining_time = game_instance.result_delay_time - (current_time - game_instance.space_pressed_time)
        if remaining_time > 0:
            timer_text = f"Next item in: {remaining_time:.1f}s"
            timer_x = center_text(timer_text, w)
            put_text_with_background(frame, timer_text, (timer_x, h - 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    
    # 결과 표시 시간 - 결과 처리 중인 경우
    elif game_instance.processing_result:
        # 남은 표시 시간 계산
        remaining_display = game_instance.result_display_time - (current_time - game_instance.result_start_time)
        if remaining_display > 0:
            display_text = f"Results will remain for: {remaining_display:.1f}s"
            display_x = center_text(display_text, w)
            put_text_with_background(frame, display_text, (display_x, h - 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
    
    # 다음 단계 안내 - 처리 중이 아닌 경우에만 표시
    else:
        next_text = "Press SPACE to continue"
        next_x = center_text(next_text, w)
        
        # 깜박이는 효과
        if int(time.time() * 2) % 2 == 0:
            put_text_with_background(frame, next_text, (next_x, h - 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    
    # 가이드 키 안내 메시지
    guide_text = "Press 'G' for recycling guide"
    put_text_with_background(frame, guide_text, (20, h - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
    
    # 초기화 키 안내 메시지
    reset_text = "Press 'R' to reset counter"
    put_text_with_background(frame, reset_text, (w - 220, h - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
    
    return frame

def handle_guide_state(game_instance, frame):
    """
    가이드 화면 상태를 처리합니다.
    
    Args:
        game_instance: 게임 인스턴스
        frame (numpy.ndarray): 현재 프레임
        
    Returns:
        numpy.ndarray: 처리된 프레임
    """
    h, w, _ = frame.shape
    
    # 반투명 배경 생성
    create_transparent_overlay(frame, (0, 0, w, h), (0, 0, 0), 0.8)
    
    # 왼쪽 상단에 누적 점수 표시
    display_score(game_instance, frame)
    
    # 상단 오른쪽에 인식된 라벨과 인식률 표시
    display_realtime_recognition(game_instance, frame)
    
    # 가이드 제목
    guide_title = f"{game_instance.current_guide_item} Recycling Guide"
    title_x = center_text(guide_title, w, font_scale=1.2)
    cv2.putText(frame, guide_title, (title_x, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    # 사용자 정의 처리 방법 가져오기
    custom_method = ""
    if game_instance.current_guide_item in game_instance.class_names:
        idx = game_instance.class_names.index(game_instance.current_guide_item)
        if idx < len(game_instance.disposal_methods):
            custom_method = game_instance.disposal_methods[idx]
    
    # 분리수거 정보 가져오기 (사용자 정의 처리 방법 포함)
    waste_info = get_waste_info(game_instance.current_guide_item, game_instance.disposal_dict)
    
    # 현재 아이템 가치 표시
    current_item_idx = -1
    if game_instance.current_guide_item in game_instance.class_names:
        current_item_idx = game_instance.class_names.index(game_instance.current_guide_item)
    
    current_value = 0
    if current_item_idx >= 0 and current_item_idx < len(game_instance.recycling_values):
        current_value = game_instance.recycling_values[current_item_idx]
    
    # 양수/음수에 따라 다른 텍스트와 색상 표시
    value_color = (0, 255, 255) if current_value > 0 else (0, 0, 255) if current_value < 0 else (255, 255, 255)
    value_prefix = "Recycling value: " if current_value >= 0 else "Recycling cost: "
    value_text = f"{value_prefix}{abs(current_value)} KRW"
    
    value_x = center_text(value_text, w)
    cv2.putText(frame, value_text, (value_x, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, value_color, 2)
    
    # 사용자 정의 처리 방법 표시 (있는 경우)
    y_pos = 140  # 시작 위치
    if custom_method:
        custom_method_text = f"How to dispose: {custom_method}"
        custom_x = center_text(custom_method_text, w)
        cv2.putText(frame, custom_method_text, (custom_x, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        y_pos = 170  # 아래로 위치 조정
    
    # 정보 표시
    separation_method = f"Method: {waste_info['method']}"
    caution = f"Caution: {waste_info['caution']}"
    recyclable = "Recyclable: Yes" if waste_info['recyclable'] else "Recyclable: No"
    
    line_spacing = 40
    
    # 텍스트를 여러 줄로 분할하여 표시
    def wrap_text(text, max_width, font_scale):
        words = text.split()
        lines = []
        line = []
        
        for word in words:
            temp_line = ' '.join(line + [word])
            text_size = cv2.getTextSize(temp_line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            
            if text_size[0] <= max_width:
                line.append(word)
            else:
                lines.append(' '.join(line))
                line = [word]
        
        if line:
            lines.append(' '.join(line))
        
        return lines
    
    # 분리방법 여러 줄 표시
    method_lines = wrap_text(separation_method, w - 80, 0.7)
    for line in method_lines:
        cv2.putText(frame, line, (40, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += line_spacing
    
    y_pos += 20  # 추가 간격
    
    # 주의사항 여러 줄 표시
    caution_lines = wrap_text(caution, w - 80, 0.7)
    for line in caution_lines:
        cv2.putText(frame, line, (40, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += line_spacing
    
    y_pos += 20  # 추가 간격
    
    # 재활용 여부 표시
    recyclable_color = (0, 255, 0) if waste_info['recyclable'] else (0, 0, 255)
    cv2.putText(frame, recyclable, (40, y_pos), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, recyclable_color, 2)
    
    # 현재 시간
    current_time = time.time()
    
    # 타이머 표시 - 스페이스바가 눌린 경우
    if game_instance.space_pressed and not game_instance.processing_result:
        # 남은 시간 계산
        remaining_time = game_instance.result_delay_time - (current_time - game_instance.space_pressed_time)
        if remaining_time > 0:
            timer_text = f"Returning in: {remaining_time:.1f}s"
            timer_x = center_text(timer_text, w)
            put_text_with_background(frame, timer_text, (timer_x, h - 80), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    else:
        # 되돌아가기 안내
        back_text = "Press SPACE to return"
        back_x = center_text(back_text, w)
        cv2.putText(frame, back_text, (back_x, h - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    return frame

def display_score(game_instance, frame):
    """
    누적 점수를 화면에 표시합니다.
    
    Args:
        game_instance: 게임 인스턴스
        frame (numpy.ndarray): 현재 프레임
    """
    # 점수 색상 설정
    total_money_color = (0, 255, 0) if game_instance.total_saved_money >= 0 else (0, 0, 255)
    total_text = "SCORE: "
    if game_instance.total_saved_money < 0:
        total_text = "SCORE: -"
    
    # 검정색 배경으로 눈에 잘 띄게 표시
    score_bg = frame.copy()
    cv2.rectangle(score_bg, (10, 10), (220, 60), (0, 0, 0), -1)
    cv2.addWeighted(score_bg, 0.7, frame, 0.3, 0, frame)
    
    # 점수 텍스트 표시
    total_money_text = f"{total_text}{abs(game_instance.total_saved_money)} KRW"
    cv2.putText(frame, total_money_text, (20, 45), 
              cv2.FONT_HERSHEY_SIMPLEX, 1.0, total_money_color, 2)

def handle_game_end_state(game_instance, frame):
    """
    게임 종료 화면 상태를 처리합니다.
    
    Args:
        game_instance: 게임 인스턴스
        frame (numpy.ndarray): 현재 프레임
        
    Returns:
        numpy.ndarray: 처리된 프레임
    """
    h, w, _ = frame.shape
    
    # 반투명 오버레이 생성
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # 상단 오른쪽에 인식된 라벨과 인식률 표시
    display_realtime_recognition(game_instance, frame)
    
    # 게임 종료 메시지
    end_message = "Game Over! Thank you for playing!"
    end_x = center_text(end_message, w, font_scale=1.5)
    put_text_with_background(frame, end_message, (end_x, h // 4), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    
    # 총 점수 표시
    total_score = f"Total Score: {abs(game_instance.total_saved_money)} KRW"
    score_x = center_text(total_score, w, font_scale=1.2)
    put_text_with_background(frame, total_score, (score_x, h // 2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    
    # 스페이스바 안내 메시지
    space_text = "Press SPACE to exit"
    space_x = center_text(space_text, w, font_scale=1.2)
    put_text_with_background(frame, space_text, (space_x, h - 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
    
    return frame 