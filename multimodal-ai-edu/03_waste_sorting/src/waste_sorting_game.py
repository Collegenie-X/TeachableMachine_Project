#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
쓰레기 분류 게임 모듈 (직접 실행 버전)

이 파일은 labels.txt를 직접 사용하고 상단에 인식된 라벨과 인식률을 표시합니다.
"""

import sys
import os
import argparse

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 필요한 모듈 임포트
from utils.model_utils import load_model, load_labels
from game.game_manager import WasteSortingGameManager

def main_direct():
    """
    직접 실행 버전의 메인 함수 - labels.txt를 직접 사용하고 인식 정보를 표시합니다.
    """
    # 인자 파싱
    parser = argparse.ArgumentParser(description='Waste Sorting Game (Direct Version)')
    parser.add_argument('--model', type=str, default='../models/keras_model.h5',
                      help='Model file path')
    parser.add_argument('--labels', type=str, default='../models/labels.txt',
                      help='Labels file path')
    parser.add_argument('--camera', type=int, default=0,
                      help='Camera device ID')
    args = parser.parse_args()
    
    # labels.txt 파일 로드
    print(f"Loading labels from: {args.labels}")
    class_names, recycling_values, disposal_methods = load_labels(args.labels)
    
    # 모델 로드
    print(f"Loading model from: {args.model}")
    model = load_model(args.model)
    
    # 모델 로드 실패 시 처리
    if model is None:
        from utils.model_utils import DummyModel
        print("Model loading failed. Using dummy model that recognizes jongphil.")
        model = DummyModel()
        model.class_count = len(class_names)
        model.always_predict = 0  # jongphil 클래스를 항상 인식
    
    # 게임 초기화 및 시작
    try:
        print("Starting game with labels.txt applied directly.")
        game = WasteSortingGameManager(
            model,
            class_names,
            recycling_values,
            disposal_methods,
            args.camera,
            224  # 기본 이미지 크기
        )
        game.start()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

# 직접 실행 버전으로 시작
if __name__ == "__main__":
    main_direct() 