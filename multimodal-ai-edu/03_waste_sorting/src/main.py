#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
쓰레기 분류 게임 메인 모듈

실시간 카메라를 통해 쓰레기를 인식하고 올바른 분리수거 방법을 안내하는 게임형 애플리케이션을 실행합니다.
"""

import os
import argparse
import tensorflow as tf

from utils.model_utils import load_model, load_labels, DummyModel
from game.game_manager import WasteSortingGameManager

def check_tensorflow_version():
    """
    TensorFlow 버전을 확인하고 경고를 출력합니다.
    
    Returns:
        str: TensorFlow 버전 정보
    """
    try:
        version = tf.__version__
        print(f"TensorFlow version: {version}")
        
        # 버전 경고
        version_parts = version.split('.')
        if len(version_parts) >= 2:
            major, minor = int(version_parts[0]), int(version_parts[1])
            if major == 2 and minor >= 12:
                print("Warning: TensorFlow 2.12+ may have compatibility issues with Teachable Machine models.")
                print("TensorFlow 2.8.0 ~ 2.11.0 is recommended.")
        
        return version
    except Exception as e:
        print(f"Error checking TensorFlow version: {e}")
        return "Unknown"

def parse_arguments():
    """
    명령행 인자를 파싱합니다.
    
    Returns:
        argparse.Namespace: 파싱된 명령행 인자
    """
    parser = argparse.ArgumentParser(description='Waste Sorting Game')
    parser.add_argument('--model', type=str, default='../models/keras_model.h5',
                        help='Model file path')
    parser.add_argument('--labels', type=str, default='../models/labels.txt',
                        help='Labels file path')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Model input image size')
    parser.add_argument('--use_dummy', action='store_true',
                        help='Force using dummy model (for testing)')
    
    return parser.parse_args()

def load_game_model(model_path, use_dummy=False):
    """
    게임에 사용할 모델을 로드합니다.
    
    Args:
        model_path (str): 모델 파일 경로
        use_dummy (bool): 더미 모델 사용 여부
        
    Returns:
        tf.keras.Model or DummyModel: 로드된 모델
    """
    if use_dummy:
        print("Using dummy model mode. Game will run with random predictions.")
        dummy = DummyModel()
        dummy.always_predict = 0  # jongphil 클래스 인덱스를 기본으로 설정
        return dummy
    
    # 모델 존재 확인
    if not os.path.exists(model_path):
        print(f"Warning: Model file does not exist: {model_path}")
        print("Download a model from Teachable Machine and save to specified path.")
        print("Using dummy model instead to run the game.")
        dummy = DummyModel()
        dummy.always_predict = 0  # jongphil 클래스 인덱스를 기본으로 설정
        return dummy
    
    # 모델 로드 시도
    model = load_model(model_path)
    
    # 모델이 None인 경우 (더미 모델은 None이 아님)
    if model is None:
        print(f"Could not load model: {model_path}")
        print("Using dummy model for jongphil prediction instead.")
        dummy = DummyModel()
        dummy.always_predict = 0  # jongphil 클래스 인덱스를 기본으로 설정
        return dummy
    
    return model

def load_game_labels(labels_path):
    """
    게임에 사용할 레이블을 로드하고 검증합니다.
    
    Args:
        labels_path (str): 레이블 파일 경로
        
    Returns:
        tuple: (레이블 목록, 재활용 금액 목록, 처리방법 목록)
    """
    # 레이블 로드 시도 - 이제 레이블, 재활용 금액, 처리 방법을 함께 반환
    class_names, recycling_values, disposal_methods = load_labels(labels_path)
    
    # 레이블이 없는 경우 기본값 사용
    if not class_names:
        print(f"Could not load labels: {labels_path}")
        print("Using default class names and recycling values.")
        class_names = ["Plastic", "Paper", "Glass", "Can", "General_Waste", "Food_Waste"]
        recycling_values = [50, 20, 30, 100, 0, 0]  # 기본 재활용 금액
        disposal_methods = ["Rinse and recycle", "Bundle and recycle", "Remove caps and recycle", 
                          "Clean and compress", "Place in standard waste bag", "Remove water and dispose"]
    else:
        # 로드된 레이블 검증
        print("Validating labels.txt format...")
        for i, (name, value, method) in enumerate(zip(class_names, recycling_values, disposal_methods)):
            if value is None:
                print(f"Warning: Missing recycling value for class '{name}'. Using default value 0.")
                recycling_values[i] = 0
            
            # 처리 방법이 없는 경우 기본값 추가
            if not method:
                if "plastic" in name.lower():
                    disposal_methods[i] = "Rinse and remove labels before recycling"
                elif "paper" in name.lower():
                    disposal_methods[i] = "Bundle and recycle"
                elif "glass" in name.lower():
                    disposal_methods[i] = "Remove caps and recycle"
                elif "can" in name.lower():
                    disposal_methods[i] = "Clean and compress"
                elif "waste" in name.lower() or "trash" in name.lower():
                    disposal_methods[i] = "Place in standard waste bag"
                else:
                    disposal_methods[i] = "Check proper disposal method"
                
                print(f"Warning: Missing disposal method for class '{name}'. Added default method.")
    
    print(f"Classes to use: {class_names}")
    print(f"Recycling values: {recycling_values}")
    print(f"Disposal methods: {disposal_methods}")
    
    return class_names, recycling_values, disposal_methods

def main():
    """
    메인 함수 - 쓰레기 분류 게임을 초기화하고 시작합니다.
    """
    # TensorFlow 버전 확인
    check_tensorflow_version()
    
    # 명령행 인자 파싱
    args = parse_arguments()
    
    # 모델 로드
    model = load_game_model(args.model, args.use_dummy)
    
    # 레이블 로드
    class_names, recycling_values, disposal_methods = load_game_labels(args.labels)
    
    # 더미 모델의 클래스 수 설정
    if hasattr(model, 'class_count'):
        model.class_count = len(class_names)
        # 강제로 항상 jongphil 인식하도록 설정 (모델 문제 디버깅용)
        if isinstance(model, DummyModel):
            print("더미 모델 설정: 항상 jongphil 클래스 인식")
            model.always_predict = 0  # jongphil 클래스 인덱스
    
    # 게임 초기화 및 시작
    try:
        game = WasteSortingGameManager(
            model, 
            class_names, 
            recycling_values, 
            disposal_methods, 
            args.camera, 
            args.image_size
        )
        game.start()
    except Exception as e:
        print(f"An error occurred while running the game: {e}")
        # 디버깅을 위한 상세 오류 정보 출력
        import traceback
        traceback.print_exc()
        print("Exiting program.")

if __name__ == "__main__":
    main() 