#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
모델 관련 유틸리티 함수들을 제공하는 모듈입니다.
모델 로딩, 예측, 결과 해석 기능 등이 포함되어 있습니다.
"""

import tensorflow as tf
import numpy as np
import os

def load_model(model_path):
    """
    TensorFlow 모델을 로드합니다.
    Teachable Machine 모델과의 호환성 문제를 처리합니다.
    
    Args:
        model_path (str): 모델 파일 경로
        
    Returns:
        tf.keras.Model: 로드된 모델
    """
    try:
        # Teachable Machine 모델 호환성 문제 해결을 위한 사용자 정의 DepthwiseConv2D
        class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
            def __init__(self, *args, **kwargs):
                # 'groups' 인자가 있으면 제거 (TF 2.19에서 문제 발생)
                if 'groups' in kwargs:
                    del kwargs['groups']
                super().__init__(*args, **kwargs)
        
        # 사용자 정의 객체로 모델 로드
        custom_objects = {"DepthwiseConv2D": CustomDepthwiseConv2D}
        
        # 모델 로드 시도
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False  # 컴파일 단계 건너뛰기
        )
        
        print(f"모델을 성공적으로 로드했습니다: {model_path}")
        return model
        
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        
        # TFLite 모델로 변환 시도
        try:
            print("TFLite 모델로 로드 시도 중...")
            # TFLite 모델 경로 (h5 → tflite)
            tflite_path = model_path.replace('.h5', '.tflite')
            
            # tflite 파일이 없는 경우 변환 시도
            if not os.path.exists(tflite_path):
                print(f"TFLite 모델 파일이 없습니다: {tflite_path}")
                print("변환을 시도합니다...")
                
                # 변환 시도 (별도의 과정 필요)
                print("변환 실패. TFLite 모델을 사전에 생성해야 합니다.")
            
            # 기존 tflite 파일 로드 시도
            if os.path.exists(tflite_path):
                interpreter = tf.lite.Interpreter(model_path=tflite_path)
                interpreter.allocate_tensors()
                
                # 입출력 정보 가져오기
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                # TFLite 인터프리터를 래핑한 클래스 생성
                class TFLiteModelWrapper:
                    def __init__(self, interpreter, input_details, output_details):
                        self.interpreter = interpreter
                        self.input_details = input_details
                        self.output_details = output_details
                    
                    def predict(self, input_data):
                        # 입력 데이터 설정
                        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                        # 추론 실행
                        self.interpreter.invoke()
                        # 출력 데이터 가져오기
                        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                        return output_data
                
                tflite_model = TFLiteModelWrapper(interpreter, input_details, output_details)
                print("TFLite 모델로 성공적으로 로드했습니다.")
                return tflite_model
                
        except Exception as tflite_error:
            print(f"TFLite 로딩도 실패: {tflite_error}")
        
        # 모든 방법이 실패한 경우
        print("간단한 더미 모델을 생성합니다. 실제 예측은 작동하지 않을 수 있습니다.")
        return None

class DummyModel:
    """
    모델 로딩 실패 시 사용할 더미 모델 클래스.
    간단한 랜덤 예측을 제공합니다.
    """
    def __init__(self):
        self.class_count = 8  # 기본값을 8로 늘림 (labels.txt에 8개 항목이 있음)
        self.always_predict = None  # 항상 예측할 클래스 인덱스 (None이면 랜덤)
        
    def predict(self, input_data):
        """
        랜덤 예측을 반환합니다.
        
        Args:
            input_data: 입력 데이터 (무시됨)
            
        Returns:
            numpy.ndarray: 랜덤 예측 확률
        """
        if self.always_predict is not None:
            # 항상 특정 클래스 예측
            prediction = np.zeros(self.class_count)
            prediction[self.always_predict] = 1.0
        else:
            # 더미 예측 생성 (랜덤 값)
            prediction = np.random.random(self.class_count)
            # 합이 1이 되도록 정규화
            prediction = prediction / np.sum(prediction)
        
        # 입력 데이터 형태에 맞춰 출력 형태 결정
        if isinstance(input_data, np.ndarray) and input_data.ndim > 1:
            batch_size = input_data.shape[0]
            if batch_size > 1:
                # 배치 크기에 맞게 예측 확장
                return np.array([prediction] * batch_size)
        
        # 기본 단일 예측 (배치 차원 추가)
        return np.array([prediction])

def load_labels(labels_path):
    """
    라벨 파일을 로드합니다.
    '0 label_text value disposal_method' 형식의 레이블을 처리합니다.
    value는 숫자(양수 또는 음수)이며 없을 경우 0으로 설정됩니다.
    disposal_method는 쓰레기 처리 방법을 설명하는 텍스트입니다.
    
    Args:
        labels_path (str): 라벨 파일 경로
        
    Returns:
        tuple: (레이블 목록, 금액 목록, 처리방법 목록)
    """
    try:
        with open(labels_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            
        # 레이블 파싱
        labels = []
        recycling_values = []
        disposal_methods = []
        
        for line in lines:
            # 공백으로 분리
            parts = line.split(' ')
            
            if len(parts) >= 2 and parts[0].isdigit():
                # 레이블 텍스트 추출
                label_text = parts[1]
                labels.append(label_text)
                
                # 재활용 금액 추출 (있을 경우)
                value = 0
                if len(parts) >= 3:
                    try:
                        value = int(parts[2])
                    except ValueError:
                        print(f"Warning: Could not parse recycling value '{parts[2]}' for label '{label_text}'. Using default value 0.")
                
                recycling_values.append(value)
                
                # 처리 방법 추출 (있을 경우)
                method = ""
                if len(parts) >= 4:
                    # 언더스코어를 공백으로 변환하여 읽기 쉽게 만듦
                    method_parts = parts[3:]
                    method = " ".join(method_parts)
                    # 언더스코어를 공백으로 변환
                    method = method.replace('_', ' ')
                
                disposal_methods.append(method)
            else:
                # 형식이 다른 경우 그대로 사용
                labels.append(line)
                recycling_values.append(0)
                disposal_methods.append("")
                print(f"Warning: Label format not recognized for '{line}'.")
                
        print(f"라벨을 성공적으로 로드했습니다: {labels_path}")
        print(f"클래스: {labels}")
        print(f"재활용 금액: {recycling_values}")
        print(f"처리 방법: {disposal_methods}")
        return labels, recycling_values, disposal_methods
    except Exception as e:
        print(f"라벨 로드 중 오류 발생: {e}")
        return [], [], []

def get_prediction(model, processed_image):
    """
    이미지에 대한 예측을 수행합니다.
    
    Args:
        model: TensorFlow 모델
        processed_image (numpy.ndarray): 전처리된 이미지
        
    Returns:
        tuple: (예측된 클래스 인덱스, 확률, 모든 클래스의 확률 배열)
    """
    try:
        # 예측 수행
        prediction = model.predict(processed_image)
        
        # 여러 예측 형태 처리 (단일 예측 또는 다중 예측)
        if isinstance(prediction, list):
            prediction = prediction[0]
        
        # 배열 차원 확인 및 수정
        if len(prediction.shape) > 1 and prediction.shape[0] == 1:
            prediction = prediction[0]  # 첫 번째 차원 제거
            
        # 예측 배열의 길이 확인
        if len(prediction) == 0:
            print("Warning: Empty prediction array")
            # 임의의 예측 생성
            prediction = np.array([1.0])
        
        # 최대 확률을 가진 클래스 인덱스 찾기
        class_idx = np.argmax(prediction)
        
        # 안전하게 확률 추출
        if 0 <= class_idx < len(prediction):
            confidence = float(prediction[class_idx])
        else:
            print(f"Warning: Index {class_idx} out of range for prediction array with size {len(prediction)}")
            class_idx = 0
            confidence = 1.0
        
        return class_idx, confidence, prediction
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        # 예외 발생 시 기본값 반환
        return 0, 1.0, np.array([1.0])

def get_waste_info(waste_type, custom_disposal_methods=None):
    """
    쓰레기 종류에 따른 분리수거 정보를 반환합니다.
    
    Args:
        waste_type (str): 쓰레기 종류
        custom_disposal_methods (dict, optional): 사용자 정의 처리 방법
        
    Returns:
        dict: 쓰레기 정보 (처리방법, 색상, 주의사항 등)
    """
    # 모든 가능한 인식 클래스에 대한 정보 제공
    waste_info = {
        "jongphil": {
            "method": "Not a recyclable item.",
            "color": (0, 0, 255),  # BGR
            "caution": "This is a person, not waste!",
            "recyclable": False
        },
        "background": {
            "method": "No waste detected.",
            "color": (128, 128, 128),  # BGR
            "caution": "No waste item detected in the frame.",
            "recyclable": False
        },
        "Plastic": {
            "method": "Empty contents and remove labels before recycling.",
            "color": (255, 0, 0),  # BGR
            "caution": "Contaminated plastics should be disposed as general waste.",
            "recyclable": True
        },
        "Paper": {
            "method": "Remove foreign materials and tie in bundles when recycling.",
            "color": (0, 0, 255),  # BGR
            "caution": "Paper with food stains should be disposed as general waste.",
            "recyclable": True
        },
        "Glass": {
            "method": "Empty contents and remove labels before recycling.",
            "color": (0, 255, 0),  # BGR
            "caution": "Wrap broken glass in newspaper before disposal.",
            "recyclable": True
        },
        "Can": {
            "method": "Empty contents and compress before recycling.",
            "color": (255, 215, 0),  # BGR
            "caution": "Separate aluminum and steel cans when possible.",
            "recyclable": True
        },
        "General_Waste": {
            "method": "Place in standard waste bags.",
            "color": (128, 128, 128),  # BGR
            "caution": "Separate recyclable materials when possible.",
            "recyclable": False
        },
        "Food_Waste": {
            "method": "Remove excess water and place in food waste bin.",
            "color": (0, 140, 0),  # BGR
            "caution": "Remove foreign materials (bones, shells, etc.).",
            "recyclable": False
        }
    }
    
    # 기본값 설정
    default_info = {
        "method": "Check classification and dispose properly.",
        "color": (100, 100, 100),  # BGR
        "caution": "Check correct disposal method.",
        "recyclable": False
    }
    
    # 기본 정보 가져오기
    info = waste_info.get(waste_type, default_info).copy()
    
    # 사용자 정의 처리 방법이 있다면 업데이트
    if custom_disposal_methods and waste_type in custom_disposal_methods:
        custom_method = custom_disposal_methods.get(waste_type)
        if custom_method:  # 빈 문자열이 아닌 경우에만 업데이트
            info["method"] = custom_method
    
    return info 