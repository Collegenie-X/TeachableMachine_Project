#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
마스크 착용 감지 애플리케이션
실시간 웹캠 피드를 통해 마스크 착용 여부를 감지하고 경고를 표시합니다.
Teachable Machine에서 생성된 모델(.h5)과 라벨(.txt)을 지원합니다.
"""

import cv2
import numpy as np
import tensorflow as tf
import time
import os
import argparse
import json

# Google Drive 관련 라이브러리 (선택적)
try:
    from google.colab import drive
    from google.colab import files
    COLAB_ENV = True
except ImportError:
    COLAB_ENV = False

# 인자 파싱
parser = argparse.ArgumentParser(description='마스크 착용 감지 애플리케이션')
parser.add_argument('--model', type=str, default='../models/model.h5',
                    help='모델 파일 경로 (기본값: ../models/model.h5)')
parser.add_argument('--labels', type=str, default='../models/labels.txt',
                    help='라벨 파일 경로 (기본값: ../models/labels.txt)')
parser.add_argument('--camera', type=int, default=0,
                    help='카메라 장치 번호 (기본값: 0)')
parser.add_argument('--image_size', type=int, default=96,
                    help='입력 이미지 크기 (기본값: 96)')
parser.add_argument('--from_drive', action='store_true',
                    help='Google Drive에서 모델을 로드합니다')
parser.add_argument('--drive_model_path', type=str, 
                    default='MyDrive/TeachableMachine/mask_model/model.h5',
                    help='Google Drive 상의 모델 파일 경로')
parser.add_argument('--drive_labels_path', type=str, 
                    default='MyDrive/TeachableMachine/mask_model/labels.txt',
                    help='Google Drive 상의 라벨 파일 경로')
args = parser.parse_args()

# Google Drive에서 파일 로드 (Colab 환경인 경우)
if args.from_drive and COLAB_ENV:
    try:
        print("Google Drive 마운트 중...")
        drive.mount('/content/drive')
        model_path = f"/content/drive/{args.drive_model_path}"
        labels_path = f"/content/drive/{args.drive_labels_path}"
        print(f"Drive에서 모델 로드: {model_path}")
        print(f"Drive에서 라벨 로드: {labels_path}")
    except Exception as e:
        print(f"Google Drive 마운트 실패: {e}")
        exit(1)
else:
    model_path = args.model
    labels_path = args.labels

# 얼굴 탐지를 위한 하르 캐스케이드 분류기 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 라벨 파일 로드
try:
    with open(labels_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    print(f"라벨 로드 성공: {class_names}")
except Exception as e:
    print(f"라벨 파일 로드 실패: {e}")
    print("기본 라벨을 사용합니다: ['마스크 착용', '마스크 미착용']")
    class_names = ['마스크 착용', '마스크 미착용']

# 모델 로드
try:
    model = tf.keras.models.load_model(model_path)
    print(f"모델 로드 성공: {model_path}")
    model.summary()
except Exception as e:
    print(f"모델 로드 실패: {e}")
    print("훈련된 모델이 필요합니다. Teachable Machine에서 모델을 다운로드하세요.")
    exit(1)

# 웹캠 설정
cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    print(f"카메라 {args.camera}를 열 수 없습니다.")
    exit(1)

# 이미지 크기 설정
IMG_SIZE = args.image_size

# 상습 위반자 추적을 위한 데이터 구조
violation_counter = {}
VIOLATION_THRESHOLD = 5  # 상습 위반자로 간주할 위반 횟수

# 이미지 전처리 함수
def preprocess_image(img):
    # 이미지 크기 조정 및 정규화
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# 메인 루프
print("마스크 감지 시스템 시작. 'q'를 눌러 종료하세요.")
frame_count = 0
last_violation_check = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라에서 프레임을 받아올 수 없습니다.")
        break
    
    # 화면 크기 조정
    frame = cv2.resize(frame, (640, 480))
    
    # 얼굴 탐지 (5프레임마다 처리하여 성능 향상)
    frame_count += 1
    if frame_count % 5 == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
        # 각 얼굴에 대해 마스크 착용 여부 예측
        current_time = time.time()
        for i, (x, y, w, h) in enumerate(faces):
            # 얼굴 영역 추출
            face_id = f"face_{i}"
            face_img = frame[y:y+h, x:x+w]
            
            try:
                # 예측을 위한 전처리
                processed_face = preprocess_image(face_img)
                
                # 마스크 착용 여부 예측
                prediction = model.predict(processed_face, verbose=0)
                
                # 예측 결과 및 신뢰도 추출
                class_idx = np.argmax(prediction[0])
                confidence = float(prediction[0][class_idx])
                label = class_names[class_idx]
                
                # 결과 표시
                if class_idx == 0:  # 마스크 착용 (첫 번째 클래스)
                    color = (0, 255, 0)  # 초록색
                    result_text = f"{label}: {confidence*100:.2f}%"
                    # 위반 카운트 초기화
                    if face_id in violation_counter:
                        violation_counter[face_id] = 0
                else:  # 마스크 미착용
                    color = (0, 0, 255)  # 빨간색
                    result_text = f"{label}: {confidence*100:.2f}%"
                    
                    # 위반 카운트 증가
                    if face_id in violation_counter:
                        violation_counter[face_id] += 1
                    else:
                        violation_counter[face_id] = 1
                    
                    # 경고 메시지
                    cv2.putText(frame, "마스크를 착용해주세요!", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    # 상습 위반자 체크
                    if violation_counter[face_id] >= VIOLATION_THRESHOLD:
                        cv2.putText(frame, "상습 위반자!", (x, y-30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # 얼굴 주위에 사각형 그리기
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # 예측 결과 텍스트 표시
                cv2.putText(frame, result_text, (x, y+h+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
            except Exception as e:
                print(f"예측 오류: {e}")
        
        # 오래된 얼굴 ID 제거 (30초마다)
        if current_time - last_violation_check > 30:
            violation_counter = {}
            last_violation_check = current_time
    
    # 시스템 정보 표시
    cv2.putText(frame, f"모델: {os.path.basename(model_path)}", 
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"해상도: {IMG_SIZE}x{IMG_SIZE}", 
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 안내 메시지
    cv2.putText(frame, "종료: 'q' 키", 
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 결과 표시
    cv2.imshow('마스크 착용 감지 시스템', frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
print("마스크 감지 시스템 종료") 