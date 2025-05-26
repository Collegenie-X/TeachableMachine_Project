# 마스크 착용 인식 프로젝트 노트북

이 디렉토리는 마스크 착용 인식 프로젝트를 위한 Jupyter 노트북 파일들을 포함하고 있습니다.

## 주요 노트북 파일

### 1. [teachable_machine_uploader.ipynb](./teachable_machine_uploader.ipynb)
- **목적**: Teachable Machine에서 생성한 모델 파일(.h5)과 라벨 파일(.txt)을 업로드하고 테스트
- **주요 기능**:
  - 로컬 또는 Google Drive에서 모델 파일 업로드
  - 모델 정보 확인 및 요약 표시
  - 테스트 이미지를 사용한 모델 성능 검증

### 2. [mask_detection_app.ipynb](./mask_detection_app.ipynb)
- **목적**: 업로드된 모델을 사용하여 마스크 착용 여부를 실시간으로 감지
- **주요 기능**:
  - 이미지 또는 웹캠을 통한 마스크 착용 감지
  - 실시간 경고 시스템 (마스크 미착용 시)
  - 상습 위반자 추적 기능

### 3. [train_model.ipynb](./train_model.ipynb)
- **목적**: 직접 마스크 착용 감지 모델을 훈련
- **주요 기능**:
  - 데이터 로드 및 전처리
  - CNN 모델 구성 및 훈련
  - 모델 평가 및 저장

## 사용 방법

### Teachable Machine으로 모델 생성 후 사용 (권장)

1. [Teachable Machine](https://teachablemachine.withgoogle.com/)에서 이미지 프로젝트 생성
2. 마스크 착용/미착용 이미지로 모델 훈련
3. 모델 내보내기 (TensorFlow 형식 선택)
4. `teachable_machine_uploader.ipynb` 노트북을 실행하여 모델 업로드
5. `mask_detection_app.ipynb` 노트북을 실행하여 모델 테스트 및 실시간 감지

### 직접 모델 훈련 (선택적)

1. `data/` 디렉토리에 마스크 착용/미착용 이미지 데이터셋 준비
2. `train_model.ipynb` 노트북을 실행하여 모델 훈련
3. `mask_detection_app.ipynb` 노트북을 실행하여 훈련된 모델로 실시간 감지

## 필요한 라이브러리

모든 노트북은 다음 라이브러리에 의존합니다:
- tensorflow
- opencv-python
- numpy
- matplotlib
- ipywidgets (Jupyter 위젯용) 