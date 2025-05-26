# 멀티모달 AI 교육 프로젝트

이 저장소는 멀티모달 AI 교육을 위한 세 가지 실용적인 프로젝트를 포함하고 있습니다.

## 프로젝트 구조

```
multimodal-ai-edu/
├── 01_mask_wear/                # 프로젝트1: 마스크 착용 인식
│   ├── data/                    # 데이터셋 업로드 폴더
│   ├── notebooks/               # 모델 훈련·테스트 노트북
│   ├── models/                  # 학습된 모델 파일 (.h5, tflite 등)
│   └── src/                     # 실시간 경고 시스템 코드
├── 02_hand_gesture/             # 프로젝트2: 손동작/수어 인식
│   ├── data/
│   ├── notebooks/
│   ├── models/
│   └── src/                     # 게임·번역 앱 코드
├── 03_waste_sorting/            # 프로젝트3: 쓰레기 분류
│   ├── data/
│   ├── notebooks/
│   ├── models/
│   └── src/                     # 분리수거 가이드 시스템 코드
└── README.md                    # 전체 개요 및 실행 방법
```

## 프로젝트 개요

### 01_mask_wear: 마스크 착용 인식
- 실시간 카메라를 통해 마스크 착용 여부를 감지
- 올바른 마스크 착용 방법 안내 및 미착용시 경고 시스템

### 02_hand_gesture: 손동작/수어 인식
- 다양한 손동작을 인식하여 게임 조작 또는 인터페이스 제어
- 기본적인 수어 번역 기능 구현

### 03_waste_sorting: 쓰레기 분류
- 이미지 인식을 통한 쓰레기 분류 시스템
- 올바른 분리수거 방법 안내

## 실행 방법

각 프로젝트 폴더의 notebooks/ 디렉토리에는 모델 훈련 및 테스트를 위한 주피터 노트북이 포함되어 있습니다.
실제 애플리케이션 실행은 각 프로젝트의 src/ 디렉토리에 있는 코드를 참조하세요.

### 환경 설정
```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 필요한 패키지 설치
pip install -r requirements.txt
```

각 프로젝트별 자세한 실행 방법은 해당 디렉토리의 README를 참조하세요. 