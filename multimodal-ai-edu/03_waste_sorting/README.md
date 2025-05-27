# 쓰레기 분류 게임 (Waste Sorting Game)

실시간 카메라를 통해 쓰레기를 인식하고 올바른 분리수거 방법을 안내하는 교육용 게임 애플리케이션입니다.

## 개요

이 프로젝트는 머신러닝을 활용하여 쓰레기를 인식하고, 해당 쓰레기의 올바른 분리수거 방법과 재활용 가치를 교육하는 인터랙티브 게임입니다. 사용자는 카메라에 쓰레기를 보여주면 시스템이 이를 인식하고 분류 정보를 제공합니다.

## 주요 기능

- 실시간 쓰레기 인식 및 분류
- 재활용 가치 표시 (양수: 절약 금액, 음수: 처리 비용)
- 올바른 처리 방법 안내
- 누적 점수 시스템
- 상세 재활용 가이드 제공

## 프로젝트 구조

```
03_waste_sorting/
├── models/                     # 모델 및 라벨 파일
│   ├── keras_model.h5          # 훈련된 머신러닝 모델
│   └── labels.txt              # 라벨 정보 (클래스, 재활용 가치, 처리 방법)
├── src/                        # 소스 코드
│   ├── game/                   # 게임 모듈
│   │   ├── __init__.py         # 패키지 초기화
│   │   ├── game_manager.py     # 게임 관리자 클래스
│   │   └── state_handlers.py   # 게임 상태 처리 함수
│   ├── utils/                  # 유틸리티 모듈
│   │   ├── image_processing.py # 이미지 처리 함수
│   │   └── model_utils.py      # 모델 관련 유틸리티
│   ├── main.py                 # 메인 모듈 (진입점)
│   └── waste_sorting_game.py   # 레거시 진입점 (리다이렉트)
└── README.md                   # 프로젝트 설명
```

## 모듈 구조 설명

### 1. 게임 모듈 (`game/`)

- **게임 관리자 (`game_manager.py`)**
  - `WasteSortingGameManager` 클래스: 게임의 상태와 데이터를 관리하고 게임 루프를 실행
  - 카메라 입력 처리, 예측 수행, 키보드 입력 처리 등의 핵심 로직 포함

- **상태 처리기 (`state_handlers.py`)**
  - 각 게임 상태별 화면 렌더링 함수 제공
  - `handle_intro_state`: 인트로 화면 처리
  - `handle_playing_state`: 게임 플레이 상태 처리
  - `handle_waiting_state`: 인식 후 대기 상태 처리
  - `handle_guide_state`: 가이드 화면 처리

### 2. 유틸리티 모듈 (`utils/`)

- **이미지 처리 (`image_processing.py`)**
  - 카메라 입력 전처리
  - UI 요소 렌더링 함수
  
- **모델 유틸리티 (`model_utils.py`)**
  - 모델 로딩 및 예측
  - 라벨 파싱
  - 쓰레기 정보 조회

### 3. 메인 모듈 (`main.py`)

- 명령행 인자 처리
- 모델 및 라벨 로딩
- 게임 인스턴스 생성 및 실행

## 라벨 파일 형식

`labels.txt` 파일은 다음 형식을 따릅니다:

```
0 class_name recycling_value disposal_method
```

예시:
```
0 jongphil 100 Not_a_recyclable_item     
1 background -50 No_waste_detected   
2 Plastic 50 Rinse_and_remove_labels_before_recycling
```

- 첫 번째 필드: 클래스 인덱스
- 두 번째 필드: 클래스 이름
- 세 번째 필드: 재활용 금액 (양수: 절약, 음수: 비용)
- 네 번째 필드 이후: 처리 방법 (언더스코어로 단어 구분)

## 실행 방법

### 기본 실행

```bash
python src/main.py
```

### 옵션 사용

```bash
python main.py --model ../models/keras_model.h5 --labels ../models/labels.txt
```

#### 옵션 설명

- `--model`: 모델 파일 경로 (기본값: ../models/keras_model.h5)
- `--labels`: 라벨 파일 경로 (기본값: ../models/labels.txt)
- `--camera`: 카메라 장치 ID (기본값: 0)
- `--image_size`: 모델 입력 이미지 크기 (기본값: 224)
- `--use_dummy`: 더미 모델 사용 (테스트용)

### 테스트 모드

모델 없이 테스트하려면 더미 모델 옵션을 사용하세요:

```bash
python src/main.py --use_dummy
```

## 게임 조작법

- **스페이스바**: 게임 시작/다음 아이템으로 진행
- **G 키**: 현재 인식된 아이템의 재활용 가이드 보기
- **R 키**: 누적 점수 초기화
- **ESC 키**: 게임 종료 