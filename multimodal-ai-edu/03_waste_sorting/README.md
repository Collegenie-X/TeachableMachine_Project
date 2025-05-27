# 쓰레기 분류 게임 (Waste Sorting Game)

실시간 카메라를 통해 쓰레기를 인식하고 올바른 분리수거 방법을 안내하는 교육용 게임 애플리케이션입니다.

## 개요

이 프로젝트는 머신러닝을 활용하여 쓰레기를 인식하고, 해당 쓰레기의 올바른 분리수거 방법과 재활용 가치를 교육하는 인터랙티브 게임입니다. 사용자는 카메라에 쓰레기를 보여주면 시스템이 이를 인식하고 분류 정보를 제공합니다.

## 처리 프로세스 흐름도

```
[시작] → [모델/환경 초기화] → [게임 상태 초기화] → [메인 루프 시작]
   ↓                                                    ↓
[INTRO 상태] ← ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐  ↓
   │                                             │  프레임 획득/처리
   ↓ [스페이스바]                                 │    ↓
[PLAYING 상태] → [안정적 인식] → [COUNTDOWN 상태] → [WAITING 상태] → [점수 업데이트]
   ↑    │           (3초)           (결과 표시)        ↓            
   │    │                                         [타이머 종료/스페이스바]
   │    ↓ [G 키]                                       ↓
   └── [GUIDE 상태] ← ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
        [재활용 정보]
```

### 세부 처리 단계

1. **모델 및 환경 초기화 단계**
   - `load_model()`: TensorFlow 모델 로드 (models/keras_model.h5)
   - `load_labels()`: 라벨 파일 파싱 (models/labels.txt)
   - `cv2.VideoCapture()`: 카메라 장치 초기화
   - `WasteSortingGameManager()`: 게임 관리자 인스턴스 생성

2. **게임 상태 처리 루프**
   - `_game_loop()`: 게임 메인 루프 함수
   - `predict_class()`: 이미지 분류 함수
   - 상태에 따른 핸들러 함수 호출:
     - `handle_intro_state()`
     - `handle_playing_state()`
     - `handle_countdown_state()`
     - `handle_waiting_state()`
     - `handle_guide_state()`

3. **사용자 입력 처리**
   - `cv2.waitKey()`: 키보드 입력 감지
   - 스페이스바: 상태 전환 (INTRO → PLAYING, WAITING → PLAYING)
   - G 키: 가이드 화면 전환
   - R 키: 점수 초기화
   - ESC 키: 게임 종료

4. **결과 및 점수 처리**
   - 인식 안정성 확인 (`stability_counter >= required_stable_frames`)
   - 점수 계산 및 업데이트 (`total_score += recycling_values[class_idx]`)
   - 타이머 설정 및 관리 (`countdown_timer`, `result_timer`)

## 프로젝트 폴더 및 파일 구조

```
03_waste_sorting/                # 프로젝트 루트 디렉토리
├── models/                      # 모델 및 라벨 파일 디렉토리
│   ├── keras_model.h5           # Teachable Machine으로 훈련된 모델 파일 (TensorFlow 형식)
│   └── labels.txt               # 클래스 정보 파일 (형식: 클래스명,재활용가치,처리방법)
│
├── src/                         # 소스 코드 디렉토리
│   ├── game/                    # 게임 관련 모듈 (함수형 패키지)
│   │   ├── __init__.py          # 패키지 초기화 파일 (from game import * 가능하도록 설정)
│   │   ├── game_manager.py      # 게임 관리자 클래스 (상태 관리, 데이터 처리, 메인 루프)
│   │   └── state_handlers.py    # 상태별 화면 처리 함수 모음 (화면 렌더링 담당)
│   │
│   ├── utils/                   # 유틸리티 모듈 디렉토리
│   │   ├── __init__.py          # 패키지 초기화 파일 (utils 패키지 정의)
│   │   ├── image_processing.py  # 이미지 처리 함수 모음 (normalize_image, resize_image 등)
│   │   └── model_utils.py       # 모델 관련 유틸리티 (load_model, load_labels, predict_class 등)
│   │
│   ├── main.py                  # 메인 실행 파일 (모듈화된 엔트리 포인트)
│   └── waste_sorting_game.py    # 레거시 실행 파일 (하위 호환성 유지용)
│
└── README.md                    # 프로젝트 설명 문서
```

## 파일별 핵심 기능 및 주석

### 1. `game/game_manager.py`

```python
"""
게임 관리자 모듈 - 쓰레기 분류 게임의 상태와 데이터를 관리합니다.
"""

class WasteSortingGameManager:
    """
    쓰레기 분류 게임 관리자 클래스
    게임의 상태 전환과 데이터 관리를 담당합니다.
    """
    
    def __init__(self, model, class_names, recycling_values, disposal_methods, camera_id=0, image_size=224):
        """
        게임 관리자 초기화 함수
        
        Args:
            model: 로드된 TensorFlow 모델 객체
            class_names: 클래스명 리스트 (문자열)
            recycling_values: 재활용 가치 리스트 (정수)
            disposal_methods: 처리 방법 리스트 (문자열)
            camera_id: 카메라 장치 ID (기본값: 0)
            image_size: 모델 입력 이미지 크기 (기본값: 224)
        """
        # 게임 상태 변수 (INTRO, PLAYING, COUNTDOWN, WAITING, GUIDE, GAME_END)
        self.game_state = "INTRO"
        
        # 안정성 추적 변수 - 인식의 일관성을 확인
        self.stability_counter = 0
        self.required_stable_frames = 10
        
        # 타이밍 관련 변수 - 게임 진행 타이밍 제어
        self.result_delay_time = 3.0    # 스페이스바 입력 후 결과 표시까지의 지연 시간
        self.result_display_time = 2.0  # 결과 표시 시간
    
    def start(self):
        """
        게임 시작 함수
        카메라를 초기화하고 게임 메인 루프를 실행합니다.
        """
        # 카메라 초기화 및 게임 루프 실행
    
    def _game_loop(self):
        """
        게임 메인 루프 함수
        프레임 획득, 예측 수행, 상태 업데이트, 화면 렌더링을 반복합니다.
        """
        # 프레임 획득, 예측 수행, 상태 업데이트, 화면 렌더링
```

### 2. `game/state_handlers.py`

```python
"""
게임 상태 처리 모듈 - 각 게임 상태에 따른 화면 렌더링 함수를 제공합니다.
"""

def display_realtime_recognition(game_instance, frame):
    """
    실시간 인식 정보 표시 함수
    
    Args:
        game_instance: WasteSortingGameManager 인스턴스
        frame: 현재 카메라 프레임
        
    Returns:
        None (프레임에 직접 그림)
    """
    # 상단 오른쪽에 인식된 라벨과 인식률 표시

def handle_intro_state(game_instance, frame):
    """
    인트로 화면 상태 처리 함수
    게임 시작 화면을 렌더링합니다.
    
    Args:
        game_instance: WasteSortingGameManager 인스턴스
        frame: 현재 카메라 프레임
        
    Returns:
        None (프레임에 직접 그림)
    """
    # 게임 시작 화면 렌더링

def handle_playing_state(game_instance, frame, class_idx):
    """
    게임 플레이 상태 처리 함수
    실시간 인식 화면을 렌더링하고 안정성을 추적합니다.
    
    Args:
        game_instance: WasteSortingGameManager 인스턴스
        frame: 현재 카메라 프레임
        class_idx: 현재 예측된 클래스 인덱스
        
    Returns:
        None (프레임에 직접 그림)
    """
    # 실시간 인식 화면 렌더링 및 안정성 추적

def handle_countdown_state(game_instance, frame):
    """
    카운트다운 상태 처리 함수
    3초 카운트다운을 표시합니다.
    
    Args:
        game_instance: WasteSortingGameManager 인스턴스
        frame: 현재 카메라 프레임
        
    Returns:
        None (프레임에 직접 그림)
    """
    # 3초 카운트다운 표시

def handle_waiting_state(game_instance, frame):
    """
    인식 결과 대기 상태 처리 함수
    인식 결과를 표시하고 다음 단계 안내를 제공합니다.
    
    Args:
        game_instance: WasteSortingGameManager 인스턴스
        frame: 현재 카메라 프레임
        
    Returns:
        None (프레임에 직접 그림)
    """
    # 인식 결과 표시 및 다음 단계 안내

def handle_guide_state(game_instance, frame):
    """
    재활용 가이드 화면 처리 함수
    선택된 쓰레기에 대한 재활용 정보를 표시합니다.
    
    Args:
        game_instance: WasteSortingGameManager 인스턴스
        frame: 현재 카메라 프레임
        
    Returns:
        None (프레임에 직접 그림)
    """
    # 선택된 쓰레기에 대한 재활용 정보 표시
```

### 3. `utils/model_utils.py`

```python
"""
모델 유틸리티 모듈 - TensorFlow 모델 로딩 및 예측 관련 함수를 제공합니다.
"""

def load_model(model_path, use_dummy=False):
    """
    TensorFlow 모델 로딩 함수
    
    Args:
        model_path: 모델 파일 경로
        use_dummy: 더미 모델 사용 여부 (테스트용)
        
    Returns:
        로드된 모델 객체
    """
    # 모델 로딩 로직

def load_labels(labels_path):
    """
    라벨 파일 로딩 함수
    labels.txt 파일에서 클래스명, 재활용 가치, 처리 방법을 파싱합니다.
    
    Args:
        labels_path: 라벨 파일 경로
        
    Returns:
        클래스명 리스트, 재활용 가치 리스트, 처리 방법 리스트
    """
    # 라벨 파일 파싱 로직

def predict_class(model, frame, image_size=224):
    """
    이미지 클래스 예측 함수
    
    Args:
        model: 로드된 TensorFlow 모델 객체
        frame: 예측할 카메라 프레임
        image_size: 모델 입력 이미지 크기
        
    Returns:
        예측된 클래스 인덱스, 예측 확률 리스트
    """
    # 이미지 전처리 및 예측 수행
```

### 4. `main.py`

```python
"""
쓰레기 분류 게임 메인 실행 파일
모듈화된 구조로 게임을 실행합니다.
"""

def parse_arguments():
    """
    명령줄 인수 파싱 함수
    
    Returns:
        파싱된 인수 객체
    """
    # 인수 파싱 로직

def main():
    """
    메인 함수
    모델 로드, 게임 관리자 생성, 게임 시작을 수행합니다.
    """
    # 메인 실행 로직

if __name__ == "__main__":
    main()
```

## 핵심 기능

1. **머신러닝 기반 실시간 쓰레기 인식 시스템**
   - 웹캠을 통해 실시간으로 쓰레기 종류 식별 (TensorFlow/Keras 모델 활용)
   - 인식 과정: `프레임 획득` → `이미지 정규화` → `모델 예측` → `결과 시각화`
   - 상단 오른쪽에 실시간 인식 결과 및 확률 표시 (`display_realtime_recognition` 함수)
   - 안정적 인식 알고리즘 구현 (`stability_counter` 시스템으로 인식 일관성 확보)
   - 모델 로드 실패 시 더미 모델 자동 대체 기능 (`use_dummy` 옵션)

2. **환경 교육 및 재활용 가치 시각화 시스템**
   - 라벨 파일(labels.txt)에서 쓰레기별 재활용 가치 및 처리 방법 데이터 활용
   - 경제적 가치 시각화: 양수(+) 값은 녹색으로 표시(절약 금액), 음수(-) 값은 빨간색으로 표시(처리 비용)
   - 누적 점수 시스템 구현 (`total_score` 변수로 환경 보호 기여도 표시)
   - 게임화된 교육 시스템으로 환경 보호 인식 고취 (목표 달성 시 성취감 제공)
   - 애니메이션 효과로 점수 변화 시각화 (점수 증가/감소 시 색상 변화 및 크기 효과)

3. **상태 기반 게임 설계 및 유연한 전환 시스템**
   - 유한 상태 기계(FSM) 패턴을 활용한 게임 상태 관리 (INTRO, PLAYING, COUNTDOWN, WAITING, GUIDE, GAME_END)
   - 상태별 전용 처리 함수를 통한 코드 모듈화 (`handle_xxx_state` 함수군)
   - 이벤트 기반 상태 전환 (키보드 입력, 타이머 완료, 안정적 인식 등)
   - 시간 기반 자동 상태 전환 (`countdown_timer`, `result_timer` 활용)
   - 유연한 인터랙션 설계 (사용자 주도 진행과 자동 진행의 조화)

4. **상세 재활용 가이드 및 교육 시스템**
   - G 키를 통해 모든 상태에서 접근 가능한 가이드 화면 제공
   - 현재 인식된 쓰레기에 대한 상세 분리수거 방법 안내
   - 시각적 가이드라인 제공 (아이콘, 색상 코드, 텍스트 설명)
   - 재활용 가능/불가능 여부 명확히 표시 (색상 코드 및 아이콘 활용)
   - 환경 보호 팁 및 추가 정보 제공 (올바른 분리수거 방법 교육)

5. **함수형 모듈화 아키텍처**
   - 관심사 분리 원칙 적용 (상태 관리, UI 렌더링, 모델 처리 분리)
   - 패키지 구조화로 코드 가독성 및 유지보수성 강화 (game, utils 패키지)
   - 명확한 함수 인터페이스 설계 (입력 파라미터 및 반환값 문서화)
   - 상세한 함수 독스(docstring)로 코드 이해도 향상
   - 객체지향과 함수형 프로그래밍의 조화 (WasteSortingGameManager 클래스 + 순수 함수)
   - 확장 가능한 설계로 추가 기능 개발 용이 (새로운 상태 또는 기능 쉽게 추가 가능)

6. **사용자 친화적 UI/UX 시스템**
   - 상태별 최적화된 화면 구성 (필요한 정보만 표시하여 인지 부하 감소)
   - 직관적인 시각적 피드백 제공 (색상, 크기, 위치로 정보 전달)
   - 실시간 피드백 시스템 (인식률, 안정성 게이지, 점수 변화 등)
   - 타이머 시각화 (카운트다운, 결과 표시 시간을 시각적으로 표현)
   - 간결한 키보드 조작법 (게임 흐름에 맞춰 직관적인 키 배치)
   - 교육용 디자인 요소 (게임 요소와 학습 요소의 균형)

7. **테스트 및 오류 처리 시스템**
   - 더미 모델을 통한 모델 없이도 테스트 가능한 구조 (`--use_dummy` 옵션)
   - 파일 존재 여부 확인 및 예외 처리 (모델 및 라벨 파일 누락 시 대응)
   - 카메라 연결 실패 시 적절한 오류 메시지 및 복구 메커니즘
   - 명령줄 인수를 통한 유연한 설정 변경 (`argparse` 활용)
   - 개발자 및 사용자 모드 분리 (개발 시 디버그 정보 활성화 옵션)

## 실행 방법

### 1. 필요한 라이브러리 설치

```bash
# 필수 라이브러리 설치
pip install tensorflow==2.8.0 opencv-python==4.5.5.64 numpy==1.22.3

# 선택적 의존성 (성능 향상)
pip install tensorflow-metal  # Apple Silicon Mac 사용자용
```

### 2. 기본 실행 (권장)

프로젝트 루트 디렉토리에서 다음 명령을 실행합니다:

```bash
# 프로젝트 루트 디렉토리로 이동
cd 03_waste_sorting

# 모듈화된 버전 실행 (권장)
python src/main.py
```

### 3. 레거시 실행 버전

```bash
# 프로젝트 루트 디렉토리에서
python src/waste_sorting_game.py
```

### 4. 옵션을 사용한 실행

```bash
python src/main.py --model models/keras_model.h5 --labels models/labels.txt --camera 0 --image_size 224
```

#### 옵션 설명

- `--model`: 모델 파일 경로 (기본값: models/keras_model.h5)
- `--labels`: 라벨 파일 경로 (기본값: models/labels.txt)
- `--camera`: 카메라 장치 ID (기본값: 0, 여러 카메라가 있는 경우 변경)
- `--image_size`: 모델 입력 이미지 크기 (기본값: 224)
- `--use_dummy`: 더미 모델 사용 (테스트용, 모델 없이 실행 가능)

### 5. 테스트 모드 실행

모델 파일 없이 테스트하려면 더미 모델 옵션을 사용합니다:

```bash
python src/main.py --use_dummy
```

### 6. 라벨 파일 형식 (labels.txt)

```
# 형식: 클래스명,재활용가치,처리방법
배경,0,분류 불필요
종이,100,종이류 분리수거함에 버리세요
플라스틱,50,플라스틱 분리수거함에 버리세요
유리,150,유리류 분리수거함에 버리세요
일반쓰레기,-50,일반쓰레기로 버리세요
```

## 게임 조작법

- **스페이스바**: 게임 시작 및 다음 단계로 진행
- **G 키**: 현재 인식된 아이템의 재활용 가이드 화면 표시
- **R 키**: 누적 점수 초기화
- **ESC 키**: 게임 종료

## 주요 상태별 화면 구성

1. **인트로 화면** (INTRO)
   - 게임 제목 및 사용법 안내
   - 상단 오른쪽에 실시간 인식 정보
   - "Press SPACE to start" 메시지

2. **플레이 화면** (PLAYING)
   - 상단 왼쪽에 누적 점수 표시
   - 상단 오른쪽에 실시간 인식 정보
   - 인식 안정성 게이지 표시

3. **카운트다운 화면** (COUNTDOWN)
   - 중앙에 3-2-1 카운트다운 표시
   - 인식된 클래스 유지 표시
   - 3초 후 자동으로 결과 화면으로 전환

4. **결과 화면** (WAITING)
   - 인식된 쓰레기 정보 중앙에 표시
   - 재활용 가치 및 처리 방법 안내
   - 2초 타이머 표시 (자동 진행 또는 스페이스바로 스킵)

5. **가이드 화면** (GUIDE)
   - 선택된 쓰레기의 상세 재활용 정보
   - 올바른 분리수거 방법 및 주의사항
   - "Press SPACE to return" 메시지

6. **게임 종료 화면** (GAME_END)
   - 최종 누적 점수 표시
   - "Game Over" 메시지
   - "Press SPACE to restart" 메시지