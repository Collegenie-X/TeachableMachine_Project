# 손동작 인식 시스템 (Hand Gesture Recognition)

이 애플리케이션은 Teachable Machine 모델을 사용하여 두 가지 다른 애플리케이션에서 손동작을 인식합니다:
1. 가위바위보 게임 (Rock Paper Scissors)
2. 수어 번역기 (Sign Language Translator)

## 처리 프로세스 흐름도

### 1. 가위바위보 게임 프로세스

```
[시작] → [모델/환경 초기화] → [메인 루프 시작]
   ↓                             ↓
[IDLE 상태] ← ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐ ↓
   │                           │ 프레임 획득/처리
   ↓ [스페이스바]                │ ↓
[COUNTDOWN 상태] → [3초 타이머] → [PLAYING 상태] → [손동작 인식 및 안정화]
                                   ↓                ↓
                                   │ [시간 초과]   [안정적 인식 달성]
                                   │                ↓
                                   └───────→ [RESULT 상태] → [승패 결정]
                                                     ↓
                                                [스페이스바]
```

### 2. 수어 번역기 프로세스

```
[시작] → [모델/환경 초기화] → [메인 루프 시작]
   ↓                             ↓
[IDLE 상태] ← ─ ─ ─ ─ ─ ─ ─ ─ ┐  프레임 획득/처리
   │         [ESC 키]         │    ↓
   │                          │ [손동작 인식 및 번역]
   ↓ ['c' 키]                 │    ↓
[CHALLENGE 상태] ← ─ ─ ─ ─ ─ ─┼─ [실시간 번역 표시]
   │                          │
   ↓ [5초 타이머]             │
[목표 수어 제시] → [수어 인식] → [RESULT 상태] → [성공/실패 결과]
                                   ↓
                               [스페이스바]
```

### 세부 처리 단계

1. **모델 및 환경 초기화 단계**
   - `patch_depthwise_conv()`: TensorFlow 호환성 패치 적용
   - `load_model_and_labels()`: 모델 및 라벨 파일 로드
   - `__init__()`: 클래스 인스턴스 초기화 (RockPaperScissorsGame 또는 SignLanguageTranslator)
   - `cv2.VideoCapture()`: 카메라 장치 초기화

2. **메인 루프 및 상태 처리**
   - `start()`: 메인 루프 시작
   - `_process_frame()`: 프레임 획득 및 처리
   - `_get_prediction()`: 모델 예측 수행
   - 상태에 따른 처리 함수 호출:
     - `_handle_idle_state()`
     - `_handle_countdown_state()`
     - `_handle_playing_state()`
     - `_handle_result_state()`
     - `_handle_challenge_state()`

3. **사용자 입력 처리**
   - `cv2.waitKey()`: 키보드 입력 감지
   - 스페이스바: 게임 시작/재시작
   - 'c' 키: 수어 도전 모드 시작
   - ESC 키: 종료

4. **결과 및 피드백 처리**
   - 손동작 안정화 확인 (`stability_counter >= required_stable_frames`)
   - 타이머 설정 및 관리
   - 시각적 피드백 렌더링 (텍스트, 색상, 오버레이)

## 프로젝트 폴더 및 파일 구조

```
02_hand_gesture/                   # 프로젝트 루트 디렉토리
├── models/                        # 모델 파일 디렉토리
│   ├── rps_model/                 # 가위바위보 모델 폴더
│   │   ├── keras_model.h5         # Teachable Machine 훈련 모델
│   │   └── labels.txt             # 클래스 라벨 파일 (가위,바위,보)
│   └── sign_model/                # 수어 번역 모델 폴더
│       ├── keras_model.h5         # 수어 인식 모델
│       └── labels.txt             # 수어 클래스 라벨 파일
│
├── data/                          # 학습 데이터 디렉토리
│   ├── rps_images/                # 가위바위보 학습 이미지
│   └── sign_images/               # 수어 학습 이미지
│
├── notebooks/                     # 주피터 노트북 디렉토리
│   ├── train_rps_model.ipynb      # 가위바위보 모델 학습 노트북
│   └── train_sign_model.ipynb     # 수어 모델 학습 노트북
│
├── src/                           # 소스 코드 디렉토리
│   ├── main.py                    # 메인 진입점 (실행 모드 선택)
│   ├── rock_paper_scissors.py     # 가위바위보 게임 클래스 구현
│   ├── sign_language.py           # 수어 번역기 클래스 구현
│   ├── game_control.py            # 레거시 코드 (참조용)
│   └── utils/                     # 유틸리티 패키지
│       ├── __init__.py            # 패키지 초기화 파일
│       ├── image_processing.py    # 이미지 처리 함수 모음
│       └── model_utils.py         # 모델 관련 유틸리티 함수
│
└── README.md                      # 프로젝트 설명 문서
```

## 파일별 핵심 기능 및 주석

### 1. `main.py`

```python
"""
손동작 인식 시스템 - 메인 애플리케이션
애플리케이션 실행을 위한 진입점입니다.

주요 기능:
1. 명령줄 인수 파싱
2. TensorFlow 호환성 패치 적용
3. 모델 및 라벨 로드
4. 선택된 모드에 따른 애플리케이션 실행
"""

def main():
    """
    메인 함수 - 인수를 파싱하고 애플리케이션을 실행합니다.
    
    Args:
        None
        
    Returns:
        None: 선택된 애플리케이션을 실행합니다.
    """
    # TensorFlow 호환성 패치 적용
    # 인수 파싱 및 모델 로드
    # 선택된 애플리케이션 실행
```

### 2. `rock_paper_scissors.py`

```python
"""
가위바위보 게임 모듈
실시간 손동작 인식을 통한 가위바위보 게임을 구현합니다.

클래스:
- RockPaperScissorsGame: 가위바위보 게임 실행 및 관리

상태:
- IDLE: 게임 시작 대기 상태
- COUNTDOWN: 3초 카운트다운 상태
- PLAYING: 게임 진행 중 상태
- RESULT: 결과 표시 상태
"""

class RockPaperScissorsGame:
    """
    가위바위보 게임 클래스
    실시간 손동작 인식을 통해 가위바위보 게임을 실행합니다.
    
    Attributes:
        model: 로드된 TensorFlow 모델
        class_names: 클래스 이름 목록 (가위, 바위, 보)
        camera_id: 카메라 장치 ID
        image_size: 모델 입력 이미지 크기
        state: 현재 게임 상태
        stability_counter: 손동작 안정화 카운터
    """
    
    def __init__(self, model, class_names, camera_id=0, image_size=224):
        """
        가위바위보 게임 초기화
        
        Args:
            model: 로드된 TensorFlow 모델
            class_names: 클래스 이름 목록
            camera_id: 카메라 장치 ID (기본값: 0)
            image_size: 모델 입력 이미지 크기 (기본값: 224)
        """
        # 초기화 로직
```

### 3. `sign_language.py`

```python
"""
수어 번역기 모듈
실시간 손동작 인식을 통한 수어 번역 및 학습 게임을 구현합니다.

클래스:
- SignLanguageTranslator: 수어 번역 및 도전 모드 실행

상태:
- IDLE: 일반 번역 모드
- CHALLENGE: 도전 과제 모드
- RESULT: 도전 결과 표시 상태
"""

class SignLanguageTranslator:
    """
    수어 번역기 클래스
    실시간 손동작 인식을 통해 수어를 번역하고 학습 게임을 제공합니다.
    
    Attributes:
        model: 로드된 TensorFlow 모델
        class_names: 수어 클래스 이름 목록
        camera_id: 카메라 장치 ID
        image_size: 모델 입력 이미지 크기
        state: 현재 번역기 상태
        challenge_word: 현재 도전 수어 단어
    """
    
    def __init__(self, model, class_names, camera_id=0, image_size=224):
        """
        수어 번역기 초기화
        
        Args:
            model: 로드된 TensorFlow 모델
            class_names: 클래스 이름 목록
            camera_id: 카메라 장치 ID (기본값: 0)
            image_size: 모델 입력 이미지 크기 (기본값: 224)
        """
        # 초기화 로직
```

### 4. `utils/model_utils.py`

```python
"""
모델 유틸리티 모듈
TensorFlow 모델 로딩 및 예측 관련 유틸리티 함수를 제공합니다.

주요 함수:
- patch_depthwise_conv: Teachable Machine 모델 호환성 패치
- load_model_and_labels: 모델 및 라벨 파일 로드
- clean_label: 라벨 텍스트 정리
- get_prediction: 이미지에서 예측 수행
"""

def patch_depthwise_conv():
    """
    DepthwiseConv2D 클래스를 패치하여 TensorFlow 호환성 문제를 해결합니다.
    Teachable Machine 모델이 'groups' 파라미터 관련 오류 없이 로드되도록 합니다.
    
    Args:
        None
        
    Returns:
        None
    """
    # 패치 로직

def load_model_and_labels(model_path, labels_path):
    """
    모델 및 라벨 파일을 로드합니다.
    
    Args:
        model_path: 모델 파일 경로
        labels_path: 라벨 파일 경로
        
    Returns:
        tuple: (로드된 모델, 클래스 이름 목록)
    """
    # 모델 및 라벨 로드 로직
```

### 5. `utils/image_processing.py`

```python
"""
이미지 처리 유틸리티 모듈
카메라 이미지 처리 및 UI 렌더링 관련 함수를 제공합니다.

주요 함수:
- preprocess_image: 모델 입력용 이미지 전처리
- put_text_with_background: 배경이 있는 텍스트 렌더링
- center_text: 화면 중앙 텍스트 위치 계산
- create_transparent_overlay: 반투명 오버레이 생성
"""

def preprocess_image(frame, image_size):
    """
    카메라 프레임을 모델 입력용으로 전처리합니다.
    
    Args:
        frame: 원본 카메라 프레임
        image_size: 목표 이미지 크기
        
    Returns:
        numpy.ndarray: 전처리된 이미지
    """
    # 이미지 전처리 로직

def put_text_with_background(frame, text, position, font_scale=1, thickness=2, 
                           text_color=(255, 255, 255), bg_color=(0, 0, 0), bg_alpha=0.5):
    """
    반투명 배경이 있는 텍스트를 프레임에 렌더링합니다.
    
    Args:
        frame: 대상 프레임
        text: 표시할 텍스트
        position: 텍스트 위치 (x, y)
        font_scale: 폰트 크기 (기본값: 1)
        thickness: 텍스트 두께 (기본값: 2)
        text_color: 텍스트 색상 (기본값: 흰색)
        bg_color: 배경 색상 (기본값: 검정색)
        bg_alpha: 배경 투명도 (기본값: 0.5)
        
    Returns:
        None: 프레임에 직접 그립니다.
    """
    # 텍스트 렌더링 로직
```

## 핵심 기능

1. **실시간 손동작 인식 및 처리 엔진**
   - TensorFlow/Keras 기반 딥러닝 모델을 사용한 실시간 손동작 분류
   - 프레임 처리 파이프라인: `프레임 획득` → `이미지 전처리` → `모델 예측` → `결과 후처리`
   - 60FPS 이상의 처리 속도 지원 (하드웨어에 따라 다름)
   - 배경 및 조명 변화에 강인한 인식 알고리즘
   - 모델 교체 없이 다양한 손동작 세트 지원 (라벨 파일만 교체)

2. **손동작 안정화 알고리즘**
   - 일시적 오인식 방지를 위한 안정화 카운터 시스템 (`stability_counter`)
   - 설정 가능한 안정성 임계값으로 인식 정확도와 속도 조절 (`required_stable_frames`)
   - 연속된 프레임에서 동일한 동작이 감지될 때만 인식 확정
   - 인식 확률 가중치 적용으로 높은 확률의 예측에 우선권 부여
   - 동작 전환 시 안정화 카운터 리셋으로 빠른 새 동작 인식

3. **시간 기반 인터랙션 시스템**
   - 타이머 기반 상태 전환 로직으로 게임 흐름 제어
   - 가위바위보: 3초 카운트다운 타이머 (`countdown_timer`)
   - 수어 도전 모드: 5초 제한 시간 타이머 (`challenge_timer`)
   - 경과 시간에 비례한 시각적 피드백 (색상 변화, 게이지 바)
   - 시간 기반 점수 시스템 (남은 시간에 비례한 점수 획득)
   - 타이머 기반 자동 상태 전환으로 게임 흐름 유지

4. **고급 UI 렌더링 시스템**
   - 반투명 배경을 가진 텍스트 렌더링 (`put_text_with_background()`)
   - 화면 크기에 상관없이 텍스트 중앙 정렬 (`center_text()`)
   - 다양한 폰트 크기와 색상 조합 지원
   - 애니메이션 효과 (점수 증가, 타이머 진행, 결과 표시)
   - 상태별 최적화된 화면 레이아웃 (게임 상태에 따른 정보 표시)
   - 정보 계층화로 가독성 향상 (중요 정보는 크게, 부가 정보는 작게)

5. **상태 기반 애플리케이션 아키텍처**
   - 유한 상태 기계(FSM) 패턴을 적용한 애플리케이션 설계
   - 가위바위보: IDLE, COUNTDOWN, PLAYING, RESULT 상태
   - 수어 번역기: IDLE, CHALLENGE, RESULT 상태
   - 상태별 전용 처리 함수로 코드 모듈화 (`_handle_xxx_state()` 함수군)
   - 키보드 및 타이머 이벤트에 따른 상태 전환
   - 상태 간 데이터 전달 메커니즘 (게임 결과, 점수 등 유지)

6. **모듈화된 객체지향 설계**
   - 클래스 기반 애플리케이션 구조 (RockPaperScissorsGame, SignLanguageTranslator)
   - 유틸리티 함수를 별도 모듈로 분리 (model_utils, image_processing)
   - 설정 가능한 클래스 속성으로 동작 커스터마이징
   - 코드 재사용성 향상을 위한 공통 기능 추상화
   - 클래스 상속 및 메서드 오버라이딩을 통한 기능 확장 가능

7. **교육 및 게임화 기능**
   - 가위바위보: 컴퓨터와의 대결, 승패 추적, 연승 기록
   - 수어 번역기: 도전 모드를 통한 수어 학습 게임화
   - 점수 시스템을 통한 동기 부여 및 진행 상황 추적
   - 시각적 피드백으로 학습 효과 향상 (성공/실패 애니메이션)
   - 난이도 조절 가능 (인식 임계값, 시간 제한 등 조정)

8. **오류 처리 및 호환성 시스템**
   - TensorFlow 버전 호환성 자동 패치 적용 (`patch_depthwise_conv()`)
   - 카메라 연결 실패 시 명확한 오류 메시지 및 복구 시도
   - 모델 로드 실패 처리 및 사용자 안내
   - 화면 크기 자동 조정으로 다양한 해상도 지원
   - 예외 상황 처리로 애플리케이션 안정성 확보

## 실행 방법

### 1. 필요한 라이브러리 설치

```bash
# 기본 필수 라이브러리 설치
pip install tensorflow==2.10.0 opencv-python==4.6.0.66 numpy==1.23.4

# 선택적 의존성 (성능 향상)
pip install tensorflow-metal  # Apple Silicon Mac 사용자용
```

### 2. 가위바위보 게임 실행

```bash
# 프로젝트 루트 디렉토리에서
cd 02_hand_gesture/src

# 기본 실행
python main.py --mode rps

# 사용자 모델 지정
python main.py --mode rps --model ../models/rps_model/keras_model.h5 --labels ../models/rps_model/labels.txt
```

### 3. 수어 번역기 실행

```bash
# 프로젝트 루트 디렉토리에서
cd 02_hand_gesture/src

# 기본 실행
python main.py --mode sign

# 사용자 모델 지정
python main.py --mode sign --model ../models/sign_model/keras_model.h5 --labels ../models/sign_model/labels.txt
```

### 4. 실행 옵션 설명

```
--mode [rps|sign]: 애플리케이션 모드 선택 (필수)
  - rps: 가위바위보 게임
  - sign: 수어 번역기

--model PATH: 모델 파일 경로 (기본값: ../models/converted_keras/keras_model.h5)
--labels PATH: 라벨 파일 경로 (기본값: ../models/converted_keras/labels.txt)
--camera NUM: 카메라 장치 번호 (기본값: 0)
--image_size NUM: 모델 입력 이미지 크기 (기본값: 224)
```

### 5. 라벨 파일 형식

```
# 가위바위보 모델 라벨 예시 (labels.txt)
0 배경
1 가위
2 바위
3 보

# 수어 모델 라벨 예시 (labels.txt)
0 배경
1 안녕하세요
2 감사합니다
3 미안합니다
4 사랑합니다
5 도와주세요
```

## 게임 조작법

### 가위바위보 게임

- **스페이스바**: 게임 시작 / 다음 라운드 시작
- **ESC 키**: 게임 종료
- **R 키**: 점수 초기화

### 수어 번역기

- **C 키**: 도전 모드 시작 (랜덤 수어 과제 제시)
- **스페이스바**: 결과 화면에서 다음으로 진행
- **ESC 키**: 애플리케이션 종료
- **S 키**: 통계 보기 (도전 성공/실패 기록)

## 문제 해결

### 카메라 문제
- 카메라가 다른 애플리케이션에서 사용 중이 아닌지 확인
- 카메라 장치 번호가 올바른지 확인 (다른 번호 시도)
- 카메라 드라이버가 제대로 설치되었는지 확인

### 모델 정확도 문제
- 각 동작에 충분한 학습 데이터 확보
- 다양한 조명 조건과 배경에서 학습
- 손동작이 카메라 프레임 내에 잘 보이도록 유지

### 패키지 설치 문제
- 필요한 패키지가 모두 설치되었는지 확인
  ```bash
  pip install opencv-python tensorflow numpy
  ```
- 호환성 문제 발생 시 버전 지정 설치
  ```bash
  pip install tensorflow==2.10.0
  ```
