# 마스크 착용 인식 프로젝트 (Mask Detection System)

실시간 카메라를 통해 마스크 착용 여부를 감지하고, 올바른 마스크 착용 방법을 안내하는 교육용 애플리케이션입니다.

## 처리 프로세스 흐름도

```
[시작] → [모델/환경 초기화] → [카메라 활성화] → [메인 루프 시작]
   ↓                                                   ↓
   │                                            [프레임 획득 및 처리]
   │                                                   ↓
   │                                            [얼굴 검출 (5프레임마다)]
   │                                                   ↓
   │                                            [검출된 얼굴 영역 추출]
   │                                                   ↓
   │                                            [이미지 전처리 (정규화)]
   │                                                   ↓
   │                                            [마스크 착용 예측 수행]
   │                                                   ↓
   │                              ┌─────────→ [결과 판정 및 표시] ───────┐
   │                              │                  ↓                   │
   │                              │            [위반 기록 관리]           │
   │                              │                  ↓                   │
   │                              │           [키 입력 확인]              │
   │                              │                  ↓                   │
   │                        'q' 키 입력           키 입력 없음            │
   │                              │                  ↓                   │
   └─────────────────────────── [종료]        [다음 프레임 처리] ← ─ ─ ─ ┘
```

### 세부 처리 단계

1. **초기화 및 설정 단계**
   - `_custom_layer_fix()`: TensorFlow 호환성 패치 적용 (Teachable Machine 모델 로드 시 'groups' 파라미터 충돌 해결)
   - `argparse`: 명령줄 인수 파싱 (모델 경로, 라벨 경로, 카메라 ID 등)
   - `tf.keras.models.load_model()`: 마스크 분류 모델 로드
   - `cv2.CascadeClassifier()`: 얼굴 검출 모델 로드
   - `cv2.VideoCapture()`: 카메라 장치 초기화

2. **프레임 처리 파이프라인**
   - `cap.read()`: 카메라로부터 프레임 획득
   - `cv2.resize()`: 화면 표시용 프레임 크기 조정
   - `cv2.cvtColor()`: 얼굴 검출을 위한 그레이스케일 변환 (5프레임마다)
   - `face_cascade.detectMultiScale()`: 얼굴 영역 검출
   - `preprocess_image()`: 검출된 얼굴 영역을 모델 입력 형식으로 전처리
   - `model.predict()`: 마스크 착용 여부 예측 수행

3. **결과 처리 및 표시**
   - `np.argmax()`: 예측 결과 중 최대 확률 클래스 선택
   - `put_text_with_background()`: 반투명 배경을 갖는 텍스트 표시
   - `cv2.rectangle()`: 마스크 착용 상태에 따른 색상 프레임 표시 (녹색: 착용, 빨간색: 미착용)
   - 위반 추적: 마스크 미착용 상태 지속 시간 추적 및 경고 메시지 관리
   - `cv2.imshow()`: 결과 화면 표시

4. **이벤트 처리 및 종료**
   - `cv2.waitKey()`: 키보드 입력 감지 (1ms 대기)
   - 'q' 키 입력 시 프로그램 종료
   - `cap.release()`: 카메라 리소스 해제
   - `cv2.destroyAllWindows()`: 모든 OpenCV 창 닫기

## 프로젝트 폴더 및 파일 구조

```
01_mask_wear/                      # 프로젝트 루트 디렉토리
│
├── data/                          # 학습 및 테스트용 데이터셋
│   ├── mask/                      # 마스크 착용 이미지들 (학습용)
│   └── no_mask/                   # 마스크 미착용 이미지들 (학습용)
│
├── models/                        # 학습된 모델 파일 저장 디렉토리
│   ├── converted_keras/           # Teachable Machine 내보내기 폴더
│   │   ├── keras_model.h5         # TensorFlow/Keras 모델 파일 (기본)
│   │   └── labels.txt             # 클래스 레이블 정보 (0: Mask, 1: No Mask)
│   └── converted_keras.zip        # 모델 파일 백업 압축본
│
├── src/                           # 소스 코드 디렉토리
│   ├── mask_detection_app.py      # 마스크 인식 애플리케이션 메인 파일
│   │   ├── _custom_layer_fix()    # Teachable Machine 모델 호환성 패치 함수
│   │   ├── preprocess_image()     # 이미지 전처리 함수 (resize, normalize)
│   │   ├── put_text_with_background() # 배경이 있는 텍스트 표시 함수
│   │   └── main loop              # 프레임 처리 및 마스크 감지 메인 루프
│   └── test_model.py              # 모델 테스트 및 정확도 평가 스크립트
│
├── notebooks/                     # 주피터 노트북 디렉토리
│   └── train_model.ipynb          # 마스크 분류 모델 학습 노트북
│
├── requirements.txt               # 필요 라이브러리 목록 (의존성)
├── HOWTO_RUN.md                   # 자세한 실행 가이드
└── README.md                      # 프로젝트 설명 문서 (현재 파일)
```

## 파일별 핵심 기능 및 주석

### mask_detection_app.py

```python
"""
마스크 인식 애플리케이션
실시간 웹캠 피드를 통해 마스크 착용 여부를 감지합니다.
Teachable Machine에서 생성된 모델(.h5)과 라벨(.txt)을 지원합니다.

주요 기능:
1. 실시간 얼굴 검출 (OpenCV Haar Cascade 사용)
2. 마스크 착용 여부 분류 (TensorFlow 모델 사용)
3. 시각적 피드백 (착용/미착용 상태 표시)
4. 반복 위반자 추적 및 경고 메시지 관리
"""

def _custom_layer_fix():
    """
    Teachable Machine 모델 로드 시 호환성 문제 해결을 위한 패치 함수
    TensorFlow DepthwiseConv2D 레이어의 'groups' 파라미터 충돌 해결
    
    Args:
        없음
        
    Returns:
        없음: TensorFlow 레이어 클래스를 직접 패치
    """
    # 패치 코드 구현...

def preprocess_image(img):
    """
    얼굴 이미지를 모델 입력에 맞게 전처리
    
    Args:
        img: 원본 얼굴 이미지 (OpenCV BGR 형식)
        
    Returns:
        numpy.ndarray: 전처리된 이미지 (크기 조정, 정규화, 차원 확장)
    """
    # 이미지 전처리 코드...

def put_text_with_background(img, text, position, font, font_scale, text_color, thickness):
    """
    반투명 배경이 있는 텍스트를 이미지에 표시
    
    Args:
        img: 대상 이미지
        text: 표시할 텍스트
        position: 텍스트 위치 (x, y)
        font: OpenCV 폰트 유형
        font_scale: 폰트 크기 배율
        text_color: 텍스트 색상 (BGR)
        thickness: 텍스트 두께
        
    Returns:
        없음: 입력 이미지를 직접 수정
    """
    # 텍스트 배경 렌더링 코드...
```

### test_model.py

```python
"""
모델 테스트 및 평가 스크립트
마스크 분류 모델의 정확도와 성능을 테스트합니다.

주요 기능:
1. 테스트 이미지에 대한 예측 수행
2. 정확도, 정밀도, 재현율 계산
3. 혼동 행렬 생성
4. 실시간 테스트 모드 지원
"""
```

## 핵심 기능

1. **실시간 마스크 착용 감지 시스템**
   - 딥러닝 기반 마스크 착용/미착용 분류 (TensorFlow/Keras 모델 활용)
   - 프레임 처리 파이프라인: `프레임 획득` → `얼굴 검출` → `이미지 전처리` → `분류 예측` → `결과 표시`
   - OpenCV 기반 얼굴 검출 (Haar Cascade Classifier)
   - 5프레임마다 얼굴 검출을 수행하여 성능 최적화
   - 얼굴 감지 안정화 알고리즘 (이전 프레임 결과 활용)

2. **고급 시각적 피드백 시스템**
   - 마스크 착용 상태에 따른 색상 코드 (녹색: 착용, 빨간색: 미착용)
   - 반투명 배경을 가진 텍스트 표시로 가독성 향상
   - 인식 결과 및 신뢰도 실시간 표시 (예: "마스크 착용: 98%")
   - 안정적인 결과 표시를 위한 프레임 간 데이터 유지
   - 화면 내 모든 감지된 얼굴에 대한 개별 상태 표시

3. **단계별 경고 시스템**
   - 마스크 미착용 시간 추적 시스템 (`violation_counter` 구현)
   - 반복 위반자에 대한 점진적 경고 메시지 강화
   - 경고 수준별 차별화된 시각적 피드백 (메시지 크기, 색상, 깜빡임)
   - 사용자별 마스크 착용 규정 준수 이력 관리
   - 설정 가능한 위반 임계값 (`VIOLATION_THRESHOLD`)

4. **TensorFlow 모델 호환성 자동 패치**
   - Teachable Machine 모델 로드 시 발생하는 'groups' 파라미터 충돌 자동 해결
   - DepthwiseConv2D 레이어의 `from_config` 메서드 패치
   - 오류 메시지 없이 모델 로드 성공 보장
   - 다양한 TensorFlow 버전과의 호환성 유지
   - 사용자 친화적인 오류 처리 및 폴백 메커니즘

5. **성능 최적화 기법**
   - 얼굴 검출 부하 감소를 위한 프레임 스킵 (5프레임마다)
   - 화면 표시용 프레임 크기 최적화 (640x480)
   - 모델 예측 시 verbose 모드 비활성화로 처리 속도 향상
   - TensorFlow 경고 메시지 필터링으로 시스템 성능 향상
   - 얼굴 검출 최소 크기 제한으로 오탐지 감소 (minSize 파라미터)

6. **사용자 설정 및 모드 지원**
   - 명령줄 인자를 통한 다양한 설정 옵션 제공
   - 조용한 모드 (`--quiet`) 지원으로 불필요한 출력 억제
   - 다양한 입력 이미지 크기 지원 (`--image_size`)
   - 사용자 정의 모델 및 라벨 경로 지정 가능
   - 다중 카메라 설정 지원 (`--camera` 옵션)

7. **확장 가능한 모듈식 설계**
   - 코드 재사용성을 높이는 함수 모듈화
   - 명확한 함수 인터페이스와 주석 처리
   - 추가 기능 구현을 위한 확장 포인트 제공
   - 코드 섹션별 명확한 구분과 설명
   - 예외 처리를 통한 안정적인 실행 보장

## 실행 방법

### 1. 필요한 라이브러리 설치

```bash
# 기본 필수 라이브러리 설치
pip install -r requirements.txt

# 또는 개별 패키지 설치
pip install tensorflow==2.4.0 opencv-python==4.5.0 numpy==1.19.0 matplotlib==3.3.0
```

### 2. 기본 실행 방법

```bash
# 프로젝트 루트 디렉토리에서
cd 01_mask_wear/src

# 기본 설정으로 실행
python mask_detection_app.py
```

### 3. 고급 실행 옵션

```bash
# 사용자 지정 모델 및 라벨 사용
python mask_detection_app.py --model ../models/converted_keras/keras_model.h5 --labels ../models/converted_keras/labels.txt

# 다른 카메라 장치 사용 (여러 카메라가 있는 경우)
python mask_detection_app.py --camera 1

# 이미지 크기 변경 (성능 vs 정확도 조절)
python mask_detection_app.py --image_size 96

# 조용한 모드로 실행 (로그 최소화)
python mask_detection_app.py --quiet
```

### 4. 명령줄 옵션 상세 설명

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--model` | 모델 파일 경로 | ../models/converted_keras/keras_model.h5 |
| `--labels` | 라벨 파일 경로 | ../models/converted_keras/labels.txt |
| `--camera` | 카메라 장치 번호 | 0 |
| `--image_size` | 입력 이미지 크기 | 224 |
| `--quiet` | 최소 출력 모드 | False |

### 5. 모델 테스트 및 평가

```bash
# 모델 성능 테스트 실행
python test_model.py --model ../models/converted_keras/keras_model.h5 --test_dir ../data/test
```

## 사용 방법

1. 프로그램이 실행되면 카메라가 활성화됩니다.
2. 카메라 앞에서 마스크를 착용하거나 미착용 상태로 있으세요.
3. 화면에 얼굴이 감지되면 주변에 색상 프레임이 표시됩니다:
   - **녹색**: 마스크 착용 상태 (올바름)
   - **빨간색**: 마스크 미착용 상태 (위반)
4. 마스크 미착용 상태가 지속되면 경고 메시지가 표시됩니다.
5. 'q' 키를 누르면 프로그램이 종료됩니다.

## 모델 학습 및 성능 향상

### 모델 재학습 방법

```bash
# 추가 데이터로 모델 재학습 (터미널에서)
cd src
python train_model.py --data_dir ../data --epochs 20 --batch_size 32
```

### 성능 향상 팁

1. **데이터 품질 개선**
   - 다양한 조명 환경에서 이미지 수집
   - 다양한 마스크 유형 및 착용 방법 포함
   - 얼굴 방향 및 각도 다양화

2. **모델 최적화**
   - 배치 정규화 레이어 추가
   - 드롭아웃 레이어로 과적합 방지
   - 더 큰 모델 아키텍처 사용 (MobileNetV2, EfficientNet)

3. **시스템 성능**
   - GPU 가속 활용 (CUDA, TensorRT)
   - 모델 양자화 (INT8 변환)
   - OpenCV DNN 모듈로 얼굴 검출 성능 향상

## 문제 해결

| 문제 | 원인 | 해결 방법 |
|------|------|----------|
| 카메라 접근 오류 | 다른 앱에서 카메라 사용 중 | 카메라 사용 앱 종료 또는 `--camera` 옵션으로 다른 카메라 선택 |
| 모델 로딩 오류 | 파일 경로 잘못됨 | 정확한 모델 및 라벨 경로 지정 또는 기본 위치에 파일 복사 |
| 얼굴 감지 안됨 | 조명 부족 또는 얼굴 가림 | 조명 개선 및 카메라 위치 조정 |
| 낮은 인식 정확도 | 모델 학습 데이터 부족 | 다양한 환경에서 추가 데이터 수집 후 재학습 |
| 느린 처리 속도 | 하드웨어 제약 | 이미지 크기 감소 (`--image_size` 값 줄임) 또는 더 강력한 하드웨어 사용 |

## 참고 문헌

1. OpenCV 얼굴 인식: https://docs.opencv.org/4.5.0/d7/d8b/tutorial_py_face_detection.html
2. TensorFlow 모델 최적화: https://www.tensorflow.org/lite/performance/model_optimization
3. Teachable Machine: https://teachablemachine.withgoogle.com/ 