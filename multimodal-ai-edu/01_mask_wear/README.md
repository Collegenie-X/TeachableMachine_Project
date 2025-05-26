# 마스크 착용 인식 프로젝트

실시간 카메라를 통해 마스크 착용 여부를 감지하고, 올바른 마스크 착용 방법을 안내하는 시스템입니다.

## 구성 요소

- **데이터셋**: `data/` 디렉토리에 마스크 착용/미착용 이미지 데이터셋
- **모델 훈련**: `notebooks/` 디렉토리의 주피터 노트북으로 모델 훈련 및 테스트
- **모델 파일**: `models/` 디렉토리에 훈련된 모델 파일 저장
- **애플리케이션**: `src/` 디렉토리의 실시간 마스크 착용 감지 시스템

## 주요 기능

- **실시간 마스크 착용 감지**: 웹캠을 통해 실시간으로 마스크 착용 여부 감지
- **경고 시스템**: 마스크 미착용 시 경고 메시지 표시
- **상습 위반자 감지**: 반복적으로 마스크를 착용하지 않는 사용자 식별
- **다양한 환경 지원**: Google Colab과 로컬 Jupyter 환경 모두 지원

## 사용 방법

### 1. Jupyter Notebook으로 실행 (권장)

이 프로젝트는 Jupyter Notebook 환경에서 쉽게 실행할 수 있도록 설계되었습니다.

1. Jupyter Notebook 설치 (아직 설치하지 않은 경우)
   ```bash
   pip install jupyter
   ```

2. 노트북 실행
   ```bash
   cd notebooks
   jupyter notebook
   ```

3. 다음 노트북 파일 중 하나를 선택하여 실행:
   - `teachable_machine_uploader.ipynb`: Teachable Machine 모델 업로드 및 테스트
   - `mask_detection_app.ipynb`: 마스크 감지 애플리케이션 실행
   - `train_model.ipynb`: 새 모델 훈련 (선택적)

자세한 지침은 [notebooks/README.md](./notebooks/README.md)를 참조하세요.

### 2. 명령줄에서 실행

터미널 또는 명령 프롬프트에서 다음 명령을 실행:

```bash
cd src
python mask_detection_app.py --model ../models/model.h5 --labels ../models/labels.txt
```

옵션:
- `--model`: 모델 파일 경로 (기본값: ../models/model.h5)
- `--labels`: 라벨 파일 경로 (기본값: ../models/labels.txt)
- `--camera`: 카메라 장치 번호 (기본값: 0)
- `--image_size`: 입력 이미지 크기 (기본값: 96)
- `--from_drive`: Google Drive에서 모델 로드 (Colab 환경에서만)

### 3. Teachable Machine 모델 사용

1. [Teachable Machine](https://teachablemachine.withgoogle.com/)에서 이미지 프로젝트 생성
2. 마스크 착용/미착용 이미지로 모델 훈련
3. 모델 내보내기 (TensorFlow 형식 선택)
4. 다운로드한 모델 파일(.h5)과 라벨 파일(.txt)을 `models/` 디렉토리에 저장
5. 위의 방법 중 하나로 애플리케이션 실행

## 필요한 라이브러리

```bash
pip install tensorflow opencv-python matplotlib numpy
```

## 시스템 구조

```
AI 기반 마스크 착용 감지
│
├─ 데이터 입력
│   └─ 실시간 카메라 영상
│
├─ 전처리
│   └─ 이미지 크기 변환 (96x96), 정규화
│
├─ 예측
│   └─ Teachable Machine 기반 CNN 모델(.h5)
│        └─ 마스크 착용/미착용 분류
│
├─ 결과 처리
│   ├─ 라벨 및 확률 출력
│   ├─ 경고 메시지 및 상습 위반자 감지
│   └─ UI(화면)에 실시간 시각화
│
└─ 사용자 인터랙션
    └─ 'q' 입력 시 종료
``` 