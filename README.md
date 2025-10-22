# 반둑

추가 예정

## 규칙 설명

추가 예정

## 알파제로

추가 예정

### 개발 환경

- HW: Apple Silicon M3 Pro
- OS: macOS Sonoma 14.8.1
- Python: 3.10.17 (Anaconda)

### 주요 패키지 버전

- TensorFlow: 2.16.2 (Apple Silicon 최적화)
  - tensorflow-macos: 2.16.2
  - tensorflow-metal: 1.2.0
- NumPy: 1.26.4
- Keras: 3.10.0
- Ray: 2.46.0 (병렬 처리)
- h5py: 3.13.0
- Matplotlib: 3.10.3
- more-itertools: 10.7.0 (list에서 같은 ID를 가진 돌들의 위치를 찾는 기능)

### 설치 방법

1. Anaconda 설치 (만약 설치되어 있지 않다면)
2. Conda 환경 생성 및 활성화

```bash
conda create -n alpha_zero_env python=3.10
conda activate alpha_zero_env
```

3. 필요한 패키지 설치

```bash
pip install tensorflow tensorflow-macos tensorflow-metal
pip install numpy keras ray h5py matplotlib
```

### 실행 방법

1. AI가 흑돌로 게임 시작

```bash
python ai_first_play.py
```

2. 사람이 흑돌로 게임 시작

```bash
python human_first_play.py
```

### 모델 트레인 하는 법

```bash
python train_cycle.py
```
