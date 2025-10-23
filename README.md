# 반둑

## 규칙 설명

[반둑 규칙 보러가기](https://hanbat1mj.github.io/about-banduk)

## 알파제로

[알파제로를 분석하며 배우는 인공지능](https://github.com/Jpub/AlphaZero)에서 소스 코드를 일부 사용하였습니다.

> 개선된점
>
> - 병렬로 Self play와 evaluate를 처리합니다.
> - Self play에서 매판이 끝날 때마다 일정한 판이 찰 때 체크포인트 파일을 생성하여 모종의 이유(Colab 꺼짐 등)로 프로세스가 종료되었을 때 데이터 손실을 방지합니다.
> - 디리클레 노이즈를 통해, Self play 한정으로 첫 탐색의 다양함을 늘려 넓게 탐색을 할 수 있게 합니다.
> - 불필요할 정도로 큰 모델 사이즈를 줄여 학습 속도를 대폭 개선  
>   -> self play, evaluate 시간 대폭 감소  
>   => 맥북에서도 모델을 학습시킬 수 있습니다.
> - Self play에서 데이터를 모을 때 한 판당 8개의 데이터를 만듭니다. (90도 180도 270도 회전, 좌우 flip 후 회전 해서 총 8개)

## 알파제로 총 학습 시간

> Apple Silicon M3 Pro에서 5일 간 train_cycle.py 실행.
>
> 반둑 고중수 수준의 성능 달성!

## 알파제로와 사람의 대국 예시

![AI-vs-Human](/assets/AI_vs_Human.png)

- 흑: 인간 (@hanbat1mj)
- 백: AI

> 18집 : 18.5집으로 `인간 개발자 패`...
>
> - (0.5집은 백돌에게 주어지는 핸디캡.)

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
