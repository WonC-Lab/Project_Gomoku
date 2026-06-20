# Gomoku (Omok) 13x13 AI System

A comprehensive Gomoku (Omok) application on a 13x13 board, incorporating Reinforcement Learning (Policy Network), GPU-accelerated Monte Carlo Tree Search (MCTS), Minimax Search, and strict Renju rules.

---

## Table of Contents
1. [English Guide](#english-guide)
   - [Overview](#overview)
   - [Features](#features)
   - [Python Scripts Breakdown](#python-scripts-breakdown)
   - [AI Architectures & Hyperparameters](#ai-architectures--hyperparameters)
   - [Renju Rule Implementation](#renju-rule-implementation)
   - [Installation & Execution](#installation--execution)
2. [Korean Guide (한글 가이드)](#korean-guide-한글-가이드)
   - [개요](#개요)
   - [주요 기능](#주요-기능)
   - [파이썬 스크립트 상세 설명](#파이썬-스크립트-상세-설명)
   - [AI 아키텍처 및 하이퍼파라미터](#ai-아키텍처-및-하이퍼파라미터)
   - [렌주 룰 구현 방식](#렌주-룰-구현-방식)
   - [설치 및 실행 방법](#설치-및-실행-방법)

---

# English Guide

## Overview
This project is an advanced Gomoku game engine featuring deep reinforcement learning and search-based AI players. Built using **PyTorch**, **Pygame**, and **NumPy**, it offers interactive matches against AI agents across different difficulties or standard local two-player modes under either casual Gomoku rules or strict Renju rules.

---

## Features
- **Flexible Rules**: Supports both general Gomoku rules and strict Renju rules (restricting Double-Three, Double-Four, and Overlines for Black).
- **Hybrid AI Architectures**: Integrates Neural Networks (Reinforcement Learning), Monte Carlo Tree Search (MCTS), and Minimax Tree Search with Alpha-Beta pruning.
- **Cross-Platform & Hardware Acceleration**: Specifically optimized for both Windows (CUDA/CPU) and macOS (MPS/CPU) environments.
- **Multi-threaded AI Processing**: Moves heavy AI computation (MCTS/Minimax) to a background thread to prevent UI freezing during calculations.

---

## Python Scripts Breakdown

### 1. `Gomuku_Project_GPU_Windows.py`
The flagship, Windows-specific graphical interface edition of the project.
- **GUI Engine**: Pygame-based rendering with menu modes, rules selection, and difficulty selection.
- **Multithreading**: Employs a Python `threading.Thread` mechanism to compute AI moves, displaying a "Thinking..." animation to keep the window responsive.
- **Four Difficulty Settings**:
  1. **Easy**: Standard RL policy network (32 filters, 3 layers) with randomized actions ($\epsilon = 0.5$).
  2. **Medium**: RL policy network (64 filters, 3 layers) with less randomized actions ($\epsilon = 0.2$).
  3. **Hard**: High-capacity RL policy network (128 filters, 4 layers) backed by **GPU-accelerated batch vectorized MCTS** (50 rollouts per turn).
  4. **SuperHard**: Minimax search engine with Alpha-Beta pruning exploring up to depth 4, evaluating board conditions with a hand-tuned heuristic score table. Does not require training weights.
- **Path Management**: Safely resolves Windows file paths using `pathlib.Path`.

### 2. `Gomuku_Project_GPU.py`
The macOS/Linux-optimized graphical edition.
- **Hardware Acceleration**: Automatically detects and registers macOS MPS (Metal Performance Shaders) or CUDA devices.
- **Vectorized MCTS**: Performs high-speed rollouts utilizing PyTorch tensor operations on the GPU to speed up Monte Carlo simulations.
- Features similar game modes and rules as the Windows edition, tailored to Unix-style file path systems.

### 3. `Gomuku_Project.py`
The standard/base CPU edition of the game.
- **CPU MCTS**: Performs classic single-instance simulation rollouts.
- Ideal for devices lacking discrete GPU units or environments where custom PyTorch installation is limited.

### 4. `pretrain_ai.py`
The offline command-line training tool.
- **Self-Play Reinforcement Learning**: Trains the policy network using the REINFORCE algorithm via self-play matches.
- **Difficulty Configurations**: Automates configuration loops to train separate networks (`gomoku_rl_easy.pth`, `gomoku_rl_medium.pth`, `gomoku_rl_hard.pth`) with customized network shapes (filters and layers) and exploration rates.
- **CLI Progress**: Renders clean textual progress bars, loss logs, and win-loss statistics.

---

## AI Architectures & Hyperparameters

The neural network (`GomokuNet`) consists of a common CNN backbone and a policy head outputting logits over the 13x13 grid:

```
[Input: 3x13x13 Tensor] -> [Conv2D Layers + BatchNorm + ReLU] -> [Policy Head (Conv1D + Flatten + Linear)] -> [Logits: 169]
```
*Note: The input tensor represents 3 channels: (1) current player's stones, (2) opponent's stones, and (3) empty spots.*

### Hyperparameter Profiles (Training & Architecture)

| Difficulty | Conv Layers | Conv Filters | Learning Rate | Exploration ($\epsilon$) | Model File |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Easy** | 3 | 32 | $1 \times 10^{-3}$ | 0.40 / 0.50 | `gomoku_rl_easy.pth` |
| **Medium** | 3 | 64 | $5 \times 10^{-4}$ | 0.25 / 0.20 | `gomoku_rl_medium.pth` |
| **Hard** | 4 | 128 | $3 \times 10^{-4}$ | 0.15 | `gomoku_rl_hard.pth` |
| **SuperHard** | *N/A (Minimax)* | *N/A* | *N/A* | 0.00 | *Rule-based / Heuristic* |

---

## Renju Rule Implementation
The Renju rules restrict Black (the starting player) from placing stones on forbidden spots to counter the first-player advantage:
1. **Overline (장목)**: Creating a continuous line of 6 or more stones.
2. **Double-Four (44)**: Simultaneously creating two or more "Fours" (lines where placing one more stone yields 5).
3. **Double-Three (33)**: Simultaneously creating two or more "Open Threes" (lines that can be extended to an open four).

### Detection Algorithm (`is_forbidden`)
- The algorithm temporarily places a black stone on candidate coordinate `(r, c)`.
- **Five wins**: If the move immediately yields a line of exactly 5, it is **not forbidden** (wins override restrictions).
- **Overline checking**: It counts consecutive stones in 4 directions (`DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]`).
- **Open-Three checking**: Recursively evaluates whether empty adjacent spaces can lead to an active winning line, pruning out branch cases where further extensions are blocked.

---

## Installation & Execution

### Prerequisites
Make sure Python 3.8+ is installed on your system. Install the required dependencies using pip:
```bash
pip install torch torchvision numpy pygame
```

### Running the Game
- **On Windows**:
  ```bash
  python Gomuku_Project_GPU_Windows.py
  ```
- **On macOS/Linux (GPU Accelerated)**:
  ```bash
  python Gomuku_Project_GPU.py
  ```
- **On Standard CPU environments**:
  ```bash
  python Gomuku_Project.py
  ```

### Training the Models Offline
If you want to retrain the reinforcement learning models from scratch:
```bash
python pretrain_ai.py
```

---

# Korean Guide (한글 가이드)

## 개요
이 프로젝트는 딥러닝 강화학습과 트리 탐색 알고리즘을 결합한 13x13 오목 게임 엔진입니다. **PyTorch**, **Pygame**, **NumPy**를 사용하여 작성되었으며, 일반 오목 규칙 또는 엄격한 렌주 룰(무르기 및 흑 삼삼/사사 금수 적용) 하에 AI와의 대국 및 로컬 2인용 대국을 진행할 수 있습니다.

---

## 주요 기능
- **다양한 규칙 설정**: 일반 오목 룰 및 엄격한 렌주 룰(흑돌의 삼삼, 사사, 장목 금수 제한)을 완벽 지원합니다.
- **하이브리드 AI 구조**: 정책 신경망(강화학습), 몬테카를로 트리 탐색(MCTS), 알파-베타 가지치기가 적용된 미니맥스 탐색 엔진을 결합했습니다.
- **하드웨어 가속 지원**: Windows(CUDA/CPU)와 macOS(MPS/CPU) 환경에 각각 최적화된 하드웨어 가속 코드를 탑재했습니다.
- **멀티스레드 AI 연산**: 무거운 AI 수읽기(MCTS 및 미니맥스)를 백그라운드 스레드에서 수행하여 연산 중에 게임 화면이 "응답 없음" 상태로 멈추는 현상을 방지합니다.

---

## 파이썬 스크립트 상세 설명

### 1. `Gomuku_Project_GPU_Windows.py`
Windows 운영체제에 특화된 GUI 에디션입니다.
- **화면 구성**: Pygame을 사용해 대국 메뉴, 규칙 설정 및 난이도 선택 화면을 구현했습니다.
- **멀티스레딩**: 파이썬의 `threading.Thread`를 통해 AI 연산을 분리하여 화면에 "생각 중..." 애니메이션을 띄우고 창 조작이 유지되도록 처리합니다.
- **4단계 난이도 구성**:
  1. **Easy**: 기본 정책 신경망(32채널, 3층) 구조에 무작위 행동 확률($\epsilon = 0.5$)을 높게 책정했습니다.
  2. **Medium**: 중간 크기 정책 신경망(64채널, 3층) 구조와 제한된 무작위 탐색($\epsilon = 0.2$)을 사용합니다.
  3. **Hard**: 고성능 정책 신경망(128채널, 4층) 및 **GPU 가속 배치 벡터화 MCTS**(50 롤아웃) 알고리즘을 적용하여 한층 더 날카로운 수읽기를 수행합니다.
  4. **SuperHard**: 딥러닝 가중치 없이 작동하는 미니맥스(Minimax) 엔진으로, 최대 4수 깊이까지 탐색하며 자체 휴리스틱 평가 점수표를 기반으로 동작하는 고성능 탐색 AI입니다.
- **경로 처리**: `pathlib.Path`를 사용하여 Windows의 폴더 경로 및 역슬래시(`\`) 시스템을 안전하게 지원합니다.

### 2. `Gomuku_Project_GPU.py`
macOS 및 Linux 환경에 최적화된 GPU 가속 버전입니다.
- **가속 자동 선택**: 애플 실리콘 Mac의 MPS(Metal Performance Shaders) 및 일반 그래픽 카드의 CUDA 디바이스를 자동 감지합니다.
- **벡터화 MCTS**: 시뮬레이션을 돌릴 때 PyTorch 텐서 연산을 통해 GPU에서 여러 판을 한 번에 병렬 연산하여 MCTS 효율을 극대화합니다.

### 3. `Gomuku_Project.py`
CPU 연산에 기반한 표준 오목 프로그램입니다.
- 단일 스레드 형태의 CPU MCTS 연산을 수행하며, 외장 그래픽 카드가 없거나 PyTorch GPU 드라이버 설정이 곤란한 환경에서 사용하기에 적합합니다.

### 4. `pretrain_ai.py`
GUI(Pygame) 없이 터미널 콘솔창에서 실행되는 오프라인 사전 학습 스크립트입니다.
- **자가 대국(Self-Play) 강화학습**: REINFORCE 알고리즘을 사용해 AI끼리 대국을 치르며 스스로 승리 패턴을 습득하게 합니다.
- **자동 학습 제어**: 설정된 난이도(Easy, Medium, Hard)별 신경망 두께와 학습률을 조절하며 `gomoku_rl_easy.pth` 등의 모델 파일을 생성합니다.
- **학습 정보 시각화**: 에피소드 진행 상황, 손실 함수(Loss), 통계치를 터미널에 텍스트 프로그레스 바 형태로 깔끔하게 표시합니다.

---

## AI 아키텍처 및 하이퍼파라미터

신경망 모델(`GomokuNet`)은 CNN 백본과 착수 확률을 도출하는 Policy Head로 이루어져 있습니다:

```
[입력: 3x13x13 텐서] -> [합성곱(Conv2D) 층 + 배치정규화 + ReLU] -> [Policy Head (Conv1D + Flatten + 선형 층)] -> [출력: 169개 착수점 확률]
```
*참고: 입력 텐서는 (1) 내 돌의 위치, (2) 상대 돌의 위치, (3) 빈 칸의 정보를 3개의 채널로 전달합니다.*

### 하이퍼파라미터 요약

| 난이도 | 합성곱 층 수 | 채널 수 (Filters) | 학습률 (Learning Rate) | 탐색 무작위성 ($\epsilon$) | 가중치 파일명 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Easy** | 3 | 32 | $1 \times 10^{-3}$ | 0.40 (학습) / 0.50 (게임) | `gomoku_rl_easy.pth` |
| **Medium** | 3 | 64 | $5 \times 10^{-4}$ | 0.25 (학습) / 0.20 (게임) | `gomoku_rl_medium.pth` |
| **Hard** | 4 | 128 | $3 \times 10^{-4}$ | 0.15 (학습) / 0.00 (MCTS) | `gomoku_rl_hard.pth` |
| **SuperHard** | *미적용 (미니맥스)* | *미적용* | *미적용* | 0.00 | *점수 룰 기반 탐색* |

---

## 렌주 룰 구현 방식
오목판에서 흑의 선공 유리함을 상쇄하기 위한 렌주 룰의 금수를 알고리즘으로 판정합니다.
1. **장목**: 흑돌이 연속해서 6목 이상을 형성하는 수.
2. **쌍사(44)**: 돌을 놓았을 때 4목이 두 방향 이상 동시에 만들어지는 수.
3. **쌍삼(33)**: 돌을 놓았을 때 3목(양쪽이 열려 4목으로 발전 가능한 3목)이 두 방향 이상 동시에 만들어지는 수.

### 금수 탐색 함수 (`is_forbidden`)
- 좌표 `(r, c)`에 흑돌을 두었다고 가정한 뒤, 가로/세로/대각선 4가지 방향(`DIRECTIONS`)을 탐색합니다.
- **5목 우선원칙**: 금수처럼 보이는 자리라도 정확히 5목이 완성되어 게임이 종료되는 수라면 **금수에서 제외(허용)**합니다.
- **쌍삼 판정**: 3목이 놓인 빈 공간이 다시 금수인지 아닌지를 재귀적으로 판별하여 양끝이 막히지 않고 실제로 4목으로 연장 가능한 '열린 3'의 개수를 세어 판정합니다.

---

## 설치 및 실행 방법

### 의존성 설치
컴퓨터에 Python 3.8 이상이 설치되어 있는지 확인하고, 다음 명령어를 실행하여 필수 패키지를 설치합니다:
```bash
pip install torch torchvision numpy pygame
```

### 대국 실행
- **Windows 컴퓨터**:
  ```bash
  python Gomuku_Project_GPU_Windows.py
  ```
- **macOS / Linux 컴퓨터**:
  ```bash
  python Gomuku_Project_GPU.py
  ```
- **표준 CPU 환경 및 호환성 모드**:
  ```bash
  python Gomuku_Project.py
  ```

### 오프라인 AI 사전 학습 진행
기존에 학습된 AI 모델 가중치 파일들을 새로 훈련하고 싶다면 아래 스크립트를 실행합니다:
```bash
python pretrain_ai.py
```
