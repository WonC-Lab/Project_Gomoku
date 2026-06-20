# 강화학습 및 실시간 수읽기 기반 오목 AI 시스템 개발
  정밀 Renju-Rule 구현 및 실시간 시뮬레이션 최적화


## 1. 프로젝트 개요 (Introduction)

* 목표: 13x13 바둑판 환경에서 정밀한 오목 렌주룰을 준수하는 강력한 실시간 강화학습 AI 개발
* 핵심 기술 스택:
  * 프레임워크: Pygame (GUI 렌더링 및 대국 시스템)
  * 백엔드: PyTorch & NumPy (딥러닝 모델 추론 및 강화학습)
* 3대 핵심 구현 요소:
  1. **정밀 Renju-Rule Engine:** 33, 44, 장목 금수 판정 및 상호 재귀적 활삼 판정
  2. **강화학습 (Self-play RL):** REINFORCE 정책망 모델 자가대국 학습
  3. **실시간 수읽기 (MCTS):** 정책망 예측과 가상 대국 시뮬레이션 결합


## 2. 오목 렌주룰 (Strict Renju Rules)

흑돌(선공)의 지나친 유리함을 억제하고 게임의 균형을 맞추기 위한 정밀 룰 구현:

* **33 금수 (Double-Three):** 돌을 놓았을 때 두 개 이상의 '열린 3(활삼)'이 동시에 만들어지는 자리
* **44 금수 (Double-Four):** 돌을 놓았을 때 두 개 이상의 '4'가 동시에 만들어지는 자리
* **장목 금수 (Overline):** 흑돌에 한해 6개 이상의 연속된 돌을 형성하는 자리
* **예외 규정 (승리 우선):** 만약 착수하는 순간 **정확히 5목**이 완성된다면, 금수 여부와 무관하게 착수가 허용되고 승리 처리됩니다.


## 3. 재귀적 쌍삼(Open Three) 판정 알고리즘

단순 형상(패턴) 분석을 넘어서는 렌주룰 고유의 수학적 재귀 판정:

* **열린 3의 정의:** 빈칸에 돌을 놓았을 때 양쪽 끝이 모두 열려 있는 '열린 4'가 될 수 있는 3개의 돌 배치
* **재귀적 금수 판정 (Recursive Check):**
  * 3을 4로 확장하는 빈칸이 **스스로 금수 자리(33/44)**라면, 그 자리는 돌을 둘 수 없으므로 '열린' 공간으로 인정되지 않습니다.
  * 따라서, 해당 3은 활삼으로 인정되지 않으며 33 금수 카운트에서 제외됩니다.
* **알고리즘 흐름:**
  `is_forbidden(r, c) -> is_open_three(r, c) -> is_forbidden(nr, nc) [Depth + 1]` (무한 루프를 방지하기 위해 최대 2단계까지 깊이 제한)

---

## 4. 강화학습 모델 아키텍처 (GomokuNet)

바둑판의 이미지적 특징을 효과적으로 포착하기 위해 **Convolutional Neural Network (CNN)** 설계:

* **입력 상태 (State Representation):** $3 \times 13 \times 13$ 텐서
  * **Channel 0:** 현재 내 돌의 위치 ($1.0$ 또는 $0.0$)
  * **Channel 1:** 상대방 돌의 위치 ($1.0$ 또는 $0.0$)
  * **Channel 2:** 빈 공간의 위치 ($1.0$ 또는 $0.0$)
* **신경망 아키텍처:**
  1. `Conv2d (Input 3, Output 64, Kernel 3x3, Padding 1) -> BatchNorm -> ReLU`
  2. `Conv2d (Input 64, Output 64, Kernel 3x3, Padding 1) -> BatchNorm -> ReLU`
  3. `Conv2d (Input 64, Output 64, Kernel 3x3, Padding 1) -> BatchNorm -> ReLU`
  4. `Policy Head: Conv2d (Input 64, Output 2, Kernel 1x1) -> Flatten -> Linear(169)`
* **출력:** 바둑판 전체 169개 칸의 가치(Logits)

---

## 5. 학습 메커니즘 및 보상 설계 (REINFORCE)

자가대국(Self-Play) 데이터를 활용한 경사 하강 훈련 구조:

* **기본 알고리즘:** REINFORCE (Monte Carlo Policy Gradient)
* **손실 함수 (Loss Function):**
  $$Loss = -\frac{1}{N}\sum_{t=1}^{N} \log \pi(a_t | s_t) \cdot G_t$$
  (여기서 $G_t$는 누적 할인 보상값, 할인율 $\gamma = 0.95$)
* **보상 설계 (Reward Shaping):**
  * **최종 보상:** 승리 시 $+2.0$, 패배 시 $-2.0$, 무승부 시 $-0.5$
  * **공격 가중치:** 4목을 완성하는 착수 시 즉시 보상 $+0.2$
  * **수비 가중치:** 상대방이 4목을 완성하려는 자리를 차단하면 즉시 보상 $+0.3$

---

## 6. 정책망 기반 실시간 수읽기 (MCTS/Rollout)

'어려움(Hard)' 난이도에서 바둑판의 거대한 경우의 수($10^{80}$)를 극복하기 위한 수읽기 알고리즘:

* **기본 배경:** 단일 신경망 추론은 직관적이지만 깊은 수읽기(장기적 예측)에 한계가 있음
* **하이브리드 MCTS 흐름:**
  1. **후보군 선별 (Policy Pruning):** 신경망 정책이 추천하는 가장 유망한 상위 5개의 수(Top 5 Candidates)만 골라 탐색 폭을 극소화합니다.
  2. **몬테카를로 시뮬레이션:** 5개의 후보 자리에 각각 돌을 두었다고 가정한 뒤, 복사된 독립 보드에서 가상 자가대국(Rollout)을 15회씩 진행합니다. (최대 40수 수읽기)
  3. **평가 및 착수:** 15회의 시뮬레이션 중 AI(백돌)의 평균 승률이 가장 높은 후보지를 선택해 실제 판에 착수합니다.
  4. **시각 피드백:** 연산하는 동안 화면에 `AI is reading ahead...` 문구를 표시하여 긴장감을 줍니다.

---

## 7. 시스템 실행 방법 및 사용 가이드

### 환경 및 의존성 라이브러리 설치
```bash
pip install numpy torch pygame
```

### 실행 방법
```bash
# 아나콘다 환경을 사용하는 경우 (권장)
/opt/anaconda3/bin/python3 Gomuku_Project.py

# 일반 환경에서 실행하는 경우
python3 Gomuku_Project.py
```

### 게임 메뉴 조작
* **1. Local 1 vs 1 Game:** 친구와 한 컴퓨터에서 번갈아 오목을 둡니다.
* **2. Player vs AI (RL):** AI를 상대로 대국합니다. (Easy / Medium / Hard 난이도 지원)
* **3. Train AI (Self-Play RL):** AI들이 스스로 오목을 두며 빠르게 실시간으로 성장하는 모습을 시각적으로 관전합니다. (`ESC` 키를 누르면 언제든지 가중치를 저장하고 종료합니다.)
* **4. Reset AI Model Weight:** 가중치 파일을 삭제하여 AI를 처음 상태로 완전히 포맷합니다.
