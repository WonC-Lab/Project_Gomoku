"""
Omok 13x13 AI System - Windows Edition
=======================================
Mac 버전(Gomuku_Project_GPU.py)과 별도로 운영되는 Windows 전용 버전.

난이도:
  1. Easy      - RL (32채널, 3층, epsilon=0.5)
  2. Medium    - RL (64채널, 3층, epsilon=0.2)
  3. Hard      - RL + GPU MCTS (128채널, 4층, 50롤아웃)
  4. SuperHard - Minimax + Alpha-Beta Pruning (4수 탐색, 학습 불필요)

모델 저장: 스크립트 폴더 내 models/ 디렉토리 (Windows 경로 자동 처리)
"""

import pygame
import sys
import pathlib
import threading

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import os
except ImportError as e:
    pygame.init()
    screen = pygame.display.set_mode((620, 600))
    pygame.display.set_caption("Error - Missing Dependencies")
    font = pygame.font.SysFont("Arial", 18)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill((255, 255, 255))
        lines = [
            "Import Error: Missing dependencies!",
            f"Error: {e}",
            "",
            "Please install via pip:",
            "  pip install torch torchvision numpy pygame",
            "",
            "Close this window to exit."
        ]
        y = 140
        for line in lines:
            color = (200, 0, 0) if "Error" in line else (0, 0, 0)
            surf = font.render(line, True, color)
            rect = surf.get_rect(center=(310, y))
            screen.blit(surf, rect)
            y += 45
        pygame.display.flip()
    pygame.quit()
    sys.exit()

# ============================================================
# --- Windows 경로 설정 ---
# ============================================================
pygame.init()

BASE_DIR = pathlib.Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
print(f"[Windows] Model directory: {MODELS_DIR}")

# ============================================================
# --- 초기 설정 ---
# ============================================================
SCREEN_SIZE = 600
GRID_SIZE   = 13
CELL_SIZE   = SCREEN_SIZE // (GRID_SIZE + 1)

screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption("Omok 13x13 AI [Windows Edition]")
font = pygame.font.SysFont("Arial", 26)

WHITE      = (255, 255, 255)
BLACK      = (0,   0,   0)
BOARD_COLOR = (240, 217, 181)

# 최소 권장 학습 에피소드 수
MIN_TRAIN_EPISODES = 100

# ============================================================
# --- 게임 상태 변수 ---
# ============================================================
state                  = "MENU_MODE"
mode                   = None          # "2P" | "AI" | "TRAIN"
rule_open3             = None          # True: 33허용, False: 33금지(렌주)
difficulty             = None          # "Easy" | "Medium" | "Hard" | "SuperHard"
current_train_difficulty = None        # 현재 학습 중인 난이도
forbidden_msg          = None
forbidden_time         = 0
board                  = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
turn                   = 1             # 1: 흑, 2: 백

# AI 스레드 상태 (응답없음 방지용)
ai_thinking     = False   # AI가 계산 중인지 여부
ai_move_result  = None    # 스레드가 채운 착수 결과
ai_thread       = None    # 스레드 객체
thinking_dots   = 0       # "생각 중..." 애니메이션 카운터
thinking_timer  = 0       # 돇 업데이트 타이머

DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]

# ============================================================
# --- 보조 함수 ---
# ============================================================

def in_bounds(r, c):
    return 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE

def draw_text(text, y, color=BLACK):
    surf = font.render(text, True, color)
    rect = surf.get_rect(center=(SCREEN_SIZE / 2, y))
    screen.blit(surf, rect)

def check_overline(r, c, color):
    """6목 이상 확인 (흑만 해당)."""
    if color != 1:
        return False
    for dr, dc in DIRECTIONS:
        count = 1
        for i in [1, -1]:
            nr, nc = r + dr * i, c + dc * i
            while in_bounds(nr, nc) and board[nr][nc] == color:
                count += 1; nr += dr * i; nc += dc * i
        if count >= 6:
            return True
    return False

def is_five(r, c, color):
    """정확히 5목(흑) 또는 5목 이상(백) 확인."""
    for dr, dc in DIRECTIONS:
        count = 1
        for i in [1, -1]:
            nr, nc = r + dr * i, c + dc * i
            while in_bounds(nr, nc) and board[nr][nc] == color:
                count += 1; nr += dr * i; nc += dc * i
        if color == 1:
            if count == 5: return True
        else:
            if count >= 5: return True
    return False

def is_five_in_dir(r, c, color, dr, dc):
    count = 1
    for i in [1, -1]:
        nr, nc = r + dr * i, c + dc * i
        while in_bounds(nr, nc) and board[nr][nc] == color:
            count += 1; nr += dr * i; nc += dc * i
    return count == 5 if color == 1 else count >= 5

def count_fours_created(r, c, color):
    """해당 위치에 놓았을 때 생성되는 4목 방향 수."""
    fours_count = 0
    board[r][c] = color
    for dr, dc in DIRECTIONS:
        has_four = False
        for k in range(-4, 5):
            if k == 0: continue
            nr, nc = r + dr * k, c + dc * k
            if in_bounds(nr, nc) and board[nr][nc] == 0:
                board[nr][nc] = color
                if is_five_in_dir(nr, nc, color, dr, dc):
                    has_four = True
                board[nr][nc] = 0
                if has_four: break
        if has_four: fours_count += 1
    board[r][c] = 0
    return fours_count

def is_open_three_in_dir(r, c, color, dr, dc, depth):
    for k in range(-4, 5):
        if k == 0: continue
        nr, nc = r + dr * k, c + dc * k
        if in_bounds(nr, nc) and board[nr][nc] == 0:
            if depth < 2:
                if is_forbidden(nr, nc, color, depth + 1):
                    continue
            board[nr][nc] = color
            winning_spots = 0
            for m in range(-4, 5):
                if m == 0 or m == k: continue
                wr, wc = r + dr * m, c + dc * m
                if in_bounds(wr, wc) and board[wr][wc] == 0:
                    board[wr][wc] = color
                    if is_five_in_dir(wr, wc, color, dr, dc):
                        winning_spots += 1
                    board[wr][wc] = 0
            board[nr][nc] = 0
            if winning_spots >= 2:
                return True
    return False

def is_forbidden(r, c, color, depth=0):
    """렌주 금수(흑) 판정: 장목, 쌍4, 쌍3."""
    if color != 1: return False
    if board[r][c] != 0: return False
    board[r][c] = color
    five = is_five(r, c, color)
    board[r][c] = 0
    if five: return False            # 5목은 금수 아님
    board[r][c] = color
    is_ov = check_overline(r, c, color)
    board[r][c] = 0
    if is_ov: return True            # 장목
    fours = count_fours_created(r, c, color)
    if fours >= 2: return True       # 쌍4
    board[r][c] = color
    open_threes = sum(
        1 for dr, dc in DIRECTIONS
        if is_open_three_in_dir(r, c, color, dr, dc, depth)
    )
    board[r][c] = 0
    if open_threes >= 2: return True # 쌍3
    return False

def check_win_for_color(r, c, color):
    """해당 위치에 color 돌을 두었을 때 승리 여부."""
    for dr, dc in DIRECTIONS:
        count = 1
        for i in [1, -1]:
            nr, nc = r + dr * i, c + dc * i
            while in_bounds(nr, nc) and board[nr][nc] == color:
                count += 1; nr += dr * i; nc += dc * i
        if color == 1 and not rule_open3:
            if count == 5: return True
        else:
            if count >= 5: return True
    return False

def draw_board():
    screen.fill(BOARD_COLOR)
    for i in range(GRID_SIZE):
        pygame.draw.line(screen, BLACK,
                         (CELL_SIZE,            CELL_SIZE * (i + 1)),
                         (CELL_SIZE * GRID_SIZE, CELL_SIZE * (i + 1)))
        pygame.draw.line(screen, BLACK,
                         (CELL_SIZE * (i + 1), CELL_SIZE),
                         (CELL_SIZE * (i + 1), CELL_SIZE * GRID_SIZE))
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            cx = CELL_SIZE * (c + 1)
            cy = CELL_SIZE * (r + 1)
            if board[r][c] == 1:
                pygame.draw.circle(screen, BLACK, (cx, cy), CELL_SIZE // 2 - 2)
            elif board[r][c] == 2:
                pygame.draw.circle(screen, WHITE, (cx, cy), CELL_SIZE // 2 - 2)
                pygame.draw.circle(screen, BLACK, (cx, cy), CELL_SIZE // 2 - 2, 1)


# ============================================================
# --- 강화학습 신경망 (난이도별 크기 조정 가능) ---
# ============================================================

class GomokuNet(nn.Module):
    """
    파라미터화된 Gomoku 정책 신경망.
    - num_filters : 채널 수 (Easy=32, Medium=64, Hard=128)
    - num_layers  : 합성곱 층 수 (Easy/Medium=3, Hard=4)
    """
    def __init__(self, board_size=13, num_filters=64, num_layers=3):
        super().__init__()
        self.board_size = board_size

        layers = [
            nn.Conv2d(3, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        ]
        for _ in range(num_layers - 1):
            layers += [
                nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_filters),
                nn.ReLU()
            ]
        self.conv = nn.Sequential(*layers)

        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, board_size * board_size)
        )

    def forward(self, x):
        return self.policy_head(self.conv(x))


def board_to_tensor(board_state, current_turn):
    board_arr = np.array(board_state)
    opponent  = 3 - current_turn
    ch0 = (board_arr == current_turn).astype(np.float32)
    ch1 = (board_arr == opponent).astype(np.float32)
    ch2 = (board_arr == 0).astype(np.float32)
    return torch.tensor(np.stack([ch0, ch1, ch2], axis=0)).unsqueeze(0)


# ============================================================
# --- 즉시 승리/방어 탐색 (모든 난이도 공통) ---
# ============================================================

def find_immediate_win_or_block(board_state, current_turn, open3_rule):
    opponent = 3 - current_turn

    # 1. AI 즉시 5목
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if board_state[r][c] == 0:
                board_state[r][c] = current_turn
                win = check_win_for_color(r, c, current_turn)
                board_state[r][c] = 0
                if win: return (r, c)

    # 2. 상대방 즉시 5목 차단
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if board_state[r][c] == 0:
                board_state[r][c] = opponent
                win = check_win_for_color(r, c, opponent)
                board_state[r][c] = 0
                if win:
                    if current_turn == 2 and not open3_rule and is_forbidden(r, c, 1):
                        continue
                    return (r, c)

    # 3. AI 양수(Open Four) 공격
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if board_state[r][c] == 0:
                if current_turn == 1 and not open3_rule and is_forbidden(r, c, 1):
                    continue
                board_state[r][c] = current_turn
                is_op4 = False
                for dr, dc in DIRECTIONS:
                    winning_spots = 0
                    for k in range(-4, 5):
                        nr, nc = r + dr * k, c + dc * k
                        if in_bounds(nr, nc) and board_state[nr][nc] == 0:
                            board_state[nr][nc] = current_turn
                            if is_five_in_dir(nr, nc, current_turn, dr, dc):
                                winning_spots += 1
                            board_state[nr][nc] = 0
                    if winning_spots >= 2: is_op4 = True; break
                board_state[r][c] = 0
                if is_op4: return (r, c)

    # 4. 상대방 양수(Open Four) 수비
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if board_state[r][c] == 0:
                board_state[r][c] = opponent
                is_op4 = False
                for dr, dc in DIRECTIONS:
                    winning_spots = 0
                    for k in range(-4, 5):
                        nr, nc = r + dr * k, c + dc * k
                        if in_bounds(nr, nc) and board_state[nr][nc] == 0:
                            board_state[nr][nc] = opponent
                            if is_five_in_dir(nr, nc, opponent, dr, dc):
                                winning_spots += 1
                            board_state[nr][nc] = 0
                    if winning_spots >= 2: is_op4 = True; break
                board_state[r][c] = 0
                if is_op4:
                    if current_turn == 2 and not open3_rule and is_forbidden(r, c, 1):
                        continue
                    return (r, c)

    return None


# ============================================================
# --- GPU 배치 벡터화 MCTS (Hard 전용) ---
# ============================================================

def get_action_mcts_vectorized(agent, board_state, current_turn, open3_rule,
                                num_rollouts=50, top_k=5):
    win_or_block = find_immediate_win_or_block(board_state, current_turn, open3_rule)
    if win_or_block is not None:
        return win_or_block

    valid_moves = []
    mask = np.zeros(agent.board_size * agent.board_size, dtype=np.float32)
    for r in range(agent.board_size):
        for c in range(agent.board_size):
            idx = r * agent.board_size + c
            if board_state[r][c] == 0:
                if not open3_rule and current_turn == 1 and is_forbidden(r, c, 1):
                    continue
                valid_moves.append((r, c))
                mask[idx] = 1.0

    if not valid_moves: return None
    if len(valid_moves) == 1: return valid_moves[0]

    agent.model.eval()
    state_tensor = board_to_tensor(board_state, current_turn).to(agent.device)
    with torch.no_grad():
        logits = agent.model(state_tensor).squeeze(0).cpu().numpy()

    logits[mask == 0] = -1e9
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)

    candidates = []
    for idx in np.argsort(probs)[::-1]:
        if mask[idx] > 0:
            candidates.append((idx // agent.board_size, idx % agent.board_size))
            if len(candidates) >= top_k: break

    if not candidates: return valid_moves[0]

    num_cand   = len(candidates)
    batch_size = num_cand * num_rollouts

    boards_batch  = np.array([np.array(board_state) for _ in range(batch_size)])
    turns_batch   = np.full(batch_size, 3 - current_turn)
    active_mask   = np.ones(batch_size, dtype=bool)
    winners_batch = np.zeros(batch_size, dtype=int)

    for i, (cr, cc) in enumerate(candidates):
        for j in range(num_rollouts):
            boards_batch[i * num_rollouts + j][cr][cc] = current_turn

    def sim_in_bounds(r, c):
        return 0 <= r < agent.board_size and 0 <= c < agent.board_size

    def sim_check_win(sim_board, r, c, color):
        for dr, dc in DIRECTIONS:
            cnt = 1
            for i in [1, -1]:
                nr, nc = r + dr * i, c + dc * i
                while sim_in_bounds(nr, nc) and sim_board[nr][nc] == color:
                    cnt += 1; nr += dr * i; nc += dc * i
            if color == 1 and not open3_rule:
                if cnt == 5: return True
            else:
                if cnt >= 5: return True
        return False

    for sim_step in range(60):
        active_indices = np.where(active_mask)[0]
        if len(active_indices) == 0: break

        state_list = []
        for idx in active_indices:
            b = boards_batch[idx]
            op = 3 - turns_batch[idx]
            state_list.append(np.stack([
                (b == turns_batch[idx]).astype(np.float32),
                (b == op).astype(np.float32),
                (b == 0).astype(np.float32)
            ], axis=0))

        batch_tensor = torch.tensor(np.array(state_list)).to(agent.device)
        agent.model.eval()
        with torch.no_grad():
            batch_logits = agent.model(batch_tensor).cpu().numpy()

        for step_idx, idx in enumerate(active_indices):
            b        = boards_batch[idx]
            sim_turn = turns_batch[idx]
            valid_idx = np.where(b.flatten() == 0)[0]
            if len(valid_idx) == 0:
                active_mask[idx] = False; continue

            lg = batch_logits[step_idx].copy()
            if np.random.rand() < 0.15:
                chosen = np.random.choice(valid_idx)
            else:
                m = np.zeros(GRID_SIZE * GRID_SIZE, dtype=np.float32)
                m[valid_idx] = 1.0
                lg[m == 0] = -1e9
                ex = np.exp(lg - np.max(lg))
                chosen = np.random.choice(GRID_SIZE * GRID_SIZE, p=ex / ex.sum())

            sr, sc = chosen // agent.board_size, chosen % agent.board_size
            boards_batch[idx][sr][sc] = sim_turn
            if sim_check_win(boards_batch[idx], sr, sc, sim_turn):
                winners_batch[idx] = sim_turn; active_mask[idx] = False
            else:
                turns_batch[idx] = 3 - sim_turn

    best_move  = candidates[0]
    best_score = -1.0
    opponent   = 3 - current_turn

    print("\n--- Hard AI (GPU MCTS) Candidates ---")
    for i, cand in enumerate(candidates):
        s = i * num_rollouts
        cw = winners_batch[s:s + num_rollouts]
        wins = int(np.sum(cw == current_turn))
        losses = int(np.sum(cw == opponent))
        draws  = int(np.sum(cw == 0))
        score  = wins + 0.2 * draws
        print(f"  {cand} | Win:{wins/num_rollouts*100:.1f}% W:{wins} L:{losses} D:{draws}")
        if score > best_score:
            best_score = score; best_move = cand

    print(f"  Selected: {best_move}\n")
    return best_move


# ============================================================
# --- RL 에이전트 (파라미터화, Windows 경로) ---
# ============================================================

class RLAgent:
    """
    강화학습(REINFORCE) 에이전트.
    - model_filename : models/ 폴더 내 저장 파일명
    - num_filters    : 합성곱 채널 수
    - num_layers     : 합성곱 층 수
    - lr             : 학습률
    """
    def __init__(self, board_size=13, lr=0.0005,
                 model_filename="gomoku_rl_model.pth",
                 num_filters=64, num_layers=3):
        self.board_size = board_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_filters = num_filters
        self.num_layers  = num_layers
        self.lr          = lr
        self.model = GomokuNet(board_size, num_filters, num_layers).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.saved_model_path = str(MODELS_DIR / model_filename)

        if os.path.exists(self.saved_model_path):
            try:
                self.model.load_state_dict(
                    torch.load(self.saved_model_path,
                               map_location=self.device,
                               weights_only=True))
                print(f"  Loaded [{model_filename}] on {self.device}")
            except Exception as ex:
                print(f"  Error loading [{model_filename}]: {ex}")

    def get_action(self, board_state, current_turn, open3_rule, epsilon=0.0):
        win_or_block = find_immediate_win_or_block(board_state, current_turn, open3_rule)
        if win_or_block is not None:
            return win_or_block

        valid_moves = []
        mask = np.zeros(self.board_size * self.board_size, dtype=np.float32)
        for r in range(self.board_size):
            for c in range(self.board_size):
                idx = r * self.board_size + c
                if board_state[r][c] == 0:
                    if not open3_rule and current_turn == 1 and is_forbidden(r, c, 1):
                        continue
                    valid_moves.append((r, c))
                    mask[idx] = 1.0

        if not valid_moves: return None
        if np.random.rand() < epsilon:
            return valid_moves[np.random.choice(len(valid_moves))]

        self.model.eval()
        with torch.no_grad():
            logits = self.model(
                board_to_tensor(board_state, current_turn).to(self.device)
            ).squeeze(0).cpu().numpy()

        logits[mask == 0] = -1e9
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / exp_l.sum()

        idx = np.random.choice(self.board_size * self.board_size, p=probs)
        return idx // self.board_size, idx % self.board_size

    def train_step(self, states, actions, rewards):
        self.model.train()
        self.optimizer.zero_grad()
        loss_list = [
            -torch.log_softmax(
                self.model(s.to(self.device)).squeeze(0), dim=0
            )[a] * G
            for s, a, G in zip(states, actions, rewards)
        ]
        loss = torch.stack(loss_list).mean()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save_model(self):
        torch.save(self.model.state_dict(), self.saved_model_path)
        print(f"  Saved: {self.saved_model_path}")

    def reset_weights(self):
        """모델 가중치 초기화 (파일 삭제 후 재초기화)."""
        if os.path.exists(self.saved_model_path):
            try:
                os.remove(self.saved_model_path)
            except Exception as ex:
                print(f"  Error deleting: {ex}")
        self.model = GomokuNet(
            self.board_size, self.num_filters, self.num_layers
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)


# --- 에이전트 3개 생성 ---
print("\n[Windows] Loading RL Agents...")
rl_agent_easy   = RLAgent(GRID_SIZE, lr=0.001,  model_filename="gomoku_rl_easy.pth",
                           num_filters=32,  num_layers=3)
rl_agent_medium = RLAgent(GRID_SIZE, lr=0.0005, model_filename="gomoku_rl_medium.pth",
                           num_filters=64,  num_layers=3)
rl_agent_hard   = RLAgent(GRID_SIZE, lr=0.0003, model_filename="gomoku_rl_hard.pth",
                           num_filters=128, num_layers=4)
print(f"  Device: {rl_agent_hard.device}\n")


# ============================================================
# --- SuperHard AI: Minimax + Alpha-Beta Pruning ---
# ============================================================

def _score_sequence(seq, color):
    """
    한 라인(리스트)에서 color의 연속 패턴 점수를 반환.

    패턴 점수 기준:
      5+ in a row : 1,000,000  (승리)
      Open 4      :   100,000  (강제 승리)
      Half-open 4 :    10,000
      Open 3      :     5,000
      Half-open 3 :       500
      Open 2      :       100
      Half-open 2 :        20
    """
    score = 0
    n = len(seq)
    i = 0
    while i < n:
        if seq[i] == color:
            j = i
            while j < n and seq[j] == color:
                j += 1
            count = j - i
            left_open  = (i > 0 and seq[i - 1] == 0)
            right_open = (j < n and seq[j] == 0)
            open_ends  = (1 if left_open else 0) + (1 if right_open else 0)

            if count >= 5:
                score += 1_000_000
            elif count == 4:
                score += 100_000 if open_ends == 2 else (10_000 if open_ends == 1 else 200)
            elif count == 3:
                score += 5_000 if open_ends == 2 else (500 if open_ends == 1 else 50)
            elif count == 2:
                score += 100 if open_ends == 2 else (20 if open_ends == 1 else 0)
            elif count == 1:
                score += 5 if open_ends == 2 else 0
            i = j
        else:
            i += 1
    return score


def _evaluate_board(board_state, ai_color):
    """
    현재 보드 상태를 ai_color 관점에서 휴리스틱 평가.
    상대방 점수에 1.2 가중치를 두어 방어 우선 전략.
    """
    opponent  = 3 - ai_color
    all_lines = []

    # 가로
    for r in range(GRID_SIZE):
        all_lines.append([board_state[r][c] for c in range(GRID_SIZE)])
    # 세로
    for c in range(GRID_SIZE):
        all_lines.append([board_state[r][c] for r in range(GRID_SIZE)])
    # 대각선 ↘
    for d in range(-(GRID_SIZE - 1), GRID_SIZE):
        line = [board_state[r][r - d]
                for r in range(GRID_SIZE) if 0 <= r - d < GRID_SIZE]
        if len(line) >= 5: all_lines.append(line)
    # 대각선 ↙
    for s in range(2 * GRID_SIZE - 1):
        line = [board_state[r][s - r]
                for r in range(GRID_SIZE) if 0 <= s - r < GRID_SIZE]
        if len(line) >= 5: all_lines.append(line)

    my_score  = sum(_score_sequence(ln, ai_color) for ln in all_lines)
    opp_score = sum(_score_sequence(ln, opponent)  for ln in all_lines)
    return my_score - int(1.2 * opp_score)


# Minimax 호출 횟수 추적 (pygame.event.pump() 주기 호출용)
_mm_node_count = [0]


def _get_adjacent_moves(board_state, radius=1):
    """기존 돌 주변 radius 범위의 빈 칸 목록 반환.
    radius=1 : 바로 인접한 칸만 → 후보 수가 적어 탐색 속도 향상
    """
    candidates = set()
    has_stone  = False
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if board_state[r][c] != 0:
                has_stone = True
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        nr, nc = r + dr, c + dc
                        if in_bounds(nr, nc) and board_state[nr][nc] == 0:
                            candidates.add((nr, nc))
    if not has_stone:
        return [(GRID_SIZE // 2, GRID_SIZE // 2)]
    return list(candidates)


def _minimax(board_state, depth, alpha, beta, is_maximizing, ai_color, open3_rule):
    """
    Minimax + Alpha-Beta Pruning 재귀 탐색.

    Parameters
    ----------
    depth         : 남은 탐색 깊이
    is_maximizing : True -> AI 차례(최대화), False -> 상대 차례(최소화)
    ai_color      : SuperHard AI의 색 (항상 2: 백)
    """
    # pygame 이벤트 펌프: 100 노드마다 호출해 '응답 없음' 방지
    _mm_node_count[0] += 1
    if _mm_node_count[0] % 100 == 0:
        pygame.event.pump()

    opponent      = 3 - ai_color
    current_color = ai_color if is_maximizing else opponent

    if depth == 0:
        return _evaluate_board(board_state, ai_color), None

    candidates = _get_adjacent_moves(board_state)
    # 흑 금수 필터 (렌주 규칙 적용 시)
    if current_color == 1 and not open3_rule:
        candidates = [(r, c) for r, c in candidates if not is_forbidden(r, c, 1)]
    if not candidates:
        return _evaluate_board(board_state, ai_color), None

    # 빠른 정렬: 얕은 평가로 alpha-beta 효율 향상
    def quick_score(mv):
        r, c = mv
        board_state[r][c] = current_color
        val = _evaluate_board(board_state, ai_color)
        board_state[r][c] = 0
        return val

    candidates.sort(key=quick_score, reverse=is_maximizing)
    candidates = candidates[:10]   # 분기 한계: 상위 10수 (15→10으로 속도 개선)

    best_move = candidates[0]

    if is_maximizing:
        max_val = -float('inf')
        for r, c in candidates:
            board_state[r][c] = current_color
            if check_win_for_color(r, c, current_color):  # 즉시 승리
                board_state[r][c] = 0
                return 100_000 + depth * 1000, (r, c)
            val, _ = _minimax(board_state, depth - 1, alpha, beta,
                              False, ai_color, open3_rule)
            board_state[r][c] = 0
            if val > max_val:
                max_val = val; best_move = (r, c)
            alpha = max(alpha, val)
            if beta <= alpha: break           # Beta cut-off
        return max_val, best_move
    else:
        min_val = float('inf')
        for r, c in candidates:
            board_state[r][c] = current_color
            if check_win_for_color(r, c, current_color):  # 상대 즉시 승리
                board_state[r][c] = 0
                return -(100_000 + depth * 1000), (r, c)
            val, _ = _minimax(board_state, depth - 1, alpha, beta,
                              True, ai_color, open3_rule)
            board_state[r][c] = 0
            if val < min_val:
                min_val = val; best_move = (r, c)
            beta = min(beta, val)
            if beta <= alpha: break           # Alpha cut-off
        return min_val, best_move


def get_action_minimax(board_state, current_turn, open3_rule, depth=4):
    """
    SuperHard AI 착수 함수.
    즉시 승리/방어 -> Minimax(depth=4) + Alpha-Beta 순으로 처리.
    """
    win_or_block = find_immediate_win_or_block(board_state, current_turn, open3_rule)
    if win_or_block is not None:
        return win_or_block

    # 탐색 시작 전 노드 카운터 리셋
    _mm_node_count[0] = 0
    _, move = _minimax(board_state, depth,
                       -float('inf'), float('inf'),
                       True, current_turn, open3_rule)
    print(f"  SuperHard Minimax depth={depth} nodes={_mm_node_count[0]} -> {move}")
    return move


# ============================================================
# --- 학습 관리 ---
# ============================================================

# 난이도별 독립 학습 통계
train_data = {
    "Easy":   {"episodes": 0, "losses": [], "win_history": []},
    "Medium": {"episodes": 0, "losses": [], "win_history": []},
    "Hard":   {"episodes": 0, "losses": [], "win_history": []},
}
black_rate = white_rate = draw_rate = 0.0


def get_current_train_agent():
    """현재 학습 중인 에이전트 반환."""
    return {
        "Easy":   rl_agent_easy,
        "Medium": rl_agent_medium,
        "Hard":   rl_agent_hard,
    }.get(current_train_difficulty)


def _get_train_epsilon(diff):
    """난이도별 학습 epsilon 반환 (탐색률)."""
    return {"Easy": 0.4, "Medium": 0.25, "Hard": 0.15}.get(diff, 0.2)


def run_self_play_episode(agent, epsilon, open3_rule):
    """
    자기대국(Self-play) 1 에피소드 실행 후 REINFORCE로 학습.
    반환: (winner: int|None, loss: float)
    """
    global board
    board = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
    current_turn_ep = 1

    states_black, actions_black = [], []
    states_white, actions_white = [], []
    winner = None

    for _ in range(GRID_SIZE * GRID_SIZE):
        move = agent.get_action(board, current_turn_ep, open3_rule, epsilon)
        if move is None: break

        r, c = move
        state_t = board_to_tensor(board, current_turn_ep)
        action_idx = r * GRID_SIZE + c

        if current_turn_ep == 1:
            states_black.append(state_t); actions_black.append(action_idx)
        else:
            states_white.append(state_t); actions_white.append(action_idx)

        board[r][c] = current_turn_ep
        if check_win_for_color(r, c, current_turn_ep):
            winner = current_turn_ep; break

        current_turn_ep = 3 - current_turn_ep

    # 보상 계산 (감가된 누적 보상)
    states, actions, rewards = [], [], []
    gamma = 0.95

    if winner == 1:
        R = 1.0
        for s, a in reversed(list(zip(states_black, actions_black))):
            states.append(s); actions.append(a); rewards.append(R); R *= gamma
        R = -1.0
        for s, a in reversed(list(zip(states_white, actions_white))):
            states.append(s); actions.append(a); rewards.append(R); R *= gamma
    elif winner == 2:
        R = 1.0
        for s, a in reversed(list(zip(states_white, actions_white))):
            states.append(s); actions.append(a); rewards.append(R); R *= gamma
        R = -1.0
        for s, a in reversed(list(zip(states_black, actions_black))):
            states.append(s); actions.append(a); rewards.append(R); R *= gamma
    else:
        for s, a in zip(states_black + states_white, actions_black + actions_white):
            states.append(s); actions.append(a); rewards.append(-0.1)

    loss = agent.train_step(states, actions, rewards) if states else 0.0
    return winner, loss


# ============================================================
# --- AI 스레드 함수 (스레드에서 실행되는 AI 계산) ---
# ============================================================

def ai_compute_thread(board_copy, ai_turn, open3):
    """
    AI 착수를 별도 스레드에서 계산합니다.
    보드 복사본을 사용해 데이터 경쟁 없이 안전합니다.
    """
    global ai_move_result, ai_thinking
    try:
        if difficulty == "SuperHard":
            move = get_action_minimax(board_copy, ai_turn, open3, depth=4)
        elif difficulty == "Hard":
            move = get_action_mcts_vectorized(
                rl_agent_hard, board_copy, ai_turn, open3,
                num_rollouts=50, top_k=5)
        elif difficulty == "Medium":
            move = rl_agent_medium.get_action(board_copy, ai_turn, open3, epsilon=0.2)
        else:  # Easy
            move = rl_agent_easy.get_action(board_copy, ai_turn, open3, epsilon=0.5)
        ai_move_result = move
    except Exception as ex:
        print(f"[AI Thread Error] {ex}")
        ai_move_result = None
    finally:
        ai_thinking = False


# ============================================================
# --- 메인 게임 루프 ---
# ============================================================

running = True
winner  = None

while running:
    # ── 이벤트 처리 ────────────────────────────────────────
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:

            # ── 메인 메뉴 ──
            if state == "MENU_MODE":
                if event.key == pygame.K_1:
                    mode, state = "2P", "OPEN_3"
                elif event.key == pygame.K_2:
                    mode, state = "AI", "DIFFICULTY_MODE"
                elif event.key == pygame.K_3:
                    mode, state = "TRAIN", "TRAIN_DIFFICULTY"
                elif event.key == pygame.K_4:
                    state = "RESET_MODE"

            # ── AI 난이도 선택 ──
            elif state == "DIFFICULTY_MODE":
                if   event.key == pygame.K_1: difficulty = "Easy";      state = "OPEN_3"
                elif event.key == pygame.K_2: difficulty = "Medium";    state = "OPEN_3"
                elif event.key == pygame.K_3: difficulty = "Hard";      state = "OPEN_3"
                elif event.key == pygame.K_4: difficulty = "SuperHard"; state = "OPEN_3"
                elif event.key == pygame.K_ESCAPE: state = "MENU_MODE"

            # ── 학습 난이도 선택 ──
            elif state == "TRAIN_DIFFICULTY":
                if   event.key == pygame.K_1: current_train_difficulty = "Easy";   state = "OPEN_3"
                elif event.key == pygame.K_2: current_train_difficulty = "Medium"; state = "OPEN_3"
                elif event.key == pygame.K_3: current_train_difficulty = "Hard";   state = "OPEN_3"
                elif event.key == pygame.K_ESCAPE:
                    state = "MENU_MODE"; mode = None

            # ── 33 규칙 선택 ──
            elif state == "OPEN_3":
                if event.key in (pygame.K_1, pygame.K_2):
                    rule_open3 = (event.key == pygame.K_1)
                    state = "TRAIN_MODE" if mode == "TRAIN" else "GAME"
                    board  = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
                    winner = None
                    turn   = 1
                    ai_thinking = False
                    ai_move_result = None

            # ── 학습 중 ──
            elif state == "TRAIN_MODE":
                if event.key == pygame.K_ESCAPE:
                    ag = get_current_train_agent()
                    if ag:
                        ag.save_model()
                        print(f"Training stopped. [{current_train_difficulty}] saved.")
                    state = "MENU_MODE"

            # ── 리셋 선택 ──
            elif state == "RESET_MODE":
                if event.key == pygame.K_1:
                    rl_agent_easy.reset_weights()
                    train_data["Easy"] = {"episodes": 0, "losses": [], "win_history": []}
                    print("Easy AI Reset.")
                elif event.key == pygame.K_2:
                    rl_agent_medium.reset_weights()
                    train_data["Medium"] = {"episodes": 0, "losses": [], "win_history": []}
                    print("Medium AI Reset.")
                elif event.key == pygame.K_3:
                    rl_agent_hard.reset_weights()
                    train_data["Hard"] = {"episodes": 0, "losses": [], "win_history": []}
                    print("Hard AI Reset.")
                elif event.key == pygame.K_4:
                    for ag in (rl_agent_easy, rl_agent_medium, rl_agent_hard):
                         ag.reset_weights()
                    for key in train_data:
                        train_data[key] = {"episodes": 0, "losses": [], "win_history": []}
                    print("All AI Models Reset.")
                elif event.key == pygame.K_ESCAPE:
                    state = "MENU_MODE"

            # ── 게임 중 ──
            elif state == "GAME":
                if event.key == pygame.K_ESCAPE:
                    state = "MENU_MODE"
                    ai_thinking = False
                    ai_move_result = None

        # ── 마우스 클릭 (플레이어 착수) ──
        if state == "GAME" and event.type == pygame.MOUSEBUTTONDOWN and not winner:
            if not ai_thinking and (mode == "2P" or (mode == "AI" and turn == 1)):
                x, y  = event.pos
                c_pos = round(x / CELL_SIZE) - 1
                r_pos = round(y / CELL_SIZE) - 1
                if 0 <= r_pos < GRID_SIZE and 0 <= c_pos < GRID_SIZE and board[r_pos][c_pos] == 0:
                    print(f"Click ({r_pos},{c_pos}) turn={turn} rule_open3={rule_open3}")
                    forbidden = is_forbidden(r_pos, c_pos, turn)
                    if not rule_open3 and turn == 1 and forbidden:
                        board[r_pos][c_pos] = turn
                        is_ov  = check_overline(r_pos, c_pos, turn)
                        board[r_pos][c_pos] = 0
                        fours  = count_fours_created(r_pos, c_pos, turn)
                        if is_ov:
                            forbidden_msg = "Forbidden: Overline (6+)!"
                        elif fours >= 2:
                            forbidden_msg = "Forbidden: Double-Four (44)!"
                        else:
                            forbidden_msg = "Forbidden: Double-Three (33)!"
                        forbidden_time = pygame.time.get_ticks()
                        continue
                    board[r_pos][c_pos] = turn
                    if check_win_for_color(r_pos, c_pos, turn):
                        winner = turn
                    else:
                        turn = 3 - turn

    # ── AI 비동기 실행 및 착수 처리 ─────────────────────────
    if state == "GAME" and mode == "AI" and turn == 2 and not winner:
        if not ai_thinking:
            # AI 계산 스레드 시작
            ai_thinking = True
            ai_move_result = None
            board_copy = [row[:] for row in board]
            ai_thread = threading.Thread(
                target=ai_compute_thread,
                args=(board_copy, turn, rule_open3),
                daemon=True
            )
            ai_thread.start()
            thinking_timer = pygame.time.get_ticks()
        else:
            # "생각 중..." 애니메이션 타이머 업데이트
            now = pygame.time.get_ticks()
            if now - thinking_timer > 300:
                thinking_dots = (thinking_dots + 1) % 4
                thinking_timer = now

            # AI 계산이 완료되었는지 확인
            if ai_move_result is not None:
                r, c = ai_move_result
                board[r][c] = turn
                if check_win_for_color(r, c, turn):
                    winner = turn
                else:
                    turn = 3 - turn
                ai_thinking = False
                ai_move_result = None

    # ── 학습 루프 ───────────────────────────────────────────
    if state == "TRAIN_MODE":
        ag = get_current_train_agent()
        if ag:
            ep_winner, loss = run_self_play_episode(
                ag, epsilon=_get_train_epsilon(current_train_difficulty),
                open3_rule=rule_open3)

            td = train_data[current_train_difficulty]
            td["episodes"] += 1
            td["losses"].append(loss)
            td["win_history"].append(ep_winner if ep_winner is not None else 0)

            recent = td["win_history"][-100:]
            n_r    = len(recent)
            if n_r > 0:
                black_rate = recent.count(1) / n_r * 100
                white_rate = recent.count(2) / n_r * 100
                draw_rate  = recent.count(0) / n_r * 100
            else:
                black_rate = white_rate = draw_rate = 0.0

    # ── 화면 렌더링 ─────────────────────────────────────────

    if state == "MENU_MODE":
        screen.fill(WHITE)
        draw_text("Omok 13x13  AI System", 80,  (0, 0, 160))
        draw_text("Windows Edition",        115, (80, 80, 200))
        draw_text("1.  Local 1 vs 1 Game",  205)
        draw_text("2.  Player vs AI",        255)
        draw_text("3.  Train AI (RL Self-Play)", 305)
        draw_text("4.  Reset AI Model",      355, (160, 0, 0))
        draw_text(f"Models: {MODELS_DIR.name}/", 430, (120, 120, 120))
        draw_text("Select with keyboard",   490, (130, 130, 130))

    elif state == "DIFFICULTY_MODE":
        screen.fill(WHITE)
        e_ep = train_data["Easy"]["episodes"]
        m_ep = train_data["Medium"]["episodes"]
        h_ep = train_data["Hard"]["episodes"]
        draw_text("Select AI Difficulty",         105, (0, 0, 160))
        draw_text(f"1.  Easy       [{e_ep:4d} ep trained]", 205, (0, 140, 0))
        draw_text(f"2.  Medium     [{m_ep:4d} ep trained]", 260, (0, 60, 200))
        draw_text(f"3.  Hard       [{h_ep:4d} ep trained]", 315, (200, 80, 0))
        draw_text("4.  SuperHard  [Minimax 4-ply]",        370, (130, 0, 180))
        draw_text("(SuperHard needs no training)",         410, (150, 150, 150))
        draw_text("ESC: Back",                             490, (120, 120, 120))

    elif state == "TRAIN_DIFFICULTY":
        screen.fill(WHITE)
        e_ep = train_data["Easy"]["episodes"]
        m_ep = train_data["Medium"]["episodes"]
        h_ep = train_data["Hard"]["episodes"]
        draw_text("Select Training Target",                120, (0, 110, 0))
        draw_text(f"1.  Train Easy AI   [{e_ep:4d} ep]",  225, (0, 140, 0))
        draw_text(f"2.  Train Medium AI [{m_ep:4d} ep]",  285, (0, 60, 200))
        draw_text(f"3.  Train Hard AI   [{h_ep:4d} ep]",  345, (200, 80, 0))
        draw_text(f"Goal: {MIN_TRAIN_EPISODES}+ episodes per difficulty", 415, (180, 100, 0))
        draw_text("SuperHard = Minimax (no training needed)", 455, (130, 0, 180))
        draw_text("ESC: Back",                             510, (120, 120, 120))

    elif state == "RESET_MODE":
        screen.fill(WHITE)
        e_ep = train_data["Easy"]["episodes"]
        m_ep = train_data["Medium"]["episodes"]
        h_ep = train_data["Hard"]["episodes"]
        draw_text("Reset AI Model Weight",                 105, (170, 0, 0))
        draw_text(f"1.  Reset Easy AI   [{e_ep:4d} ep]",  210, (0, 140, 0))
        draw_text(f"2.  Reset Medium AI [{m_ep:4d} ep]",  268, (0, 60, 200))
        draw_text(f"3.  Reset Hard AI   [{h_ep:4d} ep]",  326, (200, 80, 0))
        draw_text("4.  Reset ALL Models",                  384, (200, 0, 0))
        draw_text("ESC: Back",                             470, (120, 120, 120))

    elif state == "OPEN_3":
        screen.fill(WHITE)
        draw_text("Double-Three (33) Rule Setting",  180, (0, 100, 0))
        draw_text("1.  Allow Double-Three (Standard)", 280)
        draw_text("2.  Prohibit Double-Three (Renju)", 340)
        draw_text("Press 1 or 2",                    440, (120, 120, 120))

    elif state == "GAME":
        draw_board()
        if winner:
            draw_text(f"{'Black' if winner==1 else 'White'} Wins!",
                      30, (200, 0, 0))
            draw_text("ESC: Back to Menu", SCREEN_SIZE - 30, (120, 120, 120))
        else:
            if ai_thinking:
                dots_str = "." * thinking_dots
                draw_text(f"AI is thinking{dots_str}", 30, (220, 0, 0))
            else:
                draw_text(f"Turn: {'Black' if turn==1 else 'White'}", 30,
                          (0, 0, 200) if turn == 1 else (80, 80, 80))
            
            if mode == "AI":
                diff_colors = {
                    "Easy":      (0, 150, 0),
                    "Medium":    (0, 60, 200),
                    "Hard":      (200, 80, 0),
                    "SuperHard": (130, 0, 180),
                }
                draw_text(f"Difficulty: {difficulty}",
                          57, diff_colors.get(difficulty, BLACK))
            if forbidden_msg and pygame.time.get_ticks() - forbidden_time < 2000:
                draw_text(forbidden_msg, SCREEN_SIZE - 30, (220, 0, 0))
            else:
                forbidden_msg = None

    elif state == "TRAIN_MODE":
        draw_board()

        # 반투명 오버레이
        overlay = pygame.Surface((410, 310))
        overlay.set_alpha(235)
        overlay.fill(WHITE)
        screen.blit(overlay, (95, 148))

        sm_font = pygame.font.SysFont("Arial", 17)
        def dt(text, y, col=BLACK):
            s = sm_font.render(text, True, col)
            screen.blit(s, s.get_rect(center=(SCREEN_SIZE / 2, y)))

        td      = train_data.get(current_train_difficulty, {})
        ep_cnt  = td.get("episodes", 0)
        losses  = td.get("losses", [])

        diff_col = {"Easy": (0,140,0), "Medium": (0,60,200), "Hard": (200,80,0)
                    }.get(current_train_difficulty, BLACK)
        dt(f"[{current_train_difficulty}] AI Self-Play Training...", 168, diff_col)

        # 100에피소드 진행률 바
        if ep_cnt < MIN_TRAIN_EPISODES:
            dt(f"Progress: {ep_cnt} / {MIN_TRAIN_EPISODES} episodes",
               198, (190, 100, 0))
            bar_x, bar_y, bar_w, bar_h = 115, 212, 370, 14
            pygame.draw.rect(screen, (210, 210, 210), (bar_x, bar_y, bar_w, bar_h))
            filled = int(bar_w * ep_cnt / MIN_TRAIN_EPISODES)
            pygame.draw.rect(screen, (200, 130, 0), (bar_x, bar_y, filled, bar_h))
            pygame.draw.rect(screen, (180, 180, 180), (bar_x, bar_y, bar_w, bar_h), 1)
            dt("100+ episodes recommended for stable play", 238, (160, 80, 0))
        else:
            dt(f"Episodes: {ep_cnt}   (100+ Goal Achieved!)",
               208, (0, 150, 0))

        dt(f"Loss:         {losses[-1]:.4f}" if losses else "Loss: N/A", 268)
        dt(f"Black Wins (last 100): {black_rate:.1f}%",                  292)
        dt(f"White Wins (last 100): {white_rate:.1f}%",                  315)
        dt(f"Draws      (last 100): {draw_rate:.1f}%",                   338)
        dt("Press ESC to Stop and Save Model",                            382, (100,100,100))

    pygame.display.flip()

pygame.quit()
sys.exit()
