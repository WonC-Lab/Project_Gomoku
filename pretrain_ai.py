"""
pretrain_ai.py  -  Gomoku AI 사전 학습 스크립트

"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pathlib
import time

# ============================================================
# 경로 설정
# ============================================================
BASE_DIR   = pathlib.Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ============================================================
# 학습 설정
# ============================================================
TRAIN_CONFIG = {
    "Easy": {
        "episodes":    1000,   
        "epsilon":     0.4,   
        "num_filters": 32,
        "num_layers":  3,
        "lr":          0.001,
        "filename":    "gomoku_rl_easy.pth",
    },
    "Medium": {
        "episodes":    2000,   
        "epsilon":     0.25,
        "num_filters": 64,
        "num_layers":  3,
        "lr":          0.0005,
        "filename":    "gomoku_rl_medium.pth",
    },
    "Hard": {
        "episodes":    3000,  
        "epsilon":     0.15,
        "num_filters": 128,
        "num_layers":  4,
        "lr":          0.0003,
        "filename":    "gomoku_rl_hard.pth",
    },
}

# 학습 시 33 규칙 기본값 (True: 허용 = 일반 룰)
OPEN3_RULE = True

# ============================================================
# 게임 전역 상태
# ============================================================
GRID_SIZE  = 13
DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]
board      = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]

# ============================================================
# 게임 로직 
# ============================================================

def in_bounds(r, c):
    return 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE

def check_overline(r, c, color):
    if color != 1: return False
    for dr, dc in DIRECTIONS:
        count = 1
        for i in [1, -1]:
            nr, nc = r + dr * i, c + dc * i
            while in_bounds(nr, nc) and board[nr][nc] == color:
                count += 1; nr += dr * i; nc += dc * i
        if count >= 6: return True
    return False

def is_five(r, c, color):
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
    fours_count = 0
    board[r][c] = color
    for dr, dc in DIRECTIONS:
        has_four = False
        for k in range(-4, 5):
            if k == 0: continue
            nr, nc = r + dr * k, c + dc * k
            if in_bounds(nr, nc) and board[nr][nc] == 0:
                board[nr][nc] = color
                if is_five_in_dir(nr, nc, color, dr, dc): has_four = True
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
                if is_forbidden(nr, nc, color, depth + 1): continue
            board[nr][nc] = color
            winning_spots = 0
            for m in range(-4, 5):
                if m == 0 or m == k: continue
                wr, wc = r + dr * m, c + dc * m
                if in_bounds(wr, wc) and board[wr][wc] == 0:
                    board[wr][wc] = color
                    if is_five_in_dir(wr, wc, color, dr, dc): winning_spots += 1
                    board[wr][wc] = 0
            board[nr][nc] = 0
            if winning_spots >= 2: return True
    return False

def is_forbidden(r, c, color, depth=0):
    if color != 1: return False
    if board[r][c] != 0: return False
    board[r][c] = color
    five = is_five(r, c, color)
    board[r][c] = 0
    if five: return False
    board[r][c] = color
    is_ov = check_overline(r, c, color)
    board[r][c] = 0
    if is_ov: return True
    if count_fours_created(r, c, color) >= 2: return True
    board[r][c] = color
    open_threes = sum(
        1 for dr, dc in DIRECTIONS
        if is_open_three_in_dir(r, c, color, dr, dc, depth)
    )
    board[r][c] = 0
    return open_threes >= 2

def check_win_for_color(r, c, color):
    for dr, dc in DIRECTIONS:
        count = 1
        for i in [1, -1]:
            nr, nc = r + dr * i, c + dc * i
            while in_bounds(nr, nc) and board[nr][nc] == color:
                count += 1; nr += dr * i; nc += dc * i
        if color == 1 and not OPEN3_RULE:
            if count == 5: return True
        else:
            if count >= 5: return True
    return False

# ============================================================
# 신경망
# ============================================================

class GomokuNet(nn.Module):
    def __init__(self, board_size=13, num_filters=64, num_layers=3):
        super().__init__()
        self.board_size = board_size
        layers = [
            nn.Conv2d(3, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters), nn.ReLU()
        ]
        for _ in range(num_layers - 1):
            layers += [
                nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_filters), nn.ReLU()
            ]
        self.conv = nn.Sequential(*layers)
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1),
            nn.BatchNorm2d(2), nn.ReLU(), nn.Flatten(),
            nn.Linear(2 * board_size * board_size, board_size * board_size)
        )

    def forward(self, x):
        return self.policy_head(self.conv(x))

def board_to_tensor(board_state, current_turn):
    arr = np.array(board_state)
    opp = 3 - current_turn
    return torch.tensor(np.stack([
        (arr == current_turn).astype(np.float32),
        (arr == opp).astype(np.float32),
        (arr == 0).astype(np.float32)
    ], axis=0)).unsqueeze(0)

# ============================================================
# RL 에이전트
# ============================================================

class RLAgent:
    def __init__(self, board_size, lr, model_filename, num_filters, num_layers):
        self.board_size = board_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_filters = num_filters
        self.num_layers  = num_layers
        self.lr = lr
        self.model = GomokuNet(board_size, num_filters, num_layers).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.saved_model_path = str(MODELS_DIR / model_filename)

        if os.path.exists(self.saved_model_path):
            try:
                self.model.load_state_dict(
                    torch.load(self.saved_model_path,
                               map_location=self.device, weights_only=True))
                print(f"  [OK] Loaded existing weights: {model_filename}")
            except Exception as ex:
                print(f"  [NG] Failed to load {model_filename}: {ex}")

    def get_action(self, board_state, current_turn, epsilon=0.0):
        valid_moves = []
        mask = np.zeros(self.board_size ** 2, dtype=np.float32)
        for r in range(self.board_size):
            for c in range(self.board_size):
                idx = r * self.board_size + c
                if board_state[r][c] == 0:
                    if not OPEN3_RULE and current_turn == 1 and is_forbidden(r, c, 1):
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
        idx = np.random.choice(self.board_size ** 2, p=probs)
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

# ============================================================
# 자기대국 에피소드
# ============================================================

def run_self_play_episode(agent, epsilon):
    global board
    board = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
    turn_ep = 1
    states_b, actions_b = [], []
    states_w, actions_w = [], []
    winner = None

    for _ in range(GRID_SIZE * GRID_SIZE):
        move = agent.get_action(board, turn_ep, epsilon)
        if move is None: break
        r, c = move
        st = board_to_tensor(board, turn_ep)
        ai = r * GRID_SIZE + c
        if turn_ep == 1:
            states_b.append(st); actions_b.append(ai)
        else:
            states_w.append(st); actions_w.append(ai)
        board[r][c] = turn_ep
        if check_win_for_color(r, c, turn_ep):
            winner = turn_ep; break
        turn_ep = 3 - turn_ep

    states, actions, rewards = [], [], []
    gamma = 0.95

    if winner == 1:
        R = 1.0
        for s, a in reversed(list(zip(states_b, actions_b))):
            states.append(s); actions.append(a); rewards.append(R); R *= gamma
        R = -1.0
        for s, a in reversed(list(zip(states_w, actions_w))):
            states.append(s); actions.append(a); rewards.append(R); R *= gamma
    elif winner == 2:
        R = 1.0
        for s, a in reversed(list(zip(states_w, actions_w))):
            states.append(s); actions.append(a); rewards.append(R); R *= gamma
        R = -1.0
        for s, a in reversed(list(zip(states_b, actions_b))):
            states.append(s); actions.append(a); rewards.append(R); R *= gamma
    else:
        for s, a in zip(states_b + states_w, actions_b + actions_w):
            states.append(s); actions.append(a); rewards.append(-0.1)

    loss = agent.train_step(states, actions, rewards) if states else 0.0
    return winner, loss

# ============================================================
# 출력 헬퍼
# ============================================================

def fmt_time(sec):
    sec = int(sec)
    if sec < 60:   return f"{sec}s"
    if sec < 3600: return f"{sec//60}m {sec%60:02d}s"
    return f"{sec//3600}h {(sec%3600)//60:02d}m"

def progress_bar(ep, total, width=28):
    filled = int(width * ep / total)
    return "#" * filled + "." * (width - filled)

# ============================================================
# 난이도별 학습 함수
# ============================================================

def train_difficulty(diff, cfg):
    total_eps = cfg["episodes"]

    print(f"\n{'='*62}")
    print(f"  [{diff.upper()}] AI Pre-Training")
    print(f"  Network : {cfg['num_filters']} filters x {cfg['num_layers']} layers")
    print(f"  Episodes: {total_eps:,}   Epsilon: {cfg['epsilon']}")
    print(f"  Save to : models/{cfg['filename']}")
    print(f"{'='*62}")

    agent = RLAgent(
        board_size=GRID_SIZE,
        lr=cfg["lr"],
        model_filename=cfg["filename"],
        num_filters=cfg["num_filters"],
        num_layers=cfg["num_layers"],
    )

    win_history, losses = [], []
    start_time = time.time()

    for ep in range(1, total_eps + 1):
        winner, loss = run_self_play_episode(agent, cfg["epsilon"])
        losses.append(loss)
        win_history.append(winner if winner is not None else 0)

        # 10 에피소드마다 한 줄 업데이트
        if ep % 10 == 0 or ep == 1:
            recent = win_history[-100:]
            n = len(recent)
            b_r = recent.count(1) / n * 100
            w_r = recent.count(2) / n * 100
            d_r = recent.count(0) / n * 100
            avg_loss = float(np.mean(losses[-20:])) if len(losses) >= 20 else float(np.mean(losses))

            elapsed = time.time() - start_time
            speed   = ep / elapsed if elapsed > 0 else 1
            eta     = (total_eps - ep) / speed

            bar = progress_bar(ep, total_eps)
            pct = ep / total_eps * 100
            print(
                f"\r  [{bar}] {pct:5.1f}%  "
                f"Ep:{ep:5d}  Loss:{avg_loss:.4f}  "
                f"B:{b_r:.0f}% W:{w_r:.0f}% D:{d_r:.0f}%  "
                f"ETA:{fmt_time(eta)}   ",
                end="", flush=True
            )

        # 100 episodes checkpoint save
        if ep % 100 == 0:
            agent.save_model()
            elapsed = time.time() - start_time
            speed   = ep / elapsed if elapsed > 0 else 1
            eta     = (total_eps - ep) / speed
            print(f"\n  >> Checkpoint saved at ep {ep}  "
                  f"Speed: {speed:.1f} ep/s  ETA: {fmt_time(eta)}")

    # 최종 저장
    agent.save_model()
    total_time = time.time() - start_time
    recent = win_history[-100:]
    n = len(recent)
    print(f"\n  [DONE] [{diff}] Training complete!  Time: {fmt_time(total_time)}")
    print(f"  Last 100 ep - Black: {recent.count(1)/n*100:.1f}%  "
          f"White: {recent.count(2)/n*100:.1f}%  "
          f"Draw: {recent.count(0)/n*100:.1f}%")
    print(f"  Saved: {agent.saved_model_path}")

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    device_name = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\n" + "="*62)
    print("  Gomoku AI Pre-Training Script [Windows]")
    print(f"  Device     : {device_name}")
    print(f"  Model dir  : {MODELS_DIR}")
    print(f"  Train mode : Self-Play RL (REINFORCE)")
    print(f"  Rule 33    : {'Allow (Standard)' if OPEN3_RULE else 'Forbid (Renju)'}")
    print("="*62)

    total_ep_all = sum(cfg["episodes"] for cfg in TRAIN_CONFIG.values())
    print(f"\n  Total episodes: {total_ep_all:,}  "
          f"(Easy:{TRAIN_CONFIG['Easy']['episodes']} + "
          f"Medium:{TRAIN_CONFIG['Medium']['episodes']} + "
          f"Hard:{TRAIN_CONFIG['Hard']['episodes']})")
    print("\n  Press Ctrl+C to stop early. Last checkpoint will be kept.\n")

    grand_start = time.time()

    for diff, cfg in TRAIN_CONFIG.items():
        try:
            train_difficulty(diff, cfg)
        except KeyboardInterrupt:
            print(f"\n\n  [STOP] Training interrupted by user.")
            print(f"  Last checkpoint has been saved.")
            break

    grand_total = time.time() - grand_start
    print(f"\n{'='*62}")
    print(f"  All training complete!  Total: {fmt_time(grand_total)}")
    print(f"  Saved model files:")
    for cfg in TRAIN_CONFIG.values():
        p = MODELS_DIR / cfg["filename"]
        if p.exists():
            size_kb = p.stat().st_size / 1024
            print(f"    [OK] {cfg['filename']}  ({size_kb:.0f} KB)")
        else:
            print(f"    [NG] {cfg['filename']}  (not saved)")
    print(f"\n  You can now play against the trained AI!")
    print("="*62 + "\n")
