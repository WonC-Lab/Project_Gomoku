import pygame
import sys

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import os
except ImportError as e:
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
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
            f"Error detail: {e}",
            "",
            "Please run this script using Anaconda Python:",
            "  /opt/anaconda3/bin/python3 Gomuku_Project.py",
            "",
            "Press close window or any key to exit."
        ]
        y = 150
        for line in lines:
            color = (200, 0, 0) if "Error" in line else (0, 0, 0)
            surf = font.render(line, True, color)
            rect = surf.get_rect(center=(300, y))
            screen.blit(surf, rect)
            y += 40
        pygame.display.flip()
    pygame.quit()
    sys.exit()

# --- 초기 설정 ---
pygame.init()
SCREEN_SIZE = 600
GRID_SIZE = 13
CELL_SIZE = SCREEN_SIZE // (GRID_SIZE + 1)
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption("Omok 13x13 - RL AI & Strict Renju")
font = pygame.font.SysFont("Arial", 28)

WHITE, BLACK, BOARD_COLOR = (255, 255, 255), (0, 0, 0), (240, 217, 181)

# 게임 상태
state = "MENU_MODE" 
mode = None          
rule_open3 = None    
difficulty = None
forbidden_msg = None
forbidden_time = 0
board = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
turn = 1 # 1: 흑, 2: 백

DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]

def in_bounds(r, c):
    return 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE

def draw_text(text, y, color=BLACK):
    text_surf = font.render(text, True, color)
    text_rect = text_surf.get_rect(center=(SCREEN_SIZE/2, y))
    screen.blit(text_surf, text_rect)

# --- 렌주룰 (Strict Renju Rules) ---

def check_overline(r, c, color):
    if color != 1:
        return False
    for dr, dc in DIRECTIONS:
        count = 1
        for i in [1, -1]:
            nr, nc = r + dr * i, c + dc * i
            while in_bounds(nr, nc) and board[nr][nc] == color:
                count += 1
                nr += dr * i
                nc += dc * i
        if count >= 6:
            return True
    return False

def is_five(r, c, color):
    for dr, dc in DIRECTIONS:
        count = 1
        for i in [1, -1]:
            nr, nc = r + dr * i, c + dc * i
            while in_bounds(nr, nc) and board[nr][nc] == color:
                count += 1
                nr += dr * i
                nc += dc * i
        if color == 1:
            if count == 5:
                return True
        else:
            if count >= 5:  # 백돌은 장목도 승리
                return True
    return False

def is_five_in_dir(r, c, color, dr, dc):
    count = 1
    for i in [1, -1]:
        nr, nc = r + dr * i, c + dc * i
        while in_bounds(nr, nc) and board[nr][nc] == color:
            count += 1
            nr += dr * i
            nc += dc * i
    if color == 1:
        return count == 5
    else:
        return count >= 5

def count_fours_created(r, c, color):
    fours_count = 0
    board[r][c] = color
    
    for dr, dc in DIRECTIONS:
        has_four_in_dir = False
        for k in range(-4, 5):
            if k == 0:
                continue
            nr, nc = r + dr * k, c + dc * k
            if in_bounds(nr, nc) and board[nr][nc] == 0:
                board[nr][nc] = color
                if is_five_in_dir(nr, nc, color, dr, dc):
                    has_four_in_dir = True
                board[nr][nc] = 0
                if has_four_in_dir:
                    break
        if has_four_in_dir:
            fours_count += 1
            
    board[r][c] = 0
    return fours_count

def is_open_three_in_dir(r, c, color, dr, dc, depth):
    """특정 방향으로 '열린 3'이 만들어지는지 확인 (재귀 금수 체크 포함)"""
    for k in range(-4, 5):
        if k == 0:
            continue
        nr, nc = r + dr * k, c + dc * k
        if in_bounds(nr, nc) and board[nr][nc] == 0:
            # 1. 3을 4로 확장시키는 해당 빈칸(nr, nc)이 스스로 금수 자리가 아닌지 확인
            if depth < 2:
                if is_forbidden(nr, nc, color, depth + 1):
                    continue
            
            # 2. 해당 자리에 돌을 놓았을 때 '열린 4'가 되는지 확인
            board[nr][nc] = color
            
            winning_spots = 0
            for m in range(-4, 5):
                if m == 0 or m == k:
                    continue
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
    """렌주룰 종합 금수 판정 (33, 44, 장목)"""
    if color != 1:  # 백돌은 금수가 없음
        return False
    if board[r][c] != 0:
        return False
        
    # 5목이 완성되는 수는 금수 룰보다 승리가 우선함
    board[r][c] = color
    five = is_five(r, c, color)
    board[r][c] = 0
    if five:
        return False
        
    # 1. 장목(Overline) 체크
    board[r][c] = color
    is_ov = check_overline(r, c, color)
    board[r][c] = 0
    if is_ov:
        return True
        
    # 2. 44(Double Four) 체크
    fours = count_fours_created(r, c, color)
    if fours >= 2:
        return True
        
    # 3. 33(Double Three) 체크
    board[r][c] = color
    open_threes = 0
    for dr, dc in DIRECTIONS:
        if is_open_three_in_dir(r, c, color, dr, dc, depth):
            open_threes += 1
    board[r][c] = 0
    
    if open_threes >= 2:
        return True
        
    return False

def check_win_for_color(r, c, color):
    for dr, dc in DIRECTIONS:
        count = 1
        for i in [1, -1]:
            nr, nc = r + dr * i, c + dc * i
            while in_bounds(nr, nc) and board[nr][nc] == color:
                count += 1
                nr += dr * i
                nc += dc * i
        if color == 1 and not rule_open3:
            # 금수 룰 적용 시 흑은 정확히 5목이어야 승리
            if count == 5:
                return True
        else:
            # 금수 룰 해제 시 및 백돌은 5목 이상이면 승리
            if count >= 5:
                return True
    return False

def draw_board():
    screen.fill(BOARD_COLOR)
    for i in range(GRID_SIZE):
        pygame.draw.line(screen, BLACK, (CELL_SIZE, CELL_SIZE * (i + 1)), (CELL_SIZE * GRID_SIZE, CELL_SIZE * (i + 1)))
        pygame.draw.line(screen, BLACK, (CELL_SIZE * (i + 1), CELL_SIZE), (CELL_SIZE * (i + 1), CELL_SIZE * GRID_SIZE))
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if board[r][c] == 1: pygame.draw.circle(screen, BLACK, (CELL_SIZE * (c + 1), CELL_SIZE * (r + 1)), CELL_SIZE // 2 - 2)
            elif board[r][c] == 2: pygame.draw.circle(screen, WHITE, (CELL_SIZE * (c + 1), CELL_SIZE * (r + 1)), CELL_SIZE // 2 - 2)

"""Reinforcement Algorithms"""
class GomokuNet(nn.Module):
    def __init__(self, board_size=13):
        super(GomokuNet, self).__init__()
        self.board_size = board_size
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, board_size * board_size)
        )
        
    def forward(self, x):
        features = self.conv(x)
        logits = self.policy_head(features)
        return logits

def board_to_tensor(board_state, current_turn):
    board_arr = np.array(board_state)
    opponent = 3 - current_turn
    ch0 = (board_arr == current_turn).astype(np.float32)
    ch1 = (board_arr == opponent).astype(np.float32)
    ch2 = (board_arr == 0).astype(np.float32)
    state = np.stack([ch0, ch1, ch2], axis=0)
    return torch.tensor(state).unsqueeze(0)

def find_immediate_win_or_block(board_state, current_turn, open3_rule):
    opponent = 3 - current_turn
    
    # 1. 즉시 5목을 만들어 승리할 수 있는 자리 탐색 (공격 우선)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if board_state[r][c] == 0:
                board_state[r][c] = current_turn
                win = check_win_for_color(r, c, current_turn)
                board_state[r][c] = 0
                if win:
                    return (r, c)
                    
    # 2. 상대방이 놓아서 바로 5목을 만들 수 있는 자리(방어 필수 자리) 탐색
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if board_state[r][c] == 0:
                board_state[r][c] = opponent
                win = check_win_for_color(r, c, opponent)
                board_state[r][c] = 0
                if win:
                    # 백돌 AI인 경우 상대방(흑)의 금수 자리는 수비할 필요 없음 (상대방이 못 두므로)
                    if current_turn == 2 and not open3_rule and is_forbidden(r, c, 1):
                        continue
                    return (r, c)

    # 3. AI가 놓아서 양수(Open Four, 두 곳의 5목 활로)를 만드는 자리 탐색 (공격)
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
                    if winning_spots >= 2:
                        is_op4 = True
                        break
                board_state[r][c] = 0
                if is_op4:
                    return (r, c)

    # 4. 상대방이 놓아서 양수(Open Four)를 만드는 자리 탐색 (수비: 상대의 열린 3 방어)
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
                    if winning_spots >= 2:
                        is_op4 = True
                        break
                board_state[r][c] = 0
                if is_op4:
                    if current_turn == 2 and not open3_rule and is_forbidden(r, c, 1):
                        continue
                    return (r, c)
    return None

def get_action_mcts(agent, board_state, current_turn, open3_rule, num_rollouts=15, top_k=5):
    # 즉시 승리/방어 자리가 있는지 최우선 확인
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
                
    if not valid_moves:
        return None
    if len(valid_moves) == 1:
        return valid_moves[0]
        
    agent.model.eval()
    state_tensor = board_to_tensor(board_state, current_turn).to(agent.device)
    with torch.no_grad():
        logits = agent.model(state_tensor).squeeze(0).cpu().numpy()
        
    logits[mask == 0] = -1e9
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    
    move_indices = np.argsort(probs)[::-1]
    candidates = []
    for idx in move_indices:
        r = idx // agent.board_size
        c = idx % agent.board_size
        if mask[idx] > 0:
            candidates.append((r, c))
            if len(candidates) >= top_k:
                break
                
    if not candidates:
        return valid_moves[0]
        
    best_move = candidates[0]
    best_win_rate = -1.0
    
    def sim_in_bounds(r, c):
        return 0 <= r < agent.board_size and 0 <= c < agent.board_size
        
    def sim_check_win(sim_board, r, c, color):
        for dr, dc in DIRECTIONS:
            count = 1
            for i in [1, -1]:
                nr, nc = r + dr * i, c + dc * i
                while sim_in_bounds(nr, nc) and sim_board[nr][nc] == color:
                    count += 1
                    nr += dr * i
                    nc += dc * i
            if color == 1 and not open3_rule:
                if count == 5:
                    return True
            else:
                if count >= 5:
                    return True
        return False

    def sim_check_overline(sim_board, r, c, color):
        if color != 1:
            return False
        for dr, dc in DIRECTIONS:
            count = 1
            for i in [1, -1]:
                nr, nc = r + dr * i, c + dc * i
                while sim_in_bounds(nr, nc) and sim_board[nr][nc] == color:
                    count += 1
                    nr += dr * i
                    nc += dc * i
            if count >= 6:
                return True
        return False

    def sim_is_five_in_dir(sim_board, r, c, color, dr, dc):
        count = 1
        for i in [1, -1]:
            nr, nc = r + dr * i, c + dc * i
            while sim_in_bounds(nr, nc) and sim_board[nr][nc] == color:
                count += 1
                nr += dr * i
                nc += dc * i
        if color == 1:
            return count == 5
        else:
            return count >= 5

    def sim_count_fours_created(sim_board, r, c, color):
        fours_count = 0
        sim_board[r][c] = color
        for dr, dc in DIRECTIONS:
            has_four_in_dir = False
            for k in range(-4, 5):
                if k == 0:
                    continue
                nr, nc = r + dr * k, c + dc * k
                if sim_in_bounds(nr, nc) and sim_board[nr][nc] == 0:
                    sim_board[nr][nc] = color
                    if sim_is_five_in_dir(sim_board, nr, nc, color, dr, dc):
                        has_four_in_dir = True
                    sim_board[nr][nc] = 0
                    if has_four_in_dir:
                        break
            if has_four_in_dir:
                fours_count += 1
        sim_board[r][c] = 0
        return fours_count

    def sim_is_open_three_in_dir(sim_board, r, c, color, dr, dc, depth):
        for k in range(-4, 5):
            if k == 0:
                continue
            nr, nc = r + dr * k, c + dc * k
            if sim_in_bounds(nr, nc) and sim_board[nr][nc] == 0:
                if depth < 2:
                    if sim_is_forbidden(sim_board, nr, nc, color, depth + 1):
                        continue
                sim_board[nr][nc] = color
                winning_spots = 0
                for m in range(-4, 5):
                    if m == 0 or m == k:
                        continue
                    wr, wc = r + dr * m, c + dc * m
                    if sim_in_bounds(wr, wc) and sim_board[wr][wc] == 0:
                        sim_board[wr][wc] = color
                        if sim_is_five_in_dir(sim_board, wr, wc, color, dr, dc):
                            winning_spots += 1
                        sim_board[wr][wc] = 0
                sim_board[nr][nc] = 0
                if winning_spots >= 2:
                    return True
        return False

    def sim_is_forbidden(sim_board, r, c, color, depth=0):
        if color != 1:
            return False
        if sim_board[r][c] != 0:
            return False
        sim_board[r][c] = color
        five = sim_check_win(sim_board, r, c, color)
        sim_board[r][c] = 0
        if five:
            return False
        sim_board[r][c] = color
        is_ov = sim_check_overline(sim_board, r, c, color)
        sim_board[r][c] = 0
        if is_ov:
            return True
        fours = sim_count_fours_created(sim_board, r, c, color)
        if fours >= 2:
            return True
        sim_board[r][c] = color
        open_threes = 0
        for dr, dc in DIRECTIONS:
            if sim_is_open_three_in_dir(sim_board, r, c, color, dr, dc, depth):
                open_threes += 1
        sim_board[r][c] = 0
        if open_threes >= 2:
            return True
        return False
        
    print("\n--- AI Move Candidate Win Rates (CPU MCTS) ---")
    opponent = 3 - current_turn
    for cand in candidates:
        cand_wins = 0
        cand_losses = 0
        cand_draws = 0
        for _ in range(num_rollouts):
            sim_board = [row[:] for row in board_state]
            cr, cc = cand
            sim_board[cr][cc] = current_turn
            
            sim_turn = 3 - current_turn
            sim_winner = None
            sim_steps = 0
            while sim_steps < 40:
                sim_valid_moves = []
                sim_mask = np.zeros(agent.board_size * agent.board_size, dtype=np.float32)
                for sr in range(agent.board_size):
                    for sc in range(agent.board_size):
                        sidx = sr * agent.board_size + sc
                        if sim_board[sr][sc] == 0:
                            if not open3_rule and sim_turn == 1 and sim_is_forbidden(sim_board, sr, sc, 1):
                                continue
                            sim_valid_moves.append((sr, sc))
                            sim_mask[sidx] = 1.0
                            
                if not sim_valid_moves:
                    break
                    
                if np.random.rand() < 0.15:
                    smove = sim_valid_moves[np.random.choice(len(sim_valid_moves))]
                else:
                    state_t = board_to_tensor(sim_board, sim_turn).to(agent.device)
                    with torch.no_grad():
                        slogits = agent.model(state_t).squeeze(0).cpu().numpy()
                    slogits[sim_mask == 0] = -1e9
                    sexp = np.exp(slogits - np.max(slogits))
                    sprobs = sexp / np.sum(sexp)
                    sidx = np.random.choice(agent.board_size * agent.board_size, p=sprobs)
                    smove = (sidx // agent.board_size, sidx % agent.board_size)
                    
                s_r, s_c = smove
                sim_board[s_r][s_c] = sim_turn
                
                if sim_check_win(sim_board, s_r, s_c, sim_turn):
                    sim_winner = sim_turn
                    break
                    
                sim_turn = 3 - sim_turn
                sim_steps += 1
                
            if sim_winner == current_turn:
                cand_wins += 1.0
            elif sim_winner == opponent:
                cand_losses += 1.0
            else:
                cand_draws += 1.0
                
        score = cand_wins + 0.2 * cand_draws
        win_rate = (cand_wins / num_rollouts) * 100
        print(f"Candidate: Move {cand} | Win: {win_rate:.1f}% (W: {int(cand_wins)}, L: {int(cand_losses)}, D: {int(cand_draws)})")
        
        sim_score_rate = score / num_rollouts
        if sim_score_rate > best_win_rate:
            best_win_rate = sim_score_rate
            best_move = cand
            
    print(f"AI Selected: {best_move}\n")
    return best_move

class RLAgent:
    def __init__(self, board_size=13, lr=0.001):
        self.board_size = board_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = GomokuNet(board_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.saved_model_path = "/Users/chln0124/Desktop/gomoku_rl_model.pth"
        if os.path.exists(self.saved_model_path):
            try:
                self.model.load_state_dict(torch.load(self.saved_model_path, map_location=self.device))
                print("Loaded saved model weights successfully.")
            except Exception as e:
                print(f"Error loading saved weights: {e}")
                
    def get_action(self, board_state, current_turn, open3_rule, epsilon=0.0):
        # 즉시 승리/방어 자리가 있는지 최우선 확인
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
                    
        if not valid_moves:
            return None
            
        if np.random.rand() < epsilon:
            return valid_moves[np.random.choice(len(valid_moves))]
            
        self.model.eval()
        state_tensor = board_to_tensor(board_state, current_turn).to(self.device)
        with torch.no_grad():
            logits = self.model(state_tensor).squeeze(0).cpu().numpy()
            
        logits[mask == 0] = -1e9
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        idx = np.random.choice(self.board_size * self.board_size, p=probs)
        return idx // self.board_size, idx % self.board_size
        
    def train_step(self, states, actions, rewards):
        self.model.train()
        self.optimizer.zero_grad()
        loss = 0
        for state, action, G in zip(states, actions, rewards):
            state_tensor = state.to(self.device)
            logits = self.model(state_tensor).squeeze(0)
            log_probs = torch.log_softmax(logits, dim=0)
            loss += -log_probs[action] * G
        loss = loss / len(states)
        loss.backward()
        self.optimizer.step()
        return loss.item()
        
    def save_model(self):
        torch.save(self.model.state_dict(), self.saved_model_path)

# 에이전트 인스턴스화
rl_agent = RLAgent(GRID_SIZE)

def run_self_play_episode(agent, epsilon, open3_rule):
    global board
    board = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    current_turn = 1
    
    states_black, actions_black = [], []
    states_white, actions_white = [], []
    
    winner = None
    steps = 0
    max_steps = GRID_SIZE * GRID_SIZE
    
    while steps < max_steps:
        move = agent.get_action(board, current_turn, open3_rule, epsilon)
        if move is None:
            break
            
        r, c = move
        action_idx = r * GRID_SIZE + c
        state_tensor = board_to_tensor(board, current_turn)
        
        if current_turn == 1:
            states_black.append(state_tensor)
            actions_black.append(action_idx)
        else:
            states_white.append(state_tensor)
            actions_white.append(action_idx)
            
        board[r][c] = current_turn
        
        if check_win_for_color(r, c, current_turn):
            winner = current_turn
            break
            
        current_turn = 3 - current_turn
        steps += 1
        
    states, actions, rewards = [], [], []
    gamma = 0.95
    
    if winner == 1:
        R = 1.0
        for s, a in reversed(list(zip(states_black, actions_black))):
            states.append(s)
            actions.append(a)
            rewards.append(R)
            R *= gamma
        R = -1.0
        for s, a in reversed(list(zip(states_white, actions_white))):
            states.append(s)
            actions.append(a)
            rewards.append(R)
            R *= gamma
    elif winner == 2:
        R = 1.0
        for s, a in reversed(list(zip(states_white, actions_white))):
            states.append(s)
            actions.append(a)
            rewards.append(R)
            R *= gamma
        R = -1.0
        for s, a in reversed(list(zip(states_black, actions_black))):
            states.append(s)
            actions.append(a)
            rewards.append(R)
            R *= gamma
    else:
        for s, a in zip(states_black, actions_black):
            states.append(s)
            actions.append(a)
            rewards.append(-0.1)
        for s, a in zip(states_white, actions_white):
            states.append(s)
            actions.append(a)
            rewards.append(-0.1)
            
    loss = 0.0
    if states:
        loss = agent.train_step(states, actions, rewards)
        
    return winner, loss

# 학습 통계
train_episodes = 0
train_losses = []
win_history = []

# 메인 루프
running = True
winner = None
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: 
            running = False
        
        if event.type == pygame.KEYDOWN:
            if state == "MENU_MODE":
                if event.key == pygame.K_1: mode, state = "2P", "OPEN_3"
                elif event.key == pygame.K_2: mode, state = "AI", "DIFFICULTY_MODE"
                elif event.key == pygame.K_3: mode, state = "TRAIN", "OPEN_3"
                elif event.key == pygame.K_4:
                    if os.path.exists(rl_agent.saved_model_path):
                        try:
                            os.remove(rl_agent.saved_model_path)
                            print("Model deleted.")
                        except Exception as e:
                            print(f"Error resetting model: {e}")
                    rl_agent = RLAgent(GRID_SIZE)
                    print("AI Model Reset Completed.")
            elif state == "DIFFICULTY_MODE":
                if event.key == pygame.K_1:
                    difficulty = "Easy"
                    state = "OPEN_3"
                elif event.key == pygame.K_2:
                    difficulty = "Medium"
                    state = "OPEN_3"
                elif event.key == pygame.K_3:
                    difficulty = "Hard"
                    state = "OPEN_3"
                elif event.key == pygame.K_ESCAPE:
                    state = "MENU_MODE"
            elif state == "OPEN_3":
                if event.key == pygame.K_1: 
                    rule_open3 = True
                    state = "GAME" if mode != "TRAIN" else "TRAIN_MODE"
                    board = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
                    winner = None
                    turn = 1
                elif event.key == pygame.K_2: 
                    rule_open3 = False
                    state = "GAME" if mode != "TRAIN" else "TRAIN_MODE"
                    board = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
                    winner = None
                    turn = 1
            elif state == "TRAIN_MODE":
                if event.key == pygame.K_ESCAPE:
                    rl_agent.save_model()
                    print("Training stopped. Saved model.")
                    state = "MENU_MODE"
            elif state == "GAME":
                if event.key == pygame.K_ESCAPE:
                    state = "MENU_MODE"

        if state == "GAME" and event.type == pygame.MOUSEBUTTONDOWN and not winner:
            if mode == "2P" or (mode == "AI" and turn == 1):
                x, y = event.pos
                c, r = round(x / CELL_SIZE) - 1, round(y / CELL_SIZE) - 1
                if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and board[r][c] == 0:
                    print(f"Click at ({r}, {c}), turn={turn}, rule_open3={rule_open3}")
                    forbidden = is_forbidden(r, c, turn)
                    print(f"is_forbidden={forbidden}")
                    if not rule_open3 and turn == 1 and forbidden:
                        # Determine exact forbidden reason for visual feedback
                        board[r][c] = turn
                        is_ov = check_overline(r, c, turn)
                        board[r][c] = 0
                        fours = count_fours_created(r, c, turn)
                        if is_ov:
                            forbidden_msg = "Forbidden: Overline (6+)!"
                        elif fours >= 2:
                            forbidden_msg = "Forbidden: Double-Four (44)!"
                        else:
                            forbidden_msg = "Forbidden: Double-Three (33)!"
                        forbidden_time = pygame.time.get_ticks()
                        print(f"Forbidden Move: {forbidden_msg}")
                        continue
                    board[r][c] = turn
                    if check_win_for_color(r, c, turn): 
                        winner = turn
                    else: 
                        turn = 3 - turn

    # AI Turn in Game Mode
    if state == "GAME" and mode == "AI" and turn == 2 and not winner:
        pygame.time.delay(300) # 시각적인 딜레이를 주어 AI 수 확인을 편하게 함
        if difficulty == "Hard":
            # 화면에 "AI가 수읽기 중..." 텍스트 그리기
            draw_board()
            draw_text("AI is reading ahead...", SCREEN_SIZE // 2, (220, 0, 0))
            pygame.display.flip()
            
            # 실시간 수읽기 시뮬레이션 실행
            move = get_action_mcts(rl_agent, board, turn, rule_open3, num_rollouts=15, top_k=5)
        else:
            epsilon_val = 0.35 if difficulty == "Easy" else 0.15 if difficulty == "Medium" else 0.0
            move = rl_agent.get_action(board, turn, rule_open3, epsilon=epsilon_val)
            
        if move is not None:
            r, c = move
            board[r][c] = turn
            if check_win_for_color(r, c, turn):
                winner = turn
            else:
                turn = 3 - turn

    # RL Self-play Training Step
    if state == "TRAIN_MODE":
        # 훈련 진행 (탐색율 epsilon=0.2로 설정해 다양한 형태 탐색)
        episode_winner, loss = run_self_play_episode(rl_agent, epsilon=0.2, open3_rule=rule_open3)
        train_episodes += 1
        train_losses.append(loss)
        win_history.append(episode_winner if episode_winner is not None else 0)
        
        # 마지막 100 경기 통계
        recent = win_history[-100:]
        total_recent = len(recent)
        if total_recent > 0:
            black_rate = recent.count(1) / total_recent * 100
            white_rate = recent.count(2) / total_recent * 100
            draw_rate = recent.count(0) / total_recent * 100
        else:
            black_rate, white_rate, draw_rate = 0, 0, 0

    # 렌더링
    if state == "MENU_MODE":
        screen.fill(WHITE)
        draw_text("Omok 13x13 Game System", 100, (0, 0, 150))
        draw_text("1. Local 1 vs 1 Game", 220)
        draw_text("2. Player vs AI (RL)", 280)
        draw_text("3. Train AI (Self-Play RL)", 340)
        draw_text("4. Reset AI Model Weight", 400, (150, 0, 0))
        draw_text("Select option with keyboard", 500, (100, 100, 100))
    elif state == "DIFFICULTY_MODE":
        screen.fill(WHITE)
        draw_text("Select AI Difficulty Level", 180, (0, 0, 150))
        draw_text("1. Easy (쉬움)", 260)
        draw_text("2. Medium (보통)", 320)
        draw_text("3. Hard (어려움)", 380)
        draw_text("Press 1, 2, or 3", 470, (100, 100, 100))
    elif state == "OPEN_3":
        screen.fill(WHITE)
        draw_text("Double-Three (33) Rule Setting", 180, (0, 100, 0))
        draw_text("1. Allow Double-Three (Standard)", 280)
        draw_text("2. Prohibit Double-Three (Renju)", 340)
        draw_text("Press 1 or 2", 440, (100, 100, 100))
    elif state == "GAME":
        draw_board()
        if winner: 
            draw_text(f"Player {winner} ({'Black' if winner==1 else 'White'}) Wins!", 30, (200, 0, 0))
            draw_text("Press ESC to Menu", SCREEN_SIZE - 30, (100, 100, 100))
        else:
            draw_text(f"Current Turn: {'Black' if turn==1 else 'White'}", 30, (0, 0, 200) if turn==1 else (100, 100, 100))
            
            # AI 모드일 때 난이도 화면에 표시
            if mode == "AI":
                draw_text(f"Difficulty: {difficulty}", 57, (120, 120, 120))
                
            # 화면에 금수 에러 메시지 렌더링 (2초간 유지)
            if forbidden_msg and pygame.time.get_ticks() - forbidden_time < 2000:
                draw_text(forbidden_msg, SCREEN_SIZE - 30, (220, 0, 0))
            else:
                forbidden_msg = None
    elif state == "TRAIN_MODE":
        draw_board()
        # 반투명 오버레이
        overlay = pygame.Surface((340, 240))
        overlay.set_alpha(230)
        overlay.fill(WHITE)
        screen.blit(overlay, (130, 180))
        
        train_font = pygame.font.SysFont("Arial", 18)
        def draw_train_text(text, y, color=BLACK):
            surf = train_font.render(text, True, color)
            rect = surf.get_rect(center=(SCREEN_SIZE/2, y))
            screen.blit(surf, rect)
            
        draw_train_text("AI Self-Play Training in Progress...", 200, (220, 0, 0))
        draw_train_text(f"Episode: {train_episodes}", 230)
        draw_train_text(f"Current Loss: {train_losses[-1]:.4f}" if train_losses else "Current Loss: N/A", 250)
        draw_train_text(f"Black Wins (Last 100): {black_rate:.1f}%", 285)
        draw_train_text(f"White Wins (Last 100): {white_rate:.1f}%", 310)
        draw_train_text(f"Draws (Last 100): {draw_rate:.1f}%", 335)
        draw_train_text("Press ESC to Stop and Save Weights", 380, (100, 100, 100))

    pygame.display.flip()

pygame.quit()
sys.exit()