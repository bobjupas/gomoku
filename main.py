# -*- coding: utf-8 -*-
"""五子棋雙人對戰 + Minimax(Alpha-Beta) AI（單檔案版本，Tkinter UI）"""
import tkinter as tk
from tkinter import messagebox

SIZE = 15          # 15x15 棋盤
CELL = 40          # 每格像素
MARGIN = 30        # 邊距
EMPTY, BLACK, WHITE = 0, 1, 2

# 簡單權重：連子數 -> 分數（開放端越多分數越高）
SCORE = {
    5: 1000000,   # 五連
    4: 10000,     # 活四/衝四
    3: 1000,      # 活三
    2: 100,       # 活二
    1: 10,
}
OPEN_BONUS = {0: 1, 1: 1, 2: 5}  # 兩端開放 -> x5，一端 -> x1，被封死 -> 0


def in_board(r, c):
    return 0 <= r < SIZE and 0 <= c < SIZE


def count_line(board, r, c, dr, dc, player):
    """從 (r,c) 沿方向數連子數與開放端數"""
    count, open_ends = 1, 0
    for sign in (1, -1):
        rr, cc = r + dr * sign, c + dc * sign
        while in_board(rr, cc) and board[rr][cc] == player:
            count += 1
            rr += dr * sign
            cc += dc * sign
        if in_board(rr, cc) and board[rr][cc] == EMPTY:
            open_ends += 1
    return count, open_ends


def line_score(board, r, c, player):
    """假設 player 落子在 (r,c)，估算四個方向的總分"""
    total = 0
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count, open_ends = count_line(board, r, c, dr, dc, player)
        if count >= 5:
            total += SCORE[5]
        elif open_ends == 0:
            continue  # 被封死沒價值
        else:
            total += SCORE.get(count, 0) * OPEN_BONUS[open_ends]
    return total


# ---------------- Minimax + Alpha-Beta 剪枝搜尋 ----------------
INF = 10 ** 9
WIN = SCORE[5]
DEPTH = 4          # 搜尋深度（4 層 = 我下、對方回、我再下、對方再回）
BRANCH_ROOT = 14   # 根節點保留的候選點數
BRANCH_INNER = 10  # 內部節點保留的候選點數


def _build_lines():
    """預先建立所有長度 >= 5 的橫、直、斜線，供盤面評分與增量更新使用"""
    lines = []
    for r in range(SIZE):
        lines.append([(r, c) for c in range(SIZE)])          # 橫
    for c in range(SIZE):
        lines.append([(r, c) for r in range(SIZE)])          # 直
    for s in range(SIZE):                                    # 斜 ↘
        lines.append([(s + k, k) for k in range(SIZE - s)])
        if s:
            lines.append([(k, s + k) for k in range(SIZE - s)])
    for s in range(SIZE):                                    # 斜 ↗
        lines.append([(s + k, SIZE - 1 - k) for k in range(SIZE - s)])
        if s:
            lines.append([(k, SIZE - 1 - s - k) for k in range(SIZE - s)])
    lines = [ln for ln in lines if len(ln) >= 5]
    cell_lines = {}
    for idx, ln in enumerate(lines):
        for cell in ln:
            cell_lines.setdefault(cell, []).append(idx)
    return lines, cell_lines


LINES, CELL_LINES = _build_lines()


def _score_line(board, line):
    """掃描一條線上雙方的連子，回傳 [黑方分, 白方分]（沿用 SCORE 權重）"""
    n = len(line)
    vals = [board[r][c] for r, c in line]
    scores = [0, 0]
    i = 0
    while i < n:
        v = vals[i]
        if v == EMPTY:
            i += 1
            continue
        j = i
        while j < n and vals[j] == v:
            j += 1
        count = j - i
        open_ends = (1 if i > 0 and vals[i - 1] == EMPTY else 0) + \
                    (1 if j < n and vals[j] == EMPTY else 0)
        if count >= 5:
            s = SCORE[5]
        elif open_ends == 0:
            s = 0
        else:
            s = SCORE.get(count, 0) * OPEN_BONUS[open_ends]
        scores[v - 1] += s
        i = j
    return scores


class _Searcher:
    """以增量更新（只重算通過落子點的 4 條線）維護全盤分數，加速 Minimax"""

    def __init__(self, board):
        self.board = board
        self.cache = [None] * len(LINES)
        self.total = [0, 0]
        for idx, ln in enumerate(LINES):
            s = _score_line(board, ln)
            self.cache[idx] = s
            self.total[0] += s[0]
            self.total[1] += s[1]

    def _refresh(self, cell):
        for idx in CELL_LINES[cell]:
            old = self.cache[idx]
            self.total[0] -= old[0]
            self.total[1] -= old[1]
            new = _score_line(self.board, LINES[idx])
            self.cache[idx] = new
            self.total[0] += new[0]
            self.total[1] += new[1]

    def make(self, r, c, player):
        self.board[r][c] = player
        self._refresh((r, c))

    def unmake(self, r, c):
        self.board[r][c] = EMPTY
        self._refresh((r, c))

    def eval(self, player):
        """以 player 視角回傳全盤分數差"""
        return self.total[player - 1] - self.total[2 - player]

    def candidates(self, player, limit):
        """已有棋子附近 2 格內的空點，以 進攻+防守 點分排序，保留前 limit 個"""
        b = self.board
        opp = 3 - player
        scored = []
        for r in range(SIZE):
            for c in range(SIZE):
                if b[r][c] != EMPTY or not has_neighbor(b, r, c):
                    continue
                attack = line_score(b, r, c, player)
                defend = line_score(b, r, c, opp)
                scored.append((attack + defend * 0.9, attack, defend, r, c))
        scored.sort(reverse=True)
        return [(attack, defend, r, c) for _, attack, defend, r, c in scored[:limit]]

    def negamax(self, depth, alpha, beta, player):
        """Negamax + Alpha-Beta 剪枝"""
        if depth == 0:
            return self.eval(player)
        cands = self.candidates(player, BRANCH_INNER)
        if not cands:
            return 0
        best = -INF
        for attack, _defend, r, c in cands:
            if attack >= WIN:
                return WIN + depth           # 這一手直接成五，越快越好
            self.make(r, c, player)
            val = -self.negamax(depth - 1, -beta, -alpha, 3 - player)
            self.unmake(r, c)
            if val > best:
                best = val
                if best > alpha:
                    alpha = best
                    if alpha >= beta:
                        break
        return best


def ai_move(board, player):
    """Minimax(Alpha-Beta) AI：先處理必勝/必防，再做深度搜尋"""
    board = [row[:] for row in board]       # 複製一份，搜尋中會暫時落子
    searcher = _Searcher(board)
    cands = searcher.candidates(player, BRANCH_ROOT)
    if not cands:
        return (SIZE // 2, SIZE // 2)       # 空盤直接下天元
    for attack, _defend, r, c in cands:     # 1) 自己可成五 → 立即獲勝
        if attack >= WIN:
            return (r, c)
    for _attack, defend, r, c in cands:     # 2) 對方將成五 → 必須攔截
        if defend >= WIN:
            return (r, c)
    best_move, best_val, alpha = (cands[0][2], cands[0][3]), -INF, -INF
    for _attack, _defend, r, c in cands:    # 3) Alpha-Beta 深度搜尋
        searcher.make(r, c, player)
        val = -searcher.negamax(DEPTH - 1, -INF, -alpha, 3 - player)
        searcher.unmake(r, c)
        if val > best_val:
            best_val, best_move = val, (r, c)
        if val > alpha:
            alpha = val
    return best_move


def has_neighbor(board, r, c, dist=2):
    for dr in range(-dist, dist + 1):
        for dc in range(-dist, dist + 1):
            rr, cc = r + dr, c + dc
            if (dr or dc) and in_board(rr, cc) and board[rr][cc] != EMPTY:
                return True
    return False


def check_win(board, r, c):
    """落子後檢查 (r,c) 是否連成五子，回傳獲勝玩家或 0"""
    player = board[r][c]
    if player == EMPTY:
        return 0
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        for sign in (1, -1):
            rr, cc = r + dr * sign, c + dc * sign
            while in_board(rr, cc) and board[rr][cc] == player:
                count += 1
                rr += dr * sign
                cc += dc * sign
        if count >= 5:
            return player
    return 0


class GomokuUI:
    def __init__(self, root):
        self.root = root
        root.title("五子棋 - 雙人對戰 / 基礎AI")
        self.mode = tk.StringVar(value="pvp")   # pvp / pve
        self.vs_ai = tk.StringVar(value="black")  # AI 執黑或執白

        top = tk.Frame(root)
        top.pack(pady=5)
        tk.Label(top, text="模式：").pack(side=tk.LEFT)
        tk.Radiobutton(top, text="雙人對戰", variable=self.mode,
                       value="pvp", command=self.reset).pack(side=tk.LEFT)
        tk.Radiobutton(top, text="人機對戰", variable=self.mode,
                       value="pve", command=self.reset).pack(side=tk.LEFT)
        tk.Label(top, text="   AI 執子：").pack(side=tk.LEFT)
        tk.Radiobutton(top, text="黑", variable=self.vs_ai,
                       value="black", command=self.reset).pack(side=tk.LEFT)
        tk.Radiobutton(top, text="白", variable=self.vs_ai,
                       value="white", command=self.reset).pack(side=tk.LEFT)
        tk.Button(top, text="重新開始", command=self.reset).pack(side=tk.LEFT, padx=10)

        self.status = tk.Label(root, text="", font=("Arial", 12))
        self.status.pack()

        w = MARGIN * 2 + CELL * (SIZE - 1)
        self.canvas = tk.Canvas(root, width=w, height=w, bg="#DEB887")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)
        self.reset()

    def reset(self):
        self.board = [[EMPTY] * SIZE for _ in range(SIZE)]
        self.current = BLACK
        self.game_over = False
        self.history = []
        self.ai_thinking = False
        self.status.config(text="黑方落子")
        self.draw()
        if self.mode.get() == "pve" and self.ai_side() == BLACK:
            self.root.after(200, self.ai_turn)

    def ai_side(self):
        return BLACK if self.vs_ai.get() == "black" else WHITE

    def draw(self):
        cv = self.canvas
        cv.delete("all")
        n = SIZE - 1
        for i in range(SIZE):
            cv.create_line(MARGIN, MARGIN + i * CELL, MARGIN + n * CELL, MARGIN + i * CELL)
            cv.create_line(MARGIN + i * CELL, MARGIN, MARGIN + i * CELL, MARGIN + n * CELL)
        for r in range(SIZE):
            for c in range(SIZE):
                if self.board[r][c] != EMPTY:
                    self.draw_stone(r, c, self.board[r][c])

    def draw_stone(self, r, c, player):
        x, y = MARGIN + c * CELL, MARGIN + r * CELL
        color = "black" if player == BLACK else "white"
        self.canvas.create_oval(x - 16, y - 16, x + 16, y + 16,
                                fill=color, outline="black")

    def on_click(self, event):
        if self.game_over or self.ai_thinking:
            return
        if self.mode.get() == "pve" and self.current == self.ai_side():
            return  # 輪到 AI，忽略點擊
        c = round((event.x - MARGIN) / CELL)
        r = round((event.y - MARGIN) / CELL)
        if not in_board(r, c) or self.board[r][c] != EMPTY:
            return
        self.place(r, c)
        if self.mode.get() == "pve" and not self.game_over:
            self.root.after(200, self.ai_turn)

    def place(self, r, c):
        self.board[r][c] = self.current
        self.history.append((r, c))
        self.draw()
        winner = check_win(self.board, r, c)
        if winner:
            self.game_over = True
            name = "黑方" if winner == BLACK else "白方"
            self.status.config(text=f"{name} 獲勝！")
            messagebox.showinfo("遊戲結束", f"{name} 獲勝！")
        elif len(self.history) == SIZE * SIZE:
            self.game_over = True
            self.status.config(text="平局")
            messagebox.showinfo("遊戲結束", "平局！")
        else:
            self.current = 3 - self.current
            self.status.config(text="黑方落子" if self.current == BLACK else "白方落子")

    def ai_turn(self):
        if self.game_over:
            return
        self.ai_thinking = True
        self.status.config(text="AI 思考中...")
        self.root.update_idletasks()
        r, c = ai_move(self.board, self.current)
        self.ai_thinking = False
        self.place(r, c)


if __name__ == "__main__":
    root = tk.Tk()
    GomokuUI(root)
    root.mainloop()
