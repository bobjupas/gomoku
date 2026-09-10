# -*- coding: utf-8 -*-
"""五子棋 Kivy 版（Android APK 入口）
雙人對戰 / 人機對戰，AI 為 Minimax + Alpha-Beta（與桌面版同一套演算法）
"""
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.popup import Popup
from kivy.graphics import Color, Line, Ellipse, Rectangle
from kivy.clock import Clock
from kivy.core.window import Window

SIZE = 15
EMPTY, BLACK, WHITE = 0, 1, 2

# ---------------- AI：Minimax + Alpha-Beta ----------------
SCORE = {5: 1000000, 4: 10000, 3: 1000, 2: 100, 1: 10}
OPEN_BONUS = {0: 1, 1: 1, 2: 5}
INF = 10 ** 9
WIN = SCORE[5]
DEPTH = 4
BRANCH_ROOT = 14
BRANCH_INNER = 10


def in_board(r, c):
    return 0 <= r < SIZE and 0 <= c < SIZE


def count_line(board, r, c, dr, dc, player):
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
    total = 0
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count, open_ends = count_line(board, r, c, dr, dc, player)
        if count >= 5:
            total += SCORE[5]
        elif open_ends == 0:
            continue
        else:
            total += SCORE.get(count, 0) * OPEN_BONUS[open_ends]
    return total


def has_neighbor(board, r, c, dist=2):
    for dr in range(-dist, dist + 1):
        for dc in range(-dist, dist + 1):
            rr, cc = r + dr, c + dc
            if (dr or dc) and in_board(rr, cc) and board[rr][cc] != EMPTY:
                return True
    return False


def _build_lines():
    lines = []
    for r in range(SIZE):
        lines.append([(r, c) for c in range(SIZE)])
    for c in range(SIZE):
        lines.append([(r, c) for r in range(SIZE)])
    for s in range(SIZE):
        lines.append([(s + k, k) for k in range(SIZE - s)])
        if s:
            lines.append([(k, s + k) for k in range(SIZE - s)])
    for s in range(SIZE):
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
        return self.total[player - 1] - self.total[2 - player]

    def candidates(self, player, limit):
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
        if depth == 0:
            return self.eval(player)
        cands = self.candidates(player, BRANCH_INNER)
        if not cands:
            return 0
        best = -INF
        for attack, _defend, r, c in cands:
            if attack >= WIN:
                return WIN + depth
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
    board = [row[:] for row in board]
    searcher = _Searcher(board)
    cands = searcher.candidates(player, BRANCH_ROOT)
    if not cands:
        return (SIZE // 2, SIZE // 2)
    for attack, _defend, r, c in cands:
        if attack >= WIN:
            return (r, c)
    for _attack, defend, r, c in cands:
        if defend >= WIN:
            return (r, c)
    best_move, best_val, alpha = (cands[0][2], cands[0][3]), -INF, -INF
    for _attack, _defend, r, c in cands:
        searcher.make(r, c, player)
        val = -searcher.negamax(DEPTH - 1, -INF, -alpha, 3 - player)
        searcher.unmake(r, c)
        if val > best_val:
            best_val, best_move = val, (r, c)
        if val > alpha:
            alpha = val
    return best_move


def check_win(board, r, c):
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


# ---------------- UI ----------------
class BoardWidget(Widget):
    """棋盤畫布：Kivy Canvas 繪製 + 觸摸落子"""

    def __init__(self, game, **kw):
        super().__init__(**kw)
        self.game = game
        self.bind(size=self._redraw, pos=self._redraw)

    def _cell_px(self):
        return min(self.width, self.height) / (SIZE + 1)

    def to_cell(self, x, y):
        cell = self._cell_px()
        c = int(round((x - self.x - cell) / cell))
        r = int(round((self.y + self.height - cell - y) / cell))
        return (r, c) if in_board(r, c) else None

    def cell_center(self, r, c):
        cell = self._cell_px()
        x = self.x + cell + c * cell
        y = self.y + self.height - cell - r * cell
        return x, y

    def _redraw(self, *args):
        self.canvas.clear()
        cell = self._cell_px()
        with self.canvas:
            Color(0.87, 0.73, 0.53, 1)
            Rectangle(pos=self.pos, size=self.size)
            Color(0.2, 0.1, 0, 1)
            for i in range(SIZE):
                x1, y1 = self.cell_center(i, 0)
                x2, y2 = self.cell_center(i, SIZE - 1)
                Line(points=[x1, y1, x2, y2], width=1)
                x1, y1 = self.cell_center(0, i)
                x2, y2 = self.cell_center(SIZE - 1, i)
                Line(points=[x1, y1, x2, y2], width=1)
            for r in range(SIZE):
                for c in range(SIZE):
                    v = self.game.board[r][c]
                    if v != EMPTY:
                        x, y = self.cell_center(r, c)
                        rad = cell * 0.4
                        if v == BLACK:
                            Color(0, 0, 0, 1)
                        else:
                            Color(1, 1, 1, 1)
                        Ellipse(pos=(x - rad, y - rad), size=(rad * 2, rad * 2))
                        Color(0, 0, 0, 1)
                        Line(circle=[x, y, rad], width=1)

    def on_touch_down(self, touch):
        if self.game.game_over or self.game.ai_thinking:
            return True
        if self.game.mode == 'pve' and self.game.current == self.game.ai_side():
            return True
        pos = self.to_cell(touch.x, touch.y)
        if pos and self.game.board[pos[0]][pos[1]] == EMPTY:
            self.game.place(*pos)
        return True


class GomokuGame(BoxLayout):
    """頂部控制列 + 棋盤"""

    def __init__(self, **kw):
        super().__init__(orientation='vertical', **kw)
        self.mode = 'pve'
        self.ai_side = WHITE
        self.reset_game()
        self.build_ui()

    def build_ui(self):
        bar = GridLayout(cols=4, size_hint_y=None, height='48dp', padding=4, spacing=4)
        self.btn_mode = Button(text='模式：人機', on_press=self.toggle_mode)
        self.btn_side = Button(text='AI執白', on_press=self.toggle_side)
        btn_reset = Button(text='重新開始', on_press=lambda *_: self.reset_game())
        bar.add_widget(self.btn_mode)
        bar.add_widget(self.btn_side)
        bar.add_widget(btn_reset)
        self.status = Label(text='黑方落子', size_hint_y=None, height='32dp')
        self.add_widget(bar)
        self.add_widget(self.status)
        self.board_w = BoardWidget(self)
        self.add_widget(self.board_w)

    # -- 控制 --
    def toggle_mode(self, *_):
        self.mode = 'pvp' if self.mode == 'pve' else 'pve'
        self.btn_mode.text = '模式：人機' if self.mode == 'pve' else '模式：雙人'
        self.reset_game()

    def toggle_side(self, *_):
        self.ai_side = BLACK if self.ai_side == WHITE else WHITE
        self.btn_side.text = 'AI執黑' if self.ai_side == BLACK else 'AI執白'
        self.reset_game()

    def reset_game(self, *_):
        self.board = [[EMPTY] * SIZE for _ in range(SIZE)]
        self.current = BLACK
        self.game_over = False
        self.ai_thinking = False
        if hasattr(self, 'board_w'):
            self.board_w._redraw()
            self.update_status()
            if self.mode == 'pve' and self.ai_side == BLACK:
                Clock.schedule_once(lambda *_: self.ai_turn(), 0.3)

    # -- 遊戲流程 --
    def place(self, r, c):
        self.board[r][c] = self.current
        self.board_w._redraw()
        winner = check_win(self.board, r, c)
        if winner:
            self.game_over = True
            name = '黑方' if winner == BLACK else '白方'
            self.status.text = f'{name} 獲勝！'
            self.show_popup('遊戲結束', f'{name} 獲勝！')
        elif all(v != EMPTY for row in self.board for v in row):
            self.game_over = True
            self.status.text = '平局'
            self.show_popup('遊戲結束', '平局！')
        else:
            self.current = 3 - self.current
            self.update_status()
            if self.mode == 'pve' and self.current == self.ai_side():
                Clock.schedule_once(lambda *_: self.ai_turn(), 0.2)

    def ai_turn(self):
        if self.game_over:
            return
        self.ai_thinking = True
        self.status.text = 'AI 思考中...'
        Clock.schedule_once(lambda *_: self._ai_apply(), 0.05)

    def _ai_apply(self):
        r, c = ai_move(self.board, self.current)
        self.ai_thinking = False
        self.place(r, c)

    def update_status(self):
        if not self.game_over:
            self.status.text = '黑方落子' if self.current == BLACK else '白方落子'

    def show_popup(self, title, msg):
        box = BoxLayout(orientation='vertical', padding=8)
        box.add_widget(Label(text=msg))
        btn = Button(text='確定', size_hint_y=None, height='44dp')
        box.add_widget(btn)
        pop = Popup(title=title, content=box, size_hint=(0.7, 0.4))
        btn.bind(on_press=pop.dismiss)
        pop.open()


class GomokuApp(App):
    title = '五子棋'

    def build(self):
        Window.clearcolor = (0.95, 0.93, 0.9, 1)
        return GomokuGame()


if __name__ == '__main__':
    GomokuApp().run()
