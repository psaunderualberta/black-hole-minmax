import numpy as np
from numba import int32, boolean
from numba.experimental import jitclass


spec = [
    ("board", int32[:, :]),
    ("player_1_turn", boolean),
    ("score", int32),
]


@jitclass(spec)
class BlackHole(object):
    board: np.ndarray
    player_1_turn: bool
    score: int

    def __init__(self, board: np.ndarray, player_1_turn: bool = True, score: int = 1):
        self.board = board
        self.score = score
        self.player_1_turn = player_1_turn

    def print_board(self):
        sep_len = 2
        sep_str = " " * sep_len
        row_length = len(sep_str.join(map(lambda num: str(num), self.board[-1])))
        for i, row in enumerate(self.board):
            row = self.board[i][: i + 1]
            current_row_s = sep_str.join(map(lambda num: str(num), row))
            fill_width = (row_length - len(current_row_s)) // 2
            printable_str = current_row_s.ljust(row_length - fill_width)
            printable_str = printable_str.rjust(row_length)
            print(printable_str)

    def get_pos(self, x, y):
        return self.board[x][y]

    def is_valid_idx(self, x, y):
        x = int(x)
        y = int(y)
        valid_x = (0 <= x) & (x < self.board.shape[0])
        valid_y = (0 <= y) & (y < self.board.shape[0])
        return (x >= y) & (valid_x) & (valid_y)

    def is_valid_move(self, x, y):
        x = int(x)
        y = int(y)
        return (self.is_valid_idx(x, y)) & (self.board[x][y] == 0)

    def idxs(self):
        rows, cols = np.indices(self.board.shape)
        idxs = np.zeros((2, self.board.shape[0] * self.board.shape[1]))
        idxs[0, :] = rows.ravel()
        idxs[1, :] = cols.ravel()
        return idxs.T

    def play_move(self, move: tuple[int, int]) -> "BlackHole":
        x, y = move[0], move[1]
        x = int(x)
        y = int(y)

        board = np.copy(self.board)
        board[x][y] = self.score if self.player_1_turn else -self.score
        new_player_1_turn = not self.player_1_turn
        new_score = self.score + int(new_player_1_turn)
        new_game = BlackHole(board, new_player_1_turn, new_score)

        return new_game

    def get_valid_moves(self) -> np.ndarray:
        rows, cols = np.indices(self.board.shape)
        return np.argwhere((self.board == 0) & (cols <= rows))

    def is_done(self) -> bool:
        playing_area = np.tril(self.board) + np.triu(self.board + 1, k=1)
        return (playing_area == 0).sum() == 1

    def get_score(self, player_1):
        directions = np.array(
            [
                [-1, -1],
                [-1, 0],
                [0, 1],
                [1, 1],
                [1, 0],
                [0, -1],
            ]
        )

        # assume game is done
        if not self.is_done():
            return 0
        rows, cols = np.indices(self.board.shape)
        nonzero_entry = np.argwhere((self.board == 0) & (cols <= rows))
        nz_x, nz_y = nonzero_entry[0][0], nonzero_entry[0][1]

        score = 0
        for direction in directions:
            x, y = nz_x + direction[0], nz_y + direction[1]

            if self.is_valid_idx(x, y):
                score += self.get_pos(x, y)

        return score if player_1 else -score
