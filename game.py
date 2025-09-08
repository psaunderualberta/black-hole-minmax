import equinox as eqx
import jax
import jax.numpy as jnp
import chex


class BlackHole(eqx.Module):
    board: chex.Array
    player_1_turn: bool
    count: int

    def __init__(self, board: chex.Array, player_1_turn: bool = True, count: int = 1):
        self.board = board
        self.count = count
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
        valid_x = jnp.logical_and(0 <= x, x <= self.board.shape[0])
        valid_y = jnp.logical_and(0 <= y, y <= self.board.shape[0])
        return jnp.logical_and(x >= y, jnp.logical_and(valid_x, valid_y))

    def is_valid_move(self, x, y):
        return jnp.logical_and(self.is_valid_idx(x, y), self.board[x][y] == 0)

    def idxs(self):
        rows, cols = jnp.indices(self.board.shape)
        return jnp.vstack([rows.ravel(), cols.ravel()]).T

    def play_move(self, move: tuple[int, int]) -> "BlackHole":
        x, y = move[0], move[1]

        board = self.board.at[x, y].set(self.count * (2 * self.player_1_turn - 1))
        new_player_1_turn = jnp.logical_not(self.player_1_turn)
        new_count = self.count + new_player_1_turn.astype(jnp.int32)
        new_game = BlackHole(board, new_player_1_turn, new_count)

        return new_game

    def get_valid_moves(self) -> chex.Array:
        return jnp.argwhere(self.board == 0, size=7 * 7, fill_value=-1)

    def is_done(self) -> bool:
        playing_area = jnp.tril(self.board)
        return (playing_area == 0).sum() == 1

    def get_score(self, player_1):
        directions = jnp.asarray(
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
        nonzero_entry = jnp.argwhere(self.board == 0, size=1)
        nz_x, nz_y = nonzero_entry[0][0], nonzero_entry[0][1]

        def add_score(carry, direction):
            x, y = nz_x + direction[0], nz_y + direction[1]
            p1_score, p2_score = carry[0], carry[1]

            value = jax.lax.cond(
                self.is_valid_idx(x, y), lambda _, __: 0, self.get_pos, x, y
            )

            p1_score += jax.lax.select(value > 0, value, 0)
            p2_score += jax.lax.select(value < 0, -value, 0)

            return (p1_score, p2_score), None

        (p1_count, p2_count), _ = jax.lax.scan(add_score, (0, 0), xs=directions)

        return jax.lax.select(player_1, p1_count - p2_count, p2_count - p1_count)
