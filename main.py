from game import BlackHole
import numpy as np
from numba import njit
import time


@njit
def minmax_search(game, depth, alpha, beta):
    if game.is_done():
        value = game.get_score(game.player_1_turn)
        return value, None
    elif depth == 0:
        return 0, None
    best_value = -float("inf")
    best_move = game.get_valid_moves()[0]

    for idx in game.get_valid_moves():
        new_game = game.play_move(idx)
        (value, _) = minmax_search(new_game, depth - 1, -beta, -alpha)

        alpha = max(alpha, value)
        if alpha >= beta:
            break

        if value >= best_value:
            best_move = idx
            best_value = value

    return best_value, best_move


def main():
    game = BlackHole(np.zeros((5, 5), dtype=np.int32))

    for idx in game.idxs():
        if game.is_valid_move(*idx):
            print(idx)

    game.print_board()
    print(game.is_done())

    start = time.time()
    print(minmax_search(game, 300, -float("inf"), float("inf")))
    end = time.time()
    print("{:.3f}".format(end - start))


if __name__ == "__main__":
    main()
