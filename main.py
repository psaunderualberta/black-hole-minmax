from game import BlackHole
import jax.numpy as jnp


def minmax_search(depth, game):
    return jax.lax.cond(
        game.is_done(),
    )


def main():
    game = BlackHole(jnp.zeros((7, 7), dtype=jnp.int32))
    game = game.play_move([0, 0])
    game = game.play_move([1, 1])
    for idx in game.idxs():
        if game.is_valid_move(*idx):
            print(idx)

    game.print_board()


if __name__ == "__main__":
    main()
