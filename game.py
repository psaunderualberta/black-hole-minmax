import equinox as eqx   
import jax
import jax.numpy as jnp



class BlackHole(eqx.Module):
    board: chex.Array
    player_1_turn: bool
    player_1_count: int
    player_2_count: int


    def __init__(self, board: chex.Array, player_1_turn: bool =True, player_1_count: int = 0, player_2_count: int = 0):
        self.board = board
        self.player_1_turn = player_1_turn

    def print_board(self):
        for row in self.board:
            jax.debug.print("{}", row)

    def play_move(self, move: tuple[int, int]):
        pass


    
