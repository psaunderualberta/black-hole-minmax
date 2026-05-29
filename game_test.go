package main

import (
	"testing"
)

func TestNewGameIsPlayerOneTurn(t *testing.T) {
	game := InitializeGame(4)
	if !game.IsP1Turn {
		t.Error("Game does not start on P1's turn")
	}
}

func TestNewGameHasShapeRows(t *testing.T) {
	for _, shape := range []int{1, 3, 6} {
		game := InitializeGame(shape)
		if len(game.Board) != shape {
			t.Errorf("Expected %d rows, got %d", shape, len(game.Board))
		}
	}
}

func TestNewGameBoardIsTriangular(t *testing.T) {
	for _, shape := range []int{1, 3, 6} {
		game := InitializeGame(shape)
		for i, row := range game.Board {
			if len(row) != i+1 {
				t.Errorf("shape %d, row %d: expected %d cells, found %d", shape, i, i+1, len(row))
			}
		}
	}
}

func TestNewGameInitsEmpty(t *testing.T) {
	for _, shape := range []int{1, 3, 6} {
		game := InitializeGame(shape)
		for i, row := range game.Board {
			for j, cell := range row {
				if cell != 0 {
					t.Errorf("shape %d, row %d, cell %d: Expected 0, found %d", shape, i, j, cell)
				}
			}
		}
	}
}

/* Testing ApplyMove
 * Each move changes only the cell at 'move'
 * The cell is set to the 'count' value
 * Count attribute increases by 1
 * IsP1Turn Flips
 */

func TestIsP1TurnFlips(t *testing.T) {
	for _, starting_turn := range []bool{true, false} {
		game := Game{
			[][]int{{0}, {0, 0}},
			starting_turn,
			0,
		}

		game.ApplyMove(Move{0, 0})
		if game.IsP1Turn != !starting_turn {
			t.Errorf("starting turn %v: expected %v, found %v", starting_turn, !starting_turn, starting_turn)
		}
	}
}

func TestCountIncreasesByOne(t *testing.T) {
	for _, starting_count := range []int{0, 1, 2} {
		game := Game{
			[][]int{{0}, {0, 0}},
			false,
			starting_count,
		}

		game.ApplyMove(Move{0, 0})
		if game.Count != starting_count+1 {
			t.Errorf("starting_count %d: found %d, expected %d", starting_count, starting_count+1, game.Count)
		}
	}
}

func TestCellIsSetToCountValue(t *testing.T) {
	moves := []Move{Move{0, 0}, Move{1, 0}}
	for _, move := range moves {
		for _, starting_count := range []int{0, 1, 2} {
			game := Game{
				[][]int{{0}, {0, 0}},
				false,
				starting_count,
			}

			game.ApplyMove(move)
			if game.Board[move.Row][move.Col] != starting_count {
				t.Errorf("starting_count %d: found %d, expected %d", starting_count, starting_count, game.Board[move.Row][move.Col])
			}
		}
	}
}
