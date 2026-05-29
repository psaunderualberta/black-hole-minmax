package main

type Game struct {
	Board    [][]int
	IsP1Turn bool
	Count    int
}

func InitializeGame(shape int) Game {
	board := make([][]int, shape)

	for i := 1; i <= shape; i++ {
		board[i-1] = make([]int, i)
	}

	return Game{board, true, 1}
}

func (game *Game) ApplyMove(move Move) {
	game.IsP1Turn = !game.IsP1Turn
	game.Board[move.Row][move.Col] = game.Count
	game.Count++
}
