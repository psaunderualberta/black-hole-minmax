package main;

type Game struct {
	Board [][]int
	p1_turn bool
}

func InitializeGame(shape int) Game {
	board := make([][]int, shape);
	for i := 1; i <= shape; i++ {
		board[i-1] = make([]int, i);
	}
	return Game{board, true};
}
