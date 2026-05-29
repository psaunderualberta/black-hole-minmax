package main;

type Move struct {
	row int
	col int
}

func NewMove(row, col int) Move {
	return Move{row, col};
}
