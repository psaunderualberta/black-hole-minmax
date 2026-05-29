package main

type Move struct {
	Row int
	Col int
}

func NewMove(row, col int) Move {
	return Move{row, col}
}
