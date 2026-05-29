# Game
- Board: [][]i32
- p1_turn: bool
+ initialize(board_size);
+ IsTerminal() -> bool;
+ StateValue() -> i32;
+ ApplyMove(move: Move);
+ UndoMove(move: Move);

# Move
- Row: i32
- Col: i32
+
