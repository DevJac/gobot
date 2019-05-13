use BoardPosition::{Empty, Black, White};

#[derive(Clone, Copy, Debug)]
enum BoardPosition {
    Empty,
    Black,
    White,
}

impl BoardPosition {
    fn other(self) -> Self {
        match self {
            Empty => Empty,
            Black => White,
            White => Black,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Point {
    x: u8,
    y: u8,
}

impl Point {
    // TODO: neighbors
}

#[derive(Debug)]
struct Board {
    size: u8,
    board: Vec<BoardPosition>,
    liberties: Vec<BoardPosition>,
    history: Vec<Board>,
}

impl Board {
    // TODO
}
