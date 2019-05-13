#![warn(
    warnings,
    future_incompatible,
    nonstandard_style,
    rust_2018_compatibility,
    rust_2018_idioms,
    rustdoc,
    unused
)]
#![allow(dead_code, non_snake_case)] // TODO: Remove

use BoardPosition::{Black, Empty, White};
use Direction::{Down, Left, Right, Up};

enum Direction {
    Up,
    Down,
    Left,
    Right,
}

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
    x: i8,
    y: i8,
}

fn P(x: i8, y: i8) -> Point {
    Point { x, y }
}

impl Point {
    fn neighbors(self) -> NeighborsIter {
        NeighborsIter::new(self)
    }
}

impl std::ops::Add for Point {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

struct NeighborsIter {
    point: Point,
    neighbor: Option<Direction>,
}

impl NeighborsIter {
    fn new(point: Point) -> Self {
        Self {
            point,
            neighbor: Some(Up),
        }
    }
}

impl Iterator for NeighborsIter {
    type Item = Point;

    fn next(&mut self) -> Option<Point> {
        match self.neighbor {
            Some(Up) => {
                self.neighbor = Some(Right);
                Some(self.point + P(0, 1))
            }
            Some(Right) => {
                self.neighbor = Some(Down);
                Some(self.point + P(1, 0))
            }
            Some(Down) => {
                self.neighbor = Some(Left);
                Some(self.point + P(0, -1))
            }
            Some(Left) => {
                self.neighbor = None;
                Some(self.point + P(-1, 0))
            }
            None => None,
        }
    }
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
