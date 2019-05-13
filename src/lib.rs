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

#[derive(Clone, Copy, Debug, PartialEq)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
    size: i8,
    board: Vec<BoardPosition>,
    liberties: Vec<i16>,
    history: Vec<Board>,
}

impl std::ops::Index<Point> for Board {
    type Output = BoardPosition;

    fn index(&self, point: Point) -> &BoardPosition {
        &self.board[self.to_index(point)]
    }
}

impl std::ops::IndexMut<Point> for Board {
    fn index_mut(&mut self, point: Point) -> &mut BoardPosition {
        let i = self.to_index(point);
        &mut self.board[i]
    }
}

impl Board {
    #[allow(clippy::cast_sign_loss)]
    fn to_index(&self, point: Point) -> usize {
        (point.x * self.size + point.y) as usize
    }

    fn on_board(&self, point: Point) -> bool {
        0 <= point.x && point.x < self.size && 0 <= point.y && point.y < self.size
    }

    fn off_board(&self, point: Point) -> bool {
        !self.on_board(point)
    }

    fn update_liberties(&mut self, points: impl IntoIterator<Item=Point>) {
        use std::collections::HashSet;
        let mut updated_liberties: Vec<i16> = vec![-1; self.liberties.len()];
        for point in points {
            if self.off_board(point) {
                continue
            }
            if updated_liberties[self.to_index(point)] != -1 {
                continue
            }
            let mut group = std::collections::HashSet::with_capacity(8);
            group.insert(point);
            let mut group_liberties = std::collections::HashSet::with_capacity(8);

            fn recurse(board: &Board, this_point: Point, group: &mut HashSet<Point>, group_liberties: &mut HashSet<Point>) {
                for neighboring_point in this_point.neighbors() {
                    if board.off_board(neighboring_point) {
                        continue
                    }
                    if board[neighboring_point] == Empty {
                        group_liberties.insert(neighboring_point);
                    } else if board[this_point] == board[neighboring_point] && !group.contains(&neighboring_point) {
                        group.insert(neighboring_point);
                        recurse(board, neighboring_point, group, group_liberties);
                    }
                }
            }

            recurse(self, point, &mut group, &mut group_liberties);
            for group_point in group {
                updated_liberties[self.to_index(group_point)] = group_liberties.len() as i16;
            }
        }
        for i in 0..updated_liberties.len() {
            if updated_liberties[i] != -1 {
                self.liberties[i] = updated_liberties[i];
            }
        }
    }
}
