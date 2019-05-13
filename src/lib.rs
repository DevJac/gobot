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

use std::collections::HashSet;
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
        NeighborsIter::new(self, false)
    }

    fn with_neighbors(self) -> NeighborsIter {
        NeighborsIter::new(self, true)
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
    include_self: bool,
    point: Point,
    neighbor: Option<Direction>,
}

impl NeighborsIter {
    fn new(point: Point, include_self: bool) -> Self {
        Self {
            include_self,
            point,
            neighbor: Some(Up),
        }
    }
}

impl Iterator for NeighborsIter {
    type Item = Point;

    fn next(&mut self) -> Option<Point> {
        if self.include_self {
            self.include_self = false;
            return Some(self.point);
        }
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

    fn get_liberties(&self, point: Point) -> i16 {
        self.liberties[self.to_index(point)]
    }

    fn set_liberties(&mut self, point: Point, liberties: i16) {
        let i = self.to_index(point);
        self.liberties[i] = liberties;
    }

    fn on_board(&self, point: Point) -> bool {
        0 <= point.x && point.x < self.size && 0 <= point.y && point.y < self.size
    }

    fn off_board(&self, point: Point) -> bool {
        !self.on_board(point)
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::needless_range_loop
    )]
    fn update_liberties(&mut self, points: impl IntoIterator<Item = Point>) {
        fn recurse(
            board: &Board,
            this_point: Point,
            group: &mut HashSet<Point>,
            group_liberties: &mut HashSet<Point>,
        ) {
            for neighboring_point in this_point.neighbors() {
                if board.off_board(neighboring_point) {
                    continue;
                }
                if board[neighboring_point] == Empty {
                    group_liberties.insert(neighboring_point);
                } else if board[this_point] == board[neighboring_point]
                    && !group.contains(&neighboring_point)
                {
                    group.insert(neighboring_point);
                    recurse(board, neighboring_point, group, group_liberties);
                }
            }
        }
        let mut updated_liberties: Vec<i16> = vec![-1; self.liberties.len()];
        for point in points {
            if self.off_board(point) {
                continue;
            }
            if self.get_liberties(point) != -1 {
                continue;
            }
            let mut group = HashSet::with_capacity(8);
            group.insert(point);
            let mut group_liberties = HashSet::with_capacity(8);
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

    fn remove_stones_without_liberties(&mut self, color_to_remove: BoardPosition) {
        let mut points_removed = HashSet::with_capacity(8);
        for x in 0..self.size {
            for y in 0..self.size {
                let p = P(x, y);
                if self[p] == color_to_remove && self.get_liberties(p) == 0 {
                    self[p] = Empty;
                    points_removed.insert(p);
                }
            }
        }
        self.update_liberties(points_removed.into_iter().flat_map(Point::with_neighbors));
    }

    fn play(&mut self, point: Point, pos: BoardPosition) {
        // TODO: We do not prevent illegal moves. Fix.
        self[point] = pos;
        self.update_liberties(point.with_neighbors());
        self.remove_stones_without_liberties(pos.other());
    }
}
