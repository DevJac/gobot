#![warn(
    warnings,
    future_incompatible,
    nonstandard_style,
    rust_2018_compatibility,
    rust_2018_idioms,
    rustdoc,
    unused
)]
#![allow(non_snake_case, clippy::module_name_repetitions)]

use std::collections::HashSet;
use BoardPosition::{Black, Empty, White};
use Direction::{Down, Left, Right, Up};

enum Direction {
    Up,
    Down,
    Left,
    Right,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BoardPosition {
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
pub struct Point {
    x: i8,
    y: i8,
}

pub fn P(x: i8, y: i8) -> Point {
    Point::new(x, y)
}

impl Point {
    pub fn new(x: i8, y: i8) -> Self {
        Self { x, y }
    }

    pub fn neighbors(self) -> NeighborsIter {
        NeighborsIter::new(self, false)
    }

    pub fn with_neighbors(self) -> NeighborsIter {
        NeighborsIter::new(self, true)
    }

    pub fn x(self) -> i8 {
        self.x
    }

    pub fn y(self) -> i8 {
        self.y
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

pub struct NeighborsIter {
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

pub struct PointIter {
    board_size: i8,
    x: i8,
    y: i8,
}

impl PointIter {
    fn new(board_size: i8) -> Self {
        Self {
            board_size,
            x: 0,
            y: 0,
        }
    }
}

impl Iterator for PointIter {
    type Item = Point;

    fn next(&mut self) -> Option<Point> {
        if self.x >= self.board_size || self.y >= self.board_size {
            return None;
        }
        let next = P(self.x, self.y);
        self.x += 1;
        if self.x >= self.board_size {
            self.x -= self.board_size;
            self.y += 1;
        }
        Some(next)
    }
}

#[derive(Clone, Debug)]
pub struct Board {
    size: i8,
    board: Vec<BoardPosition>,
    liberties: Vec<i16>,
    // TODO: With multiple copies of the board, there will be a lot of history duplication. Fix.
    history: HashSet<Vec<BoardPosition>>,
}

impl PartialEq for Board {
    fn eq(&self, other: &Self) -> bool {
        self.board == other.board
    }
}

impl Board {
    #[allow(clippy::cast_sign_loss)]
    pub fn new(size: i8) -> Self {
        let vec_len = (size as usize).pow(2);
        Self {
            size,
            board: vec![Empty; vec_len],
            liberties: vec![0; vec_len],
            history: HashSet::with_capacity(8),
        }
    }

    #[allow(clippy::cast_sign_loss)]
    fn to_index(&self, point: Point) -> usize {
        (point.x as usize) * (self.size as usize) + (point.y as usize)
    }

    pub fn size(&self) -> i8 {
        self.size
    }

    pub fn position(&self, point: Point) -> BoardPosition {
        self.board[self.to_index(point)]
    }

    pub fn set_position(&mut self, point: Point, pos: BoardPosition) {
        let i = self.to_index(point);
        self.board[i] = pos;
        self.update_liberties(point.with_neighbors());
    }

    pub fn liberties(&self, point: Point) -> i16 {
        self.liberties[self.to_index(point)]
    }

    fn set_liberties(&mut self, point: Point, liberties: i16) {
        let i = self.to_index(point);
        self.liberties[i] = liberties;
    }

    pub fn on_board(&self, point: Point) -> bool {
        0 <= point.x && point.x < self.size && 0 <= point.y && point.y < self.size
    }

    pub fn off_board(&self, point: Point) -> bool {
        !self.on_board(point)
    }

    pub fn points(&self) -> PointIter {
        PointIter::new(self.size)
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
                if board.position(neighboring_point) == Empty {
                    group_liberties.insert(neighboring_point);
                } else if board.position(this_point) == board.position(neighboring_point)
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
            if self.position(point) == Empty {
                self.set_liberties(point, 0);
                continue;
            }
            if updated_liberties[self.to_index(point)] != -1 {
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
        for p in self.points() {
            if self.position(p) == color_to_remove && self.liberties(p) == 0 {
                self.set_position(p, Empty);
                points_removed.insert(p);
            }
        }
        self.update_liberties(points_removed.into_iter().flat_map(Point::with_neighbors));
    }

    pub fn valid_moves<'a>(&'a self, pos: BoardPosition) -> impl IntoIterator<Item = Point> + 'a {
        self.points()
            .filter(move |p: &Point| self.can_place_stone_at(*p, pos) && self.not_ko(*p, pos))
    }

    fn can_place_stone_at(&self, point: Point, pos: BoardPosition) -> bool {
        // We can't play on an occupied point.
        if self.position(point) != Empty {
            return false;
        }
        for neighboring_point in point.neighbors() {
            if self.off_board(neighboring_point) {
                continue;
            }
            let neighboring_position = self.position(neighboring_point);
            // If a neighboring point is empty, then the placed stone will have a liberty.
            if neighboring_position == Empty {
                return true;
            }
            let neighboring_liberties = self.liberties(neighboring_point);
            // We can add to one of our groups, as long as it has enough liberties.
            if neighboring_position == pos && neighboring_liberties > 1 {
                return true;
            }
            // We can take the last liberty of an opposing group.
            if neighboring_position == pos.other() && neighboring_liberties == 0 {
                return true;
            }
        }
        false
    }

    fn not_ko(&self, point: Point, pos: BoardPosition) -> bool {
        let not_opposing_stone_in_atari = |neighboring_point| {
            self.off_board(neighboring_point)
                || (self.position(neighboring_point) != pos.other()
                    && self.liberties(neighboring_point) != 1)
        };
        if point.neighbors().all(not_opposing_stone_in_atari) {
            return true;
        }
        // TODO: We should be able to avoid a full clone here.
        let mut b = self.clone();
        b.play(point, pos);
        b.history.contains(&b.board)
    }

    pub fn play(&mut self, point: Point, pos: BoardPosition) {
        // TODO: We do not prevent illegal moves. Fix.
        self.set_position(point, pos);
        self.update_liberties(point.with_neighbors());
        self.remove_stones_without_liberties(pos.other());
        self.history.insert(self.board.clone());
    }
}

#[test]
fn fill_board() {
    let mut b = Board::new(19);
    for p in b.points() {
        b.set_position(p, Black);
    }
}
