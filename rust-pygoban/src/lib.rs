#![allow(clippy::use_self)]
use goban;
use pyo3::class;
use pyo3::exceptions;
use pyo3::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

fn str_to_pos(s: &str) -> goban::BoardPosition {
    match s {
        "b" => goban::BoardPosition::Black,
        "w" => goban::BoardPosition::White,
        _ => goban::BoardPosition::Empty,
    }
}

fn pos_to_str(pos: goban::BoardPosition) -> &'static str {
    match pos {
        goban::BoardPosition::Empty => " ",
        goban::BoardPosition::Black => "b",
        goban::BoardPosition::White => "w",
    }
}

#[pyclass]
struct Point {
    point: goban::Point,
}

#[pyproto]
impl class::basic::PyObjectProtocol for Point {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("Point({}, {})", self.point.x(), self.point.y()))
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn __hash__(&self) -> PyResult<isize> {
        let mut hasher = DefaultHasher::new();
        self.point.hash(&mut hasher);
        let hash_num: u64 = hasher.finish();
        Ok(hash_num as isize)
    }

    fn __richcmp__(&self, other: &Point, op: class::basic::CompareOp) -> PyResult<bool> {
        match op {
            class::basic::CompareOp::Eq => Ok(self.point == other.point),
            class::basic::CompareOp::Ne => Ok(self.point != other.point),
            _ => Err(exceptions::NotImplementedError::py_err(())),
        }
    }
}

#[pymethods]
impl Point {
    #[new]
    fn init(obj: &PyRawObject, x: i8, y: i8) {
        obj.init(Self {
            point: goban::P(x, y),
        })
    }

    #[getter]
    fn x(&self) -> PyResult<i8> {
        Ok(self.point.x())
    }

    #[getter]
    fn y(&self) -> PyResult<i8> {
        Ok(self.point.y())
    }
}

#[pyclass]
struct Board {
    board: goban::Board,
}

#[pyproto]
impl class::basic::PyObjectProtocol for Board {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "<Board {}x{}>",
            self.board.size(),
            self.board.size()
        ))
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn __hash__(&self) -> PyResult<isize> {
        let mut hasher = DefaultHasher::new();
        self.board.hash(&mut hasher);
        let hash_num: u64 = hasher.finish();
        Ok(hash_num as isize)
    }

    fn __richcmp__(&self, other: &Board, op: class::basic::CompareOp) -> PyResult<bool> {
        match op {
            class::basic::CompareOp::Eq => Ok(self.board == other.board),
            class::basic::CompareOp::Ne => Ok(self.board != other.board),
            _ => Err(exceptions::NotImplementedError::py_err(())),
        }
    }
}

#[pyproto]
impl class::mapping::PyMappingProtocol for Board {
    fn __getitem__(&self, key: &Point) -> PyResult<&'p str> {
        Ok(pos_to_str(self.board.position(key.point)))
    }

    fn __setitem__(&mut self, key: &Point, value: &str) -> PyResult<()> {
        self.board.set_position(key.point, str_to_pos(value));
        Ok(())
    }
}

#[pymethods]
impl Board {
    #[new]
    fn init(obj: &PyRawObject, size: i8) {
        obj.init(Self {
            board: goban::Board::new(size),
        });
    }

    fn copy(&self) -> PyResult<Self> {
        Ok(Self {
            board: self.board.clone(),
        })
    }

    #[getter]
    fn size(&self) -> PyResult<i8> {
        Ok(self.board.size())
    }

    fn liberties(&self, point: &Point) -> PyResult<i16> {
        Ok(self.board.liberties(point.point))
    }

    fn valid_moves(&self, pos: &str) -> PyResult<Vec<Point>> {
        Ok(self
            .board
            .valid_moves(str_to_pos(pos))
            .into_iter()
            .map(|p| Point { point: p })
            .collect())
    }

    fn play(&mut self, point: &Point, pos: &str) -> PyResult<()> {
        self.board.play(point.point, str_to_pos(pos));
        Ok(())
    }
}

#[pymodule]
fn pygoban(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Point>()?;
    m.add_class::<Board>()?;
    Ok(())
}
