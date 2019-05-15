#![allow(clippy::use_self)]
use goban;
use pyo3::class;
use pyo3::exceptions;
use pyo3::prelude::*;

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
#[derive(PartialEq)]
struct Point {
    point: goban::Point,
}

#[pyproto]
impl class::basic::PyObjectProtocol for Point {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("Point({}, {})", self.point.x(), self.point.y()))
    }

    fn __richcmp__(&self, other: &Point, op: class::basic::CompareOp) -> PyResult<bool> {
        match op {
            class::basic::CompareOp::Eq => Ok(self == other),
            class::basic::CompareOp::Ne => Ok(self != other),
            _ => Err(exceptions::NotImplementedError::py_err(())),
        }
    }
}

#[pymethods]
impl Point {
    #[new]
    fn pynew(obj: &PyRawObject, x: i8, y: i8) {
        obj.init(Self {
            point: goban::P(x, y),
        })
    }

    #[getter]
    fn get_x(&self) -> PyResult<i8> {
        Ok(self.point.x())
    }

    #[getter]
    fn get_y(&self) -> PyResult<i8> {
        Ok(self.point.y())
    }
}

#[pyclass]
struct Board {
    board: goban::Board,
}

#[pyproto]
impl class::mapping::PyMappingProtocol for Board {
    fn __getitem__(&self, key: &Point) -> PyResult<&'p str> {
        Ok(pos_to_str(self.board.get_position(key.point)))
    }

    fn __setitem__(&mut self, key: &Point, value: &str) -> PyResult<()> {
        self.board.set_position(key.point, str_to_pos(value));
        Ok(())
    }
}

#[pymethods]
impl Board {
    #[new]
    fn pynew(obj: &PyRawObject, size: i8) {
        obj.init(Self {
            board: goban::Board::new(size),
        });
    }

    #[getter]
    fn get_size(&self) -> PyResult<i8> {
        Ok(self.board.get_size())
    }

    fn get_liberties(&self, point: &Point) -> PyResult<i16> {
        Ok(self.board.get_liberties(point.point))
    }

    fn valid_moves(&self, pos: &str) -> PyResult<Vec<Point>> {
        let pos = match pos {
            "b" => goban::BoardPosition::Black,
            "w" => goban::BoardPosition::White,
            _ => goban::BoardPosition::Empty,
        };
        Ok(self
            .board
            .valid_moves(pos)
            .into_iter()
            .map(|p| Point { point: p })
            .collect())
    }
}

#[pymodule]
fn pygoban(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Point>()?;
    m.add_class::<Board>()?;
    Ok(())
}
