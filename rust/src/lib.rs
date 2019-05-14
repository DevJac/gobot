#![allow(clippy::new_ret_no_self, clippy::use_self)]

mod board;

use pyo3::prelude::*;
use pyo3::PyObjectProtocol;

#[pyclass]
struct Point {
    point: board::Point,
}

#[pyproto]
impl PyObjectProtocol for Point {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("Point({}, {})", self.point.x(), self.point.y()))
    }
}

#[pymethods]
impl Point {
    #[new]
    fn new(obj: &PyRawObject, x: i8, y: i8) {
        obj.init(Self {
            point: board::P(x, y),
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
    board: board::Board,
}

#[pymethods]
impl Board {
    #[new]
    fn new(obj: &PyRawObject, size: i8) {
        obj.init(Self {
            board: board::Board::new(size),
        });
    }

    fn valid_moves(&self, pos: &str) -> PyResult<Vec<Point>> {
        let pos = match pos {
            "b" => board::BoardPosition::Black,
            "w" => board::BoardPosition::White,
            _ => board::BoardPosition::Empty,
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
fn goban(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Point>()?;
    m.add_class::<Board>()?;
    Ok(())
}
