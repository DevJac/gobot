mod board;

use pyo3::prelude::*;

#[pyclass]
struct Point {
    point: board::Point,
}

#[pymethods]
impl Point {
    #[new]
    fn pynew(obj: &PyRawObject, x: i8, y: i8) {
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
    fn pynew(obj: &PyRawObject, size: i8) {
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

    fn test(&self, point: PyObject) -> PyResult<&str> {
        let x: &Point = point.cast_as(Python::acquire_gil().python())?;
        Ok(match self.board.get_position(x.point) {
            board::BoardPosition::Empty => "e",
            board::BoardPosition::Black => "b",
            board::BoardPosition::White => "w",
        })
    }
}

#[pymodule]
fn goban(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Point>()?;
    m.add_class::<Board>()?;
    Ok(())
}
