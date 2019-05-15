use goban;
use pyo3::prelude::*;

#[pyclass]
struct Point {
    point: goban::Point,
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

#[pymethods]
impl Board {
    #[new]
    fn pynew(obj: &PyRawObject, size: i8) {
        obj.init(Self {
            board: goban::Board::new(size),
        });
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

    fn test(&self, point: &Point) -> PyResult<&str> {
        Ok(match self.board.get_position(point.point) {
            goban::BoardPosition::Empty => "e",
            goban::BoardPosition::Black => "b",
            goban::BoardPosition::White => "w",
        })
    }
}

#[pymodule]
fn pygoban(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Point>()?;
    m.add_class::<Board>()?;
    Ok(())
}
