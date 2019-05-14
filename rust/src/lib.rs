#![allow(clippy::new_ret_no_self)]

mod board;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

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

    fn valid_moves(&self, pos: &str) -> PyResult<Vec<(i8, i8)>> {
        let pos = match pos {
            "b" => board::BoardPosition::Black,
            "w" => board::BoardPosition::White,
            _ => board::BoardPosition::Empty,
        };
        Ok(self.board.valid_moves(pos).into_iter().collect())
    }
}

#[pymodule]
fn goban(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Board>()?;
    Ok(())
}
