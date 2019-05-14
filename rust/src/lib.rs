use pyo3::prelude::*;

#[pymodule]
fn goban(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "goban_test")]
    fn goban_test(_py: Python, s: &str) -> PyResult<i64> {
        println!("Test: {}", s);
        Ok(43)
    }
    Ok(())
}
