#[derive(Debug)]
pub enum TensorError {
    InvalidShape { expected: usize, actual: usize },
    ShapeMismatch,
    InvalidIndexCount,
    IndexOutOfBounds,
    InvalidMatrixOperation,
}
