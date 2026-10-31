#[derive(Debug)]
enum TensorError {
    InvalidShape { expected: usize, actual: usize },
    ShapeMismatch,
    InvalidIndexCount,
    IndexOutOfBounds,
}

#[derive(Debug)]
struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl Tensor {
    fn new(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, TensorError> {
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            return Err(TensorError::InvalidShape {
                expected: (expected_size),
                actual: (data.len()),
            });
        }
        let strides: Vec<usize> = calculate_strides(&shape);
        Ok(Self {
            data,
            shape,
            strides,
        })
    }

    fn numel(&self) -> usize {
        self.data.len()
    }

    fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    fn multiply_scalar(&mut self, scalar: f32) {
        for i in 0..self.data.len() {
            self.data[i] *= scalar;
        }
    }

    fn scaled(mut self, scalar: f32) -> Self {
        for value in &mut self.data {
            *value *= scalar;
        }
        self
    }

    fn add(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch);
        }
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(x, y)| x + y)
            .collect();
        Self::new(data, self.shape.clone())
    }

    fn flat_index(&self, indices: &[usize]) -> Result<usize, TensorError> {
        if self.shape.len() != indices.len() {
            return Err(TensorError::InvalidIndexCount);
        }
        let valid: bool = self.shape.iter().zip(indices.iter()).all(|(&x, &y)| x > y);
        if !valid {
            return Err(TensorError::IndexOutOfBounds);
        }
        let idxfinal: usize = self
            .strides
            .iter()
            .zip(indices.iter())
            .map(|(x, y)| x * y)
            .sum();
        Ok(idxfinal)
    }

    fn get(&self, indices: &[usize]) -> Result<f32, TensorError> {
        let flatidx: usize = self.flat_index(indices)?;
        Ok(self.data[flatidx])
    }

    fn set(&mut self, indices: &[usize], val: f32) -> Result<(), TensorError> {
        let flatidx: usize = self.flat_index(indices)?;
        self.data[flatidx] = val;
        Ok(())
    }
}

fn add_to_slice(data: &mut [f32], amount: f32) {
    for value in data {
        *value += amount;
    }
}

fn calculate_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides: Vec<usize> = shape
        .iter()
        .rev()
        .scan(1usize, |stride, &i| {
            let current_stride = *stride;
            *stride *= i;
            Some(current_stride)
        })
        .collect();
    strides.reverse();
    strides
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_get_and_set() {
        let mut tensor = Tensor::new((1..=24).map(|x| x as f32).collect(), vec![2, 3, 4]).unwrap();

        assert_eq!(tensor.get(&[1, 2, 3]).unwrap(), 24.0);

        tensor.set(&[1, 2, 3], 100.0).unwrap();

        assert_eq!(tensor.get(&[1, 2, 3]).unwrap(), 100.0);
        assert_eq!(tensor.get(&[0, 0, 0]).unwrap(), 1.0);
    }
    #[test]
    fn test_invalid_indices() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

        assert!(matches!(
            tensor.get(&[2, 0]),
            Err(TensorError::IndexOutOfBounds)
        ));

        assert!(matches!(
            tensor.get(&[0]),
            Err(TensorError::InvalidIndexCount)
        ));
    }
}

fn main() {
    let mut tensor = Tensor::new((1..=24).map(|x| x as f32).collect(), vec![2, 3, 4]).unwrap();

    tensor.set(&[1, 2, 3], 100.0).unwrap();

    println!("{:?}", tensor.get(&[1, 2, 3]));
    println!("{:?}", tensor.get(&[0, 0, 0]));

    println!("{:?}", tensor.set(&[2, 0, 0], 50.0));
}
