use std::ops::{Add, Div, Mul, Sub};

use crate::tensor;

#[derive(Debug)]
pub enum TensorError {
    InvalidShape { expected: usize, actual: usize },
    ShapeMismatch,
    InvalidIndexCount,
    IndexOutOfBounds,
    InvalidMatrixOperation,
}

#[derive(Debug, PartialEq)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, TensorError> {
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            return Err(TensorError::InvalidShape {
                expected: (expected_size),
                actual: (data.len()),
            });
        }
        let strides: Vec<usize> = Self::calculate_strides(&shape);
        Ok(Self {
            data,
            shape,
            strides,
        })
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    pub fn multiply_scalar(&mut self, scalar: f32) {
        for i in 0..self.data.len() {
            self.data[i] *= scalar;
        }
    }

    pub fn scaled(mut self, scalar: f32) -> Self {
        for value in &mut self.data {
            *value *= scalar;
        }
        self
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

    pub fn get(&self, indices: &[usize]) -> Result<f32, TensorError> {
        let flatidx: usize = self.flat_index(indices)?;
        Ok(self.data[flatidx])
    }

    pub fn set(&mut self, indices: &[usize], val: f32) -> Result<(), TensorError> {
        let flatidx: usize = self.flat_index(indices)?;
        self.data[flatidx] = val;
        Ok(())
    }

    pub fn full(shape: &[usize], val: f32) -> Result<Tensor, TensorError> {
        let data: Vec<f32> = vec![val; shape.iter().product()];
        match Tensor::new(data, shape.to_vec()) {
            Ok(tensor) => Ok(tensor),
            Err(err) => Err(err),
        }
    }

    pub fn zeros(shape: &[usize]) -> Result<Self, TensorError> {
        Self::full(shape, 0.0)
    }

    pub fn ones(shape: &[usize]) -> Result<Self, TensorError> {
        Self::full(shape, 1.0)
    }

    pub fn reshape(&mut self, new_shape: &[usize]) -> Result<(), TensorError> {
        if new_shape.iter().product::<usize>() != self.shape.iter().product::<usize>() {
            return Err(TensorError::ShapeMismatch);
        }
        let new_stride_vector: Vec<usize> = Self::calculate_strides(new_shape);
        self.shape = new_shape.to_vec();
        self.strides = new_stride_vector;
        Ok(())
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

    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(TensorError::InvalidMatrixOperation);
        }

        if self.shape[1] != other.shape[0] {
            return Err(TensorError::InvalidMatrixOperation);
        }

        let m: usize = self.shape[0];
        let n: usize = self.shape[1];
        let p: usize = other.shape[1];

        let mut result = vec![0.0_f32; m * p];

        for i in 0..m {
            for j in 0..p {
                let mut sum = 0.0_f32;
                for k in 0..n {
                    let a = self.data[i * n + k];
                    let b = other.data[k * p + j];
                    sum += a * b;
                }

                result[i * p + j] = sum;
            }
        }
        Tensor::new(result, vec![m, p])
    }

    fn elementwise_operation<F>(&self, other: &Tensor, operation: F) -> Result<Self, TensorError>
    where
        F: Fn(f32, f32) -> f32,
    {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch);
        }
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(x, y)| operation(*x, *y))
            .collect();
        Self::new(data, self.shape.clone())
    }

    pub fn add(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        self.elementwise_operation(other, |x, y| x + y)
    }

    pub fn sub(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        self.elementwise_operation(other, |x, y| x - y)
    }

    pub fn mul(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        self.elementwise_operation(other, |x, y| x * y)
    }

    pub fn div(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        self.elementwise_operation(other, |x, y| x / y)
    }
}

fn add_to_slice(data: &mut [f32], amount: f32) {
    for value in data {
        *value += amount;
    }
}

impl Add<&Tensor> for &Tensor {
    type Output = Result<Tensor, TensorError>;
    fn add(self, rhs: &Tensor) -> Self::Output {
        Tensor::add(self, rhs)
    }
}

impl Sub<&Tensor> for &Tensor {
    type Output = Result<Tensor, TensorError>;
    fn sub(self, rhs: &Tensor) -> Self::Output {
        Tensor::sub(self, rhs)
    }
}

impl Mul<&Tensor> for &Tensor {
    type Output = Result<Tensor, TensorError>;
    fn mul(self, rhs: &Tensor) -> Self::Output {
        Tensor::mul(self, rhs)
    }
}

impl Div<&Tensor> for &Tensor {
    type Output = Result<Tensor, TensorError>;
    fn div(self, rhs: &Tensor) -> Self::Output {
        Tensor::div(&self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use std::f32::NAN;

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
    #[test]
    fn test_failed_reshape_preserves_tensor() {
        let mut tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        tensor.reshape(&[3, 2]).unwrap();
        let result = tensor.reshape(&[2, 2]);
        assert!(result.is_err());
        assert_eq!(tensor.shape, vec![3, 2]);
        assert_eq!(tensor.strides, vec![2, 1]);
        assert_eq!(tensor.data.len(), 6);
    }

    #[test]
    fn test_matmul_correctness() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.strides, vec![2, 1]);
        assert_eq!(c.data, vec![58.0, 64.0, 139.0, 154.0]);
    }
    #[test]
    fn test_matmul_invalid_dimensions() {
        let a = Tensor::ones(&[2, 3]).unwrap();
        let b = Tensor::ones(&[3]).unwrap();

        assert!(matches!(
            a.matmul(&b),
            Err(TensorError::InvalidMatrixOperation)
        ));
    }

    #[test]
    fn test_matmul_inner_dimension_mismatch() {
        let a = Tensor::ones(&[2, 3]).unwrap();
        let b = Tensor::ones(&[4, 2]).unwrap();

        assert!(matches!(
            a.matmul(&b),
            Err(TensorError::InvalidMatrixOperation)
        ));
    }

    #[test]
    fn test_get_functions_on_priv_variables_of_a_tensor() {
        let a = Tensor::ones(&[4, 5]).unwrap();
        assert_eq!(a.shape(), &[4, 5]);
        assert_eq!(a.strides(), &[5, 1]);
        assert_eq!(a.data(), &[1.0; 20]);
    }

    #[test]
    fn test_tensor_metadata() {
        let a = Tensor::zeros(&[2, 3, 4]).unwrap();

        assert_eq!(a.shape(), &[2, 3, 4]);
        assert_eq!(a.strides(), &[12, 4, 1]);
        assert_eq!(a.data().len(), 24);
    }

    #[test]
    fn test_mul_sub_operations() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![2, 3]).unwrap();

        let c: Tensor = Tensor::new(vec![-6.0; 6], vec![2, 3]).unwrap();
        let d: Tensor = Tensor::new(vec![7.0, 16.0, 27.0, 40.0, 55.0, 72.0], vec![2, 3]).unwrap();
        assert_eq!(a.sub(&b).unwrap(), c);
        assert_eq!(a.mul(&b).unwrap(), d);
    }

    #[test]
    fn test_mul_sub_failing_operations() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]).unwrap();

        let c: Tensor = Tensor::new(vec![-6.0; 6], vec![2, 3]).unwrap();
        assert!(matches!(a.sub(&b), Err(TensorError::ShapeMismatch)));
    }

    #[test]
    fn test_operator_overloading_tensor_operations() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![2, 3]).unwrap();

        let c: Tensor = Tensor::new(vec![-6.0; 6], vec![2, 3]).unwrap();
        let d: Tensor = Tensor::new(vec![7.0, 16.0, 27.0, 40.0, 55.0, 72.0], vec![2, 3]).unwrap();
        assert_eq!((&a - &b).unwrap(), c);
        assert_eq!((&a * &b).unwrap(), d);
    }

    #[test]
    fn test_mul_sub_failing_operations_operator_overloading() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]).unwrap();

        let c: Tensor = Tensor::new(vec![-6.0; 6], vec![2, 3]).unwrap();
        assert!(matches!((&a - &b), Err(TensorError::ShapeMismatch)));
    }

    #[test]
    fn test_div_operations_overloaded() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]).unwrap();
        let c: Tensor = Tensor::new(
            vec![
                1.0 / 7.0,
                2.0 / 8.0,
                3.0 / 9.0,
                4.0 / 10.0,
                5.0 / 11.0,
                6.0 / 12.0,
            ],
            vec![2, 3],
        )
        .unwrap();
        assert_eq!((&a / &b).unwrap(), c);
    }

    #[test]
    fn test_div_operations_overloaded_failing() {
        let a = Tensor::new(vec![0.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();

        let b = Tensor::new(vec![0.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]).unwrap();

        assert!(matches!(&a / &b, Err(TensorError::ShapeMismatch)));
    }

    #[test]
    fn test_division_by_zero() {
        let a = Tensor::new(vec![0.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();

        let b = Tensor::new(vec![0.0, 0.0, 9.0, 10.0, 11.0, 12.0], vec![2, 3]).unwrap();

        let result = (&a / &b).unwrap();

        assert!(result.data()[0].is_nan());
        assert!(result.data()[1].is_infinite());

        assert_eq!(result.data()[2], 3.0 / 9.0);
        assert_eq!(result.data()[3], 4.0 / 10.0);
        assert_eq!(result.data()[4], 5.0 / 11.0);
        assert_eq!(result.data()[5], 6.0 / 12.0);
    }
}
