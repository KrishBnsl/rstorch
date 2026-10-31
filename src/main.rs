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

    fn get(&self, indices: &[usize]) -> Result<f32, TensorError> {
        let valid: bool = self.shape.iter().ge(indices.iter());
        if !valid {
            return Err(TensorError::IndexOutOfBounds);
        }
        let idxfinal: usize = self
            .shape
            .iter()
            .zip(indices.iter())
            .map(|(x, y)| x * y)
            .sum();
        Ok(self.data[idxfinal])
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

fn main() {
    // let tensor: Tensor = Tensor::new((1..=24).map(|x| x as f32).collect(), vec![2, 3, 4]).unwrap();
    // println!("Shape: {:?}", tensor.shape);
    // println!("Strides: {:?}", tensor.strides);
    // println!("Data: {:?}", tensor.data);
    // //    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    //     println!("{}", tensor.numel());
    //     println!("{:?}", tensor.shape);
    //     let mut numbers = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    //
    //     add_to_slice(&mut numbers[1..4], 10.0);
    //
    //     println!("{:?}", numbers);
    //
    //     let tensor =
    //         Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).expect("Invalid Tensor Shape");
    //     println!("{:?}", tensor);
    //     println!("{:?}", tensor.data);
    //     let shape = vec![2, 3, 4, 5];
    //
    //     let strides = calculate_strides(&shape);
    //
    //     println!("{:?}", strides);
    let tensor = Tensor::new((1..=24).map(|x| x as f32).collect(), vec![2, 3, 4]).unwrap();

    println!("{:?}", tensor.get(&[1, 2, 3]));
    println!("{:?}", tensor.get(&[0, 1, 2]));
    println!("{:?}", tensor.get(&[1, 3, 2]));
}
