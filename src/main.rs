use rstorch::Tensor;

fn main() {
    let mut tensor = Tensor::ones(&[2, 3]).unwrap();

    tensor.set(&[1, 2], 42.0).unwrap();

    println!("{:?}", tensor.get(&[1, 2]));
}
