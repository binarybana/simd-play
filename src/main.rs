#![feature(rust_2018_preview)]

use simd_play::*;

fn main() {
    let a = get_rand_matrix(2,3);
    let b = get_rand_matrix(3,3);
    let c = matmul(&a, &b);
    println!("Hello, world boya! {:?}", c);
}
