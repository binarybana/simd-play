#[macro_use]
extern crate criterion;

use criterion::Criterion;

use simd_play::*;

const M: usize = 64;
const N: usize = 64;
const K: usize = 64;

fn criterion_benchmark(c: &mut Criterion) {
    let a_mat = get_rand_matrix(M,N);
    let b_mat = get_rand_matrix(N,K);
    c.bench_function("matmul 64", move |b| b.iter(|| matmul(&a_mat, &b_mat)));
}

fn criterion_benchmark2(c: &mut Criterion) {
    let a_mat = get_rand_matrix(M,N);
    let b_mat = get_rand_matrix(N,K);
    use ndarray::prelude::*;
    let a_nd = Array::from_vec(a_mat.data).into_shape((M, N)).unwrap();
    let b_nd = Array::from_vec(b_mat.data).into_shape((N, K)).unwrap();
    c.bench_function("matmul_ndarray 64", move |b| b.iter(|| a_nd.dot(&b_nd)));
}

criterion_group!(benches, criterion_benchmark, criterion_benchmark2);
criterion_main!(benches);
