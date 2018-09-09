#![feature(rust_2018_preview)]

#[derive(Debug)]
pub struct Mat {
    pub data: Vec<f64>,
    pub height: usize,
    pub width: usize,
}

pub fn get_rand_matrix(m: usize, n: usize) -> Mat {
    use rand::distributions::{Normal, Distribution};
    let normal = Normal::new(0.0, 3.0);
    let mut rng = rand::thread_rng();
    let mut mat = Mat{data: Vec::with_capacity(n*m), width: n, height: m};
    for _ in 0..n*m {
            mat.data.push(normal.sample(&mut rng));
    }
    mat
}

pub fn matmul(a: &Mat, b: &Mat) -> Mat {
    let mut result = Mat{data: vec![0.0; a.height * b.width], height: a.height, width: b.width};
    // let res = result.data.as_mut_ptr();
    for row in 0..a.height {
        for col in 0..b.width {
            for k in 0..a.width {
                // for block_row in 0..8 {
                // unsafe {
                // *result.data.get_unchecked_mut(row*result.width + col) +=
                //     *a.data.get_unchecked(row*a.width + k) *
                //     *b.data.get_unchecked(k*b.width + col);
                // }
                result.data[row*result.width + col] += a.data[row*a.width + k] * b.data[k*b.width + col];
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test() {
        const M: usize = 8;
        const K: usize = 8;
        const N: usize = 8;
        let a = get_rand_matrix(M,K);
        let b = get_rand_matrix(K,N);
        let c = matmul(&a, &b);

        use ndarray::prelude::*;
        let a_nd = Array::from_vec(a.data).into_shape((M,K)).unwrap();
        let b_nd = Array::from_vec(b.data).into_shape((K,N)).unwrap();
        let c_nd = Array::from_vec(c.data).into_shape((M,N)).unwrap();

        let c_nd_true = a_nd.dot(&b_nd);
        println!("True: {}\nc: {}", c_nd_true, c_nd);
        assert!(c_nd_true.all_close(&c_nd, 1e-4));
    }
}
