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

// Order somewhat inspired from http://www.cs.utexas.edu/users/flame/pubs/blis1_toms_rev3.pdf
pub fn matmul(a: &Mat, b: &Mat) -> Mat {
    let mut result = Mat{data: vec![0.0; a.height * b.width], height: a.height, width: b.width};
    let n = a.height;
    let block = 8;
    let mut a_pack = vec![0.0; block*block];
    let mut b_pack = vec![0.0; block*block];

    // mkn ordering

    // loop 5: 
    for nc in 0..n/block {
        // loop 4:
        for kc in 0..n/block {
            unsafe {
                // Pack b matrix chunk and transpose
                for u in 0..block {
                    for v in 0..block {
                        *b_pack.get_unchecked_mut(block*u + v) = *b.data.get_unchecked(nc*block+v + n*(kc*block+u));
                    }
                }
            }
            // loop 3
            for mc in 0..n/block {
                unsafe {
                    // Pack the a matrix chunk
                    for u in 0..block {
                        for v in 0..block {
                            *a_pack.get_unchecked_mut(block*u + v) = *a.data.get_unchecked(kc*block*u + n*(mc*block*v));
                        }
                    }
                }
                for u in 0..block {
                    for v in 0..block {
                        unsafe {
                            let mut sum = *result.data.get_unchecked((mc*block+u)*n + nc*block+v);
                            for kk in 0..block {
                                sum += *a_pack.get_unchecked(u*block + kk) *
                                       *b_pack.get_unchecked(v*block + kk);
                            }
                            *result.data.get_unchecked_mut((mc*block+u)*n + nc*block+v) = sum;
                        }
                    }
                }
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
