#![feature(rust_2018_preview)]

#[derive(Debug)]
pub struct Mat {
    pub data: Vec<f32>,
    pub height: usize,
    pub width: usize,
}

pub fn get_rand_matrix(m: usize, n: usize) -> Mat {
    use rand::distributions::{Normal, Distribution};
    let normal = Normal::new(0.0, 3.0);
    let mut rng = rand::thread_rng();
    let mut mat = Mat{data: Vec::with_capacity(n*m), width: n, height: m};
    for _ in 0..n*m {
            mat.data.push(normal.sample(&mut rng) as f32);
    }
    mat
}

// Order somewhat inspired from http://www.cs.utexas.edu/users/flame/pubs/blis1_toms_rev3.pdf
pub fn matmul(a: &Mat, b: &Mat) -> Mat {
    let mut result = Mat{data: vec![0.0; a.height * b.width], height: a.height, width: b.width};
    let n = a.height;
    let block = 8;
    let mut a_pack = unsafe {[_mm256_setzero_ps(); 8]};
    let mut b_pack = unsafe {[_mm256_setzero_ps(); 8]};
    let mut c_pack = [[0.0; 8]; 8];
    let mut tmp = [0.0; 8];

    // mkn ordering
    use std::arch::x86_64::*;

    // loop 5: 
    for nc in 0..n/block {
        // loop 4:
        for kc in 0..n/block {
            unsafe {
                // Pack b matrix chunk and transpose
                for u in 0..block {
                    b_pack[u] = _mm256_set_ps(*b.data.get_unchecked(nc*block+u + n*(kc*block+0)),
                                              *b.data.get_unchecked(nc*block+u + n*(kc*block+1)),
                                              *b.data.get_unchecked(nc*block+u + n*(kc*block+2)),
                                              *b.data.get_unchecked(nc*block+u + n*(kc*block+3)),
                                              *b.data.get_unchecked(nc*block+u + n*(kc*block+4)),
                                              *b.data.get_unchecked(nc*block+u + n*(kc*block+5)),
                                              *b.data.get_unchecked(nc*block+u + n*(kc*block+6)),
                                              *b.data.get_unchecked(nc*block+u + n*(kc*block+7)));
                }
            }
            // loop 3
            for mc in 0..n/block {
                // Pack the a matrix chunk
                for u in 0..block {
                    unsafe {
                        a_pack[u] = _mm256_load_ps(a.data.as_ptr().offset((kc*block + n*(mc*block+u)) as isize));
                    }
                }
                for u in 0..block {
                    for v in 0..block {
                        unsafe {
                            let products =  _mm256_mul_ps(a_pack[u], b_pack[v]);
                            _mm256_store_ps(tmp.as_ptr(), products);
                            c_pack[u][v] = tmp.iter().fold(0.0, |a, b| a+b);
                        }
                    }
                }
                // copy back to C
                for u in 0..block {
                    for v in 0..block {
                        unsafe {
                            *result.data.get_unchecked_mut((mc*block+u)*n + nc*block+v) += c_pack[u][v];
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
