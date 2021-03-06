## Playing with matmul and SIMD

This was a fun hacking attempt try and write a high performance matmul kernel by myself in Rust.

I ended up getting within an order of magnitude (8x slower) of Intel MKL, though my implementation is only single threaded so I'm within perhaps 4x. Though my implementation is only marginally accurate (tests were passing for 8x8x8, but not 8x16x8) and only supports multiple sizes of 8. Also, coding style etc is not great, so please don't judge this as my ability to write good code.

Run bench with `RUSTFLAGS="-C target-cpu=native" cargo bench 

```
using 64 bit double precision fp:

Naive matmul (0f41b2a):
matmul 64               time:   [617.42 us 618.63 us 620.00 us]

Removing bounds checks (a43c84b):
matmul 64               time:   [242.98 us 243.87 us 244.84 us]
                        change: [-60.917% -60.537% -60.019%] (p = 0.00 < 0.05)

Block by 8 (bda49f1)
matmul 64               time:   [182.35 us 183.13 us 184.19 us]
                        change: [-26.114% -25.159% -24.405%] (p = 0.00 < 0.05)

Further blocking (different) and a,b packing (a8c47e5):
matmul 64               time:   [132.59 us 132.92 us 133.30 us]

C packing too (and some fixes, but still not quite correct for multi-tiles) (98c31e):
matmul 64               time:   [58.021 us 58.134 us 58.266 us]
                        change: [-56.869% -56.272% -55.724%] (p = 0.00 < 0.05)

By Comparison, here is ndarray's implementation:
matmul_ndarray 64       time:   [29.604 us 30.222 us 31.029 us]

and ndarray's implementation when backed with MKL:
matmul_ndarray 64       time:   [9.0853 us 9.1128 us 9.1453 us]

(this is ~57GFlops on my little laptop: Intel(R) Core(TM) i5-7200U CPU @ 2.50GHz)

Where theoretical DP peak is around 99 GFlop/s (2 cores x 3.1GHz turbo * 16 DP flops/cycle)
And theoretical SP peak is around 198 GFlop/s (2 cores x 3.1GHz turbo * 32 SP flops/cycle)
(if I change to 32bit, I can achieve about 130 GFlop/s which is still off theoretical peak)

Now switching to SP:

(started at "C packing too" from above):
matmul 64               time:   [33.769 us 33.837 us 33.917 us]                       
                        change: [-41.702% -41.242% -40.586%] (p = 0.00 < 0.05)

MKL:
matmul_ndarray 64       time:   [4.2311 us 4.2421 us 4.2547 us]
```

I also tried manual f32x8 packing and manual AVX{2} intrinsics attempts, but both were significantly slower, and I wasn't able to tell why when looking at the assembly.

