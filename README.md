##
Run bench with `RUSTFLAGS="-C target-cpu=native" cargo bench -- "matmul 512"`

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
(if I change to 32bit and use bigger matrices, I can achieve about 130 GFlop/s (but this is off battery, so I'm not sure turbo is on, which would be peak of 160 GF/s).
```

