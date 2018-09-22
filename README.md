Run bench with `RUSTFLAGS="-C target-cpu=native" cargo bench -- "matmul 512"`

```
Naive matmul (0f41b2a):
matmul 64               time:   [617.42 us 618.63 us 620.00 us]

Removing bounds checks (a43c84b):
matmul 64               time:   [242.98 us 243.87 us 244.84 us]
                        change: [-60.917% -60.537% -60.019%] (p = 0.00 < 0.05)

Block by 8 (bda49f1)
matmul 64               time:   [182.35 us 183.13 us 184.19 us]
                        change: [-26.114% -25.159% -24.405%] (p = 0.00 < 0.05)

```

