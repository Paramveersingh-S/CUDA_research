# Cross-Shape Optimal Configuration Stability

## Kernel: vector_add

- **Shape 5242880**
  - Min Energy: BLOCK_SIZE: 256, num_warps: 4, num_ctas: 1, num_stages: 3, maxnreg: None -> 5.90 mJ
  - Min Latency: BLOCK_SIZE: 1024, num_warps: 16, num_ctas: 1, num_stages: 3, maxnreg: None -> 0.27 ms

- **Shape 10485760**
  - Min Energy: BLOCK_SIZE: 256, num_warps: 2, num_ctas: 1, num_stages: 3, maxnreg: None -> 22.93 mJ
  - Min Latency: BLOCK_SIZE: 512, num_warps: 16, num_ctas: 1, num_stages: 3, maxnreg: None -> 0.52 ms

- **Shape 20971520**
  - Min Energy: BLOCK_SIZE: 512, num_warps: 16, num_ctas: 1, num_stages: 3, maxnreg: None -> 50.01 mJ
  - Min Latency: BLOCK_SIZE: 512, num_warps: 16, num_ctas: 1, num_stages: 3, maxnreg: None -> 1.02 ms

## Kernel: softmax

- **Shape (2048, 2048)**
  - Min Energy: num_warps: 4, num_ctas: 1, num_stages: 3, maxnreg: None -> 3.61 mJ
  - Min Latency: num_warps: 4, num_ctas: 1, num_stages: 3, maxnreg: None -> 0.17 ms

- **Shape (4096, 4096)**
  - Min Energy: num_warps: 16, num_ctas: 1, num_stages: 3, maxnreg: None -> 42.90 mJ
  - Min Latency: num_warps: 32, num_ctas: 1, num_stages: 3, maxnreg: None -> 0.59 ms

- **Shape (8192, 8192)**
  - Min Energy: num_warps: 32, num_ctas: 1, num_stages: 3, maxnreg: None -> 160.21 mJ
  - Min Latency: num_warps: 32, num_ctas: 1, num_stages: 3, maxnreg: None -> 2.32 ms

## Kernel: rmsnorm

- **Shape (2048, 2048)**
  - Min Energy: num_warps: 4, num_ctas: 1, num_stages: 3, maxnreg: None -> 3.52 mJ
  - Min Latency: num_warps: 32, num_ctas: 1, num_stages: 3, maxnreg: None -> 0.17 ms

- **Shape (4096, 4096)**
  - Min Energy: num_warps: 16, num_ctas: 1, num_stages: 3, maxnreg: None -> 27.50 mJ
  - Min Latency: num_warps: 32, num_ctas: 1, num_stages: 3, maxnreg: None -> 0.59 ms

- **Shape (8192, 8192)**
  - Min Energy: num_warps: 16, num_ctas: 1, num_stages: 3, maxnreg: None -> 141.18 mJ
  - Min Latency: num_warps: 32, num_ctas: 1, num_stages: 3, maxnreg: None -> 2.32 ms

## Kernel: matmul

- **Shape (1024, 1024, 1024)**
  - Min Energy: BLOCK_SIZE_M: 64, BLOCK_SIZE_N: 64, BLOCK_SIZE_K: 32, GROUP_SIZE_M: 8, num_warps: 8, num_ctas: 1, num_stages: 4, maxnreg: None -> 27.01 mJ
  - Min Latency: BLOCK_SIZE_M: 64, BLOCK_SIZE_N: 64, BLOCK_SIZE_K: 32, GROUP_SIZE_M: 8, num_warps: 8, num_ctas: 1, num_stages: 2, maxnreg: None -> 0.72 ms

- **Shape (2048, 2048, 2048)**
  - Min Energy: BLOCK_SIZE_M: 64, BLOCK_SIZE_N: 64, BLOCK_SIZE_K: 32, GROUP_SIZE_M: 8, num_warps: 8, num_ctas: 1, num_stages: 4, maxnreg: None -> 385.95 mJ
  - Min Latency: BLOCK_SIZE_M: 64, BLOCK_SIZE_N: 64, BLOCK_SIZE_K: 32, GROUP_SIZE_M: 8, num_warps: 8, num_ctas: 1, num_stages: 2, maxnreg: None -> 5.75 ms

- **Shape (4096, 4096, 4096)**
  - Min Energy: BLOCK_SIZE_M: 64, BLOCK_SIZE_N: 64, BLOCK_SIZE_K: 32, GROUP_SIZE_M: 8, num_warps: 8, num_ctas: 1, num_stages: 4, maxnreg: None -> 3147.21 mJ
  - Min Latency: BLOCK_SIZE_M: 64, BLOCK_SIZE_N: 64, BLOCK_SIZE_K: 32, GROUP_SIZE_M: 8, num_warps: 8, num_ctas: 1, num_stages: 4, maxnreg: None -> 48.00 ms

## Kernel: fused_attention

- **Shape (2, 8, 512, 64)**
  - Min Energy: BLOCK_M: 64, BLOCK_N: 32, num_warps: 8, num_ctas: 1, num_stages: 4, maxnreg: None -> 27.69 mJ
  - Min Latency: BLOCK_M: 64, BLOCK_N: 32, num_warps: 8, num_ctas: 1, num_stages: 3, maxnreg: None -> 0.59 ms

- **Shape (2, 8, 1024, 64)**
  - Min Energy: BLOCK_M: 64, BLOCK_N: 32, num_warps: 8, num_ctas: 1, num_stages: 3, maxnreg: None -> 125.88 mJ
  - Min Latency: BLOCK_M: 64, BLOCK_N: 32, num_warps: 8, num_ctas: 1, num_stages: 3, maxnreg: None -> 1.96 ms

- **Shape (2, 8, 2048, 64)**
  - Min Energy: BLOCK_M: 64, BLOCK_N: 32, num_warps: 8, num_ctas: 1, num_stages: 4, maxnreg: None -> 486.70 mJ
  - Min Latency: BLOCK_M: 64, BLOCK_N: 32, num_warps: 8, num_ctas: 1, num_stages: 3, maxnreg: None -> 7.27 ms
