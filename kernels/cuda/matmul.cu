#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define TILE_SIZE 32

__global__ void matmul_kernel_cuda(const half* A, const half* B, half* C, int M, int N, int K) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    __shared__ half sA[TILE_SIZE][TILE_SIZE];
    __shared__ half sB[TILE_SIZE][TILE_SIZE];
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            sA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = __float2half(0.0f);
            
        if (t * TILE_SIZE + threadIdx.y < K && col < N)
            sB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = __float2half(0.0f);
            
        __syncthreads();
        
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += __half2float(sA[threadIdx.y][i]) * __half2float(sB[i][threadIdx.x]);
        }
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = __float2half(sum);
    }
}

torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b) {
    auto c = torch::empty({a.size(0), b.size(1)}, a.options());
    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    matmul_kernel_cuda<<<blocks, threads>>>(
        reinterpret_cast<const half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<half*>(c.data_ptr<at::Half>()),
        M, N, K
    );
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &matmul_cuda, "Tiled Matmul (CUDA)");
}
