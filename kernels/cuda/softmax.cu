#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void softmax_kernel_cuda(const float* input, float* output, int n_rows, int n_cols) {
    int row = blockIdx.x;
    if (row < n_rows) {
        const float* row_in = input + row * n_cols;
        float* row_out = output + row * n_cols;
        
        float max_val = -INFINITY;
        for (int i = threadIdx.x; i < n_cols; i += blockDim.x) {
            max_val = max(max_val, row_in[i]);
        }
        
        // Block reduction for max
        __shared__ float shared_max[1024];
        shared_max[threadIdx.x] = max_val;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                shared_max[threadIdx.x] = max(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
            }
            __syncthreads();
        }
        max_val = shared_max[0];
        
        float sum_val = 0.0f;
        for (int i = threadIdx.x; i < n_cols; i += blockDim.x) {
            sum_val += exp(row_in[i] - max_val);
        }
        
        // Block reduction for sum
        __shared__ float shared_sum[1024];
        shared_sum[threadIdx.x] = sum_val;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
            }
            __syncthreads();
        }
        sum_val = shared_sum[0];
        
        for (int i = threadIdx.x; i < n_cols; i += blockDim.x) {
            row_out[i] = exp(row_in[i] - max_val) / sum_val;
        }
    }
}

torch::Tensor softmax_cuda(torch::Tensor x) {
    auto out = torch::empty_like(x);
    int n_rows = x.size(0);
    int n_cols = x.size(1);
    
    int threads = 1024;
    if (n_cols < 1024) threads = n_cols; // basic optimization for small cols
    // next power of 2 for threads if less than 1024
    int p = 1;
    while (p < threads) p *= 2;
    threads = p;
    
    int blocks = n_rows;
    
    softmax_kernel_cuda<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), n_rows, n_cols);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax", &softmax_cuda, "Softmax (CUDA)");
}
