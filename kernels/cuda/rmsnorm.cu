#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void rmsnorm_kernel_cuda(const float* input, const float* weight, float* output, int n_rows, int n_cols, float eps) {
    int row = blockIdx.x;
    if (row < n_rows) {
        const float* row_in = input + row * n_cols;
        float* row_out = output + row * n_cols;
        
        float sum_sq = 0.0f;
        for (int i = threadIdx.x; i < n_cols; i += blockDim.x) {
            float val = row_in[i];
            sum_sq += val * val;
        }
        
        // Block reduction
        __shared__ float shared_sum[1024];
        shared_sum[threadIdx.x] = sum_sq;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
            }
            __syncthreads();
        }
        
        float variance = shared_sum[0] / n_cols;
        float rsqrt = rsqrtf(variance + eps);
        
        for (int i = threadIdx.x; i < n_cols; i += blockDim.x) {
            row_out[i] = row_in[i] * rsqrt * weight[i];
        }
    }
}

torch::Tensor rmsnorm_cuda(torch::Tensor x, torch::Tensor weight, float eps) {
    auto out = torch::empty_like(x);
    int n_rows = x.size(0);
    int n_cols = x.size(1);
    
    int threads = 1024;
    if (n_cols < 1024) threads = n_cols;
    int p = 1;
    while (p < threads) p *= 2;
    threads = p;
    
    int blocks = n_rows;
    
    rmsnorm_kernel_cuda<<<blocks, threads>>>(x.data_ptr<float>(), weight.data_ptr<float>(), out.data_ptr<float>(), n_rows, n_cols, eps);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm", &rmsnorm_cuda, "RMSNorm (CUDA)");
}
