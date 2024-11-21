#include "stencil.cuh"
#include <cuda_runtime.h>
#include <cstdio>

#include "stencil.cuh"
#include <cuda_runtime.h>
#include <cstdio>

__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R) {
    extern __shared__ float shared[];

    float* shared_image = shared;
    float* shared_mask = &shared[blockDim.x + 2 * R];
    float* shared_output = &shared[blockDim.x + 2 * R + (2 * R + 1)];
    
    unsigned int tid = threadIdx.x;
    unsigned int global_index = blockIdx.x * blockDim.x + tid;

    if (tid < 2 * R + 1) {
        shared_mask[tid] = mask[tid];
    }

    if (global_index < n) {
        shared_image[tid + R] = image[global_index];
    } else {
        shared_image[tid + R] = 1.0f;
    }

    if (tid < R) {
        unsigned int left_index = global_index < R ? 0 : global_index - R;
        shared_image[tid] = (global_index < R) ? 1.0f : image[left_index];
    }

    if (tid >= blockDim.x - R) {
        unsigned int right_index = global_index + R >= n ? n - 1 : global_index + R;
        shared_image[tid + 2 * R] = (global_index + R >= n) ? 1.0f : image[right_index];
    }

    __syncthreads();
    
    float temp_R = (float)R;
    if (global_index < n) {
        float result = 0.0f;
        for (int j = -temp_R; j <= temp_R; ++j) {
            result += shared_image[tid + R + j] * shared_mask[j + R];
        }
        shared_output[tid] = result;
    }

    __syncthreads();
    
    if (global_index < n) {
        output[global_index] = shared_output[tid];
    }
}

void stencil(const float* image, const float* mask, float* output, unsigned int n, unsigned int R, unsigned int threads_per_block) {
    float *d_image, *d_mask, *d_output;
    cudaMalloc((void**)&d_image, n * sizeof(float));
    cudaMalloc((void**)&d_mask, (2 * R + 1) * sizeof(float));
    cudaMalloc((void**)&d_output, n * sizeof(float));

    cudaMemcpy(d_image, image, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, (2 * R + 1) * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block_dim(threads_per_block);
    dim3 grid_dim((n + threads_per_block - 1) / threads_per_block);

    size_t shared_mem_size = (threads_per_block + 2 * R) * sizeof(float) +  // For shared_image
                         (2 * R + 1) * sizeof(float) +                 // For shared_mask
                         threads_per_block * sizeof(float);
    
    stencil_kernel<<<grid_dim, block_dim, shared_mem_size>>>(d_image, d_mask, d_output, n, R);

    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_mask);
    cudaFree(d_output);
}

