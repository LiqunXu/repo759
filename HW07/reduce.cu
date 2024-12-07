#include "reduce.cuh"
#include <cuda_runtime.h>
#include <iostream>

// CUDA kernel for parallel reduction using "First Add During Load"
__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[]; // Shared memory for partial sums

    unsigned int tid = threadIdx.x;  // Thread ID within block
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x; // Global index

    // Load elements into shared memory, applying the "first add during load"
    sdata[tid] = (idx < n ? g_idata[idx] : 0.0f) +
                 (idx + blockDim.x < n ? g_idata[idx + blockDim.x] : 0.0f);

    __syncthreads();

    // Perform reduction within the block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

// Host function for parallel reduction
__host__ void reduce(float **input, float **output, unsigned int N, unsigned int threads_per_block) {
    // Calculate number of blocks needed for the first kernel launch
    unsigned int num_blocks = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);

    // Keep reducing until the result fits into one block
    float *idata = *input;
    float *odata = *output;
    size_t shared_mem_size = threads_per_block * sizeof(float);

    while (num_blocks > 1) {
        // Launch kernel
        reduce_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(idata, odata, N);
        cudaDeviceSynchronize();

        // Prepare for the next iteration
        N = num_blocks;
        num_blocks = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);

        // Swap input and output pointers
        float *temp = idata;
        idata = odata;
        odata = temp;
    }

    // Final reduction step (if needed)
    reduce_kernel<<<1, threads_per_block, shared_mem_size>>>(idata, odata, N);
    cudaDeviceSynchronize();

    // The final result is now in the first element of odata
    *input = odata;
}