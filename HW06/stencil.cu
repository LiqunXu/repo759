#include "stencil.cuh"
#include <cuda_runtime.h>
#include <cstdio>


// Kernel for performing stencil computation
__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R) {
    extern __shared__ float shared_memory[];
    // Shared memory allocation
    float* shared_image = shared_memory; // Shared memory for image
    float* shared_mask = &shared_memory[blockDim.x + 2 * R]; // Shared memory for mask
    float* shared_output = &shared_memory[blockDim.x + 2 * R + (2 * R + 1)]; // Shared memory for output
    
    unsigned int tid = threadIdx.x; // Thread index within the block
    unsigned int global_index = blockIdx.x * blockDim.x + tid; // Global index for the thread
    
    // Load stencil mask into shared memory
    if (tid < 2 * R + 1) {
        shared_mask[tid] = mask[tid];
    }
    // Load the main part of the image into shared memory
    if (global_index < n) {
        shared_image[tid + R] = image[global_index];
    } else {
        shared_image[tid + R] = 1.0f; // Boundary padding
    }

    // Handle left boundary padding
    if (tid < R) {
        unsigned int left_index = global_index < R ? 0 : global_index - R;
        shared_image[tid] = (global_index < R) ? 1.0f : image[left_index];
    }

    // Handle right boundary padding
    if (tid >= blockDim.x - R) {
        unsigned int right_index = global_index + R >= n ? n - 1 : global_index + R;
        shared_image[tid + 2 * R] = (global_index + R >= n) ? 1.0f : image[right_index];
    }

    __syncthreads(); // Synchronize threads within the block
    
    float temp_R = (float)R;
    // Perform stencil computation
    if (global_index < n) {
        float result = 0.0f;
        for (int j = -temp_R; j <= temp_R; ++j) {
            result += shared_image[tid + R + j] * shared_mask[j + R];
        }
        shared_output[tid] = result;
    }

    __syncthreads(); // Synchronize threads within the block
    
    // Store the result in global memory
    if (global_index < n) {
        output[global_index] = shared_output[tid];
    }
}
// Host function for stencil computation
void stencil(const float* image, const float* mask, float* output, unsigned int n, unsigned int R, unsigned int threads_per_block) {
    // Device pointers for input, mask, and output
    float *d_image, *d_mask, *d_output;
    cudaMalloc((void**)&d_image, n * sizeof(float));
    cudaMalloc((void**)&d_mask, (2 * R + 1) * sizeof(float));
    cudaMalloc((void**)&d_output, n * sizeof(float));

    // Copy input data and mask to device memory
    cudaMemcpy(d_image, image, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, (2 * R + 1) * sizeof(float), cudaMemcpyHostToDevice);

    // Configure grid and block dimensions
    dim3 block_dim(threads_per_block);
    dim3 grid_dim((n + threads_per_block - 1) / threads_per_block);

    // Calculate shared memory size
    size_t shared_mem_size = (threads_per_block + 2 * R) * sizeof(float) +  // For shared_image
                         (2 * R + 1) * sizeof(float) +                 // For shared_mask
                         threads_per_block * sizeof(float);
    // Launch kernel
    stencil_kernel<<<grid_dim, block_dim, shared_mem_size>>>(d_image, d_mask, d_output, n, R);

    cudaDeviceSynchronize(); // Wait for kernel execution to complete
    // Copy result back to host memory
    cudaMemcpy(output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(d_image);
    cudaFree(d_mask);
    cudaFree(d_output);
}

