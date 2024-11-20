#include <cuda_runtime.h>
#include "stencil.cuh"

__global__ void stencil_kernel(const float *image, const float *mask, float *output, unsigned int n, unsigned int R) {
    extern __shared__ float shared_memory[];

    // Pointers for shared memory
    float *shared_image = shared_memory;
    float *shared_mask = shared_memory + blockDim.x + 2 * R;

    unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int local_idx = threadIdx.x;

    // Load mask into shared memory (only once per block)
    if (local_idx < 2 * R + 1) {
        shared_mask[local_idx] = mask[local_idx];
    }

    // Load image into shared memory (with ghost elements for the boundary)
    if (global_idx < n) {
        shared_image[R + local_idx] = image[global_idx];
    }
    if (local_idx < R) {
        // Load left boundary
        shared_image[local_idx] = (global_idx >= R) ? image[global_idx - R] : 1.0f;
        // Load right boundary
        unsigned int right_idx = global_idx + blockDim.x;
        shared_image[R + blockDim.x + local_idx] = (right_idx < n) ? image[right_idx] : 1.0f;
    }

    __syncthreads();

    // Perform convolution
    if (global_idx < n) {
        float result = 0.0f;
        for (int j = -static_cast<int>(R); j <= static_cast<int>(R); ++j) {
            result += shared_image[R + local_idx + j] * shared_mask[j + R];
        }
        output[global_idx] = result;
    }
}

void stencil(const float *image, const float *mask, float *output, unsigned int n, unsigned int R, unsigned int threads_per_block) {
    float *d_image, *d_mask, *d_output;

    // Allocate device memory
    cudaMalloc(&d_image, n * sizeof(float));
    cudaMalloc(&d_mask, (2 * R + 1) * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_image, image, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, (2 * R + 1) * sizeof(float), cudaMemcpyHostToDevice);

    // Configure kernel launch parameters
    dim3 threads(threads_per_block);
    dim3 blocks((n + threads_per_block - 1) / threads_per_block);
    size_t shared_memory_size = (threads_per_block + 2 * R) * sizeof(float) + (2 * R + 1) * sizeof(float);

    // Launch the kernel
    stencil_kernel<<<blocks, threads, shared_memory_size>>>(d_image, d_mask, d_output, n, R);

    // Copy the result back to the host
    cudaMemcpy(output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_image);
    cudaFree(d_mask);
    cudaFree(d_output);
}
