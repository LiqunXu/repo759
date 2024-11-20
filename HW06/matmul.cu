#include <cuda_runtime.h>
#include "matmul.cuh"

// CUDA kernel for matrix multiplication
__global__ void matmul_kernel(const float *A, const float *B, float *C, size_t n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float value = 0.0f;
        for (int i = 0; i < n; i++) {
            value += A[row * n + i] * B[i * n + col];
        }
        C[row * n + col] = value;
    }
}

// Host function for matrix multiplication
void matmul(const float *A, const float *B, float *C, size_t n, unsigned int threads_per_block) {
    float *d_A, *d_B, *d_C;

    // Allocate device memory
    cudaMalloc(&d_A, n * n * sizeof(float));
    cudaMalloc(&d_B, n * n * sizeof(float));
    cudaMalloc(&d_C, n * n * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * n * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 threads(threads_per_block, threads_per_block);
    dim3 blocks((n + threads_per_block - 1) / threads_per_block,
                (n + threads_per_block - 1) / threads_per_block);

    // Launch the kernel
    matmul_kernel<<<blocks, threads>>>(d_A, d_B, d_C, n);

    // Copy the result back to the host
    cudaMemcpy(C, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
