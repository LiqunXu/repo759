#include "matmul.cuh"
#include <cuda_runtime.h>
#include <iostream>

// CUDA kernel for tiled matrix multiplication
template <typename T>
__global__ void matmul_kernel(const T* A, const T* B, T* C, unsigned int n, unsigned int block_dim) {
    extern __shared__ char shared_mem[];
    T* tile_A = reinterpret_cast<T*>(shared_mem);
    T* tile_B = reinterpret_cast<T*>(shared_mem + block_dim * block_dim * sizeof(T));

    unsigned int row = threadIdx.y + blockIdx.y * block_dim;
    unsigned int col = threadIdx.x + blockIdx.x * block_dim;
    unsigned int local_row = threadIdx.y;
    unsigned int local_col = threadIdx.x;

    T value = 0;

    for (unsigned int t = 0; t < (n + block_dim - 1) / block_dim; ++t) {
        if (row < n && t * block_dim + local_col < n)
            tile_A[local_row * block_dim + local_col] = A[row * n + t * block_dim + local_col];
        else
            tile_A[local_row * block_dim + local_col] = 0;

        if (col < n && t * block_dim + local_row < n)
            tile_B[local_row * block_dim + local_col] = B[(t * block_dim + local_row) * n + col];
        else
            tile_B[local_row * block_dim + local_col] = 0;

        __syncthreads();

        for (unsigned int k = 0; k < block_dim; ++k) {
            value += tile_A[local_row * block_dim + k] * tile_B[k * block_dim + local_col];
        }

        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = value;
}

template <typename T>
__host__ void matmul(const T* A, const T* B, T* C, unsigned int n, unsigned int block_dim) {
    // Allocate memory on the device
    T *device_A, *device_B, *device_C;
    size_t size = n * n * sizeof(T);
    cudaMalloc((void**)&device_A, size);
    cudaMalloc((void**)&device_B, size);
    cudaMalloc((void**)&device_C, size);

    // Copy matrices to the device
    cudaMemcpy(device_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, B, size, cudaMemcpyHostToDevice);

    // Configure kernel dimensions
    dim3 threads_per_block(block_dim, block_dim);
    dim3 num_blocks((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);

    // Launch the kernel
    size_t shared_mem_size = 2 * block_dim * block_dim * sizeof(T);
    matmul_kernel<T><<<num_blocks, threads_per_block, shared_mem_size>>>(device_A, device_B, device_C, n, block_dim);

    // Wait for completion
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(C, device_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);
}

// Define the required functions as per matmul.cuh
__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim) {
    matmul<int>(A, B, C, n, block_dim);
}

__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n, unsigned int block_dim) {
    matmul<float>(A, B, C, n, block_dim);
}

__host__ void matmul_3(const double *A, const double *B, double *C, unsigned int n, unsigned int block_dim) {
    matmul<double>(A, B, C, n, block_dim);
}
