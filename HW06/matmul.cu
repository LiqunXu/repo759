#include "matmul.cuh"
#include <cuda_runtime.h>
#include <iostream>

// CUDA kernel for matrix multiplication
// Computes matrix C as the product of matrices A and B, all stored in row-major format
__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n) {
    // Calculate the global position of the current thread
    size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    size_t size = n * n;

    // Ensure the thread is within bounds
    if (pos < size) {
	float value = 0.0f;
	int r = pos / n; // Row index
	int c = pos % n; // Column index

        // Perform the dot product for the current element
        for (size_t k = 0; k < n; k++) {
            value += A[r * n + k] * B[k * n + c];
        }
        C[r * n + c] = value;
    }
}	
// Host function to perform matrix multiplication using CUDA
// Transfers data between host and device, launches kernel, and retrieves the result
void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block) {
    // Calculate the required number of blocks to cover all elements
    size_t num_block = (threads_per_block - 1 + n * n) / threads_per_block;

    // Device pointers for matrices
    float *device_A, *device_B, *device_C;
    // Allocate memory on the GPU for matrices
    cudaMalloc((void**)&device_A, n * n * sizeof(float));
    cudaMalloc((void**)&device_B, n * n * sizeof(float));
    cudaMalloc((void**)&device_C, n * n * sizeof(float));
	
    // Copy input matrices from host to device
    cudaMemcpy(device_A, A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, B, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(device_C, 0, n*n*sizeof(float));

    // Launch the matrix multiplication kernel
    matmul_kernel<<<num_block, threads_per_block>>>(device_A, device_B, device_C, n);

    // Wait for the kernel to finish execution
    cudaDeviceSynchronize();

    // Copy the result matrix back to the host
    cudaMemcpy(C, device_C, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);
}
