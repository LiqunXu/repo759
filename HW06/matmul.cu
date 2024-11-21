#include "matmul.cuh"
#include <cuda_runtime.h>
//#include <cstdio>
#include <iostream>

__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n) {
    size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    size_t size = n * n;

    if (pos < size) {
	float value = 0.0f;
	int r = pos / n;
	int c = pos % n;
        for (size_t k = 0; k < n; k++) {
            value += A[r * n + k] * B[k * n + c];
        }
        C[r * n + c] = value;
    }
}	

void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block) {

    size_t num_block = (threads_per_block - 1 + n * n) / threads_per_block;

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, n * n * sizeof(float));
    cudaMalloc((void**)&d_B, n * n * sizeof(float));
    cudaMalloc((void**)&d_C, n * n * sizeof(float));
	
    cudaMemcpy(d_A, A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, n*n*sizeof(float));

    matmul_kernel<<<num_block, threads_per_block>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();
    
    cudaMemcpy(C, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
#include "matmul.cuh"
#include <cuda_runtime.h>
#include <iostream>

// CUDA kernel for matrix multiplication
__global__ void matmul_kernel(const float* matrix_A, const float* matrix_B, float* matrix_C, size_t matrix_size) {
    // Calculate the thread's unique global position
    size_t thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = matrix_size * matrix_size;

    // Check if the thread's position is within the bounds of the matrix
    if (thread_pos < total_elements) {
        float temp_value = 0.0f; // Accumulator for the resulting matrix value
        int row = thread_pos / matrix_size; // Row index
        int col = thread_pos % matrix_size; // Column index
        
        // Perform the dot product for the row of A and column of B
        for (size_t k = 0; k < matrix_size; k++) {
            temp_value += matrix_A[row * matrix_size + k] * matrix_B[k * matrix_size + col];
        }
        matrix_C[row * matrix_size + col] = temp_value; // Write the result to the output matrix
    }
}

// Host function for matrix multiplication
void matmul(const float* matrix_A, const float* matrix_B, float* matrix_C, size_t matrix_size, unsigned int threads_per_block) {
    // Calculate the number of blocks needed
    size_t num_blocks = (threads_per_block - 1 + matrix_size * matrix_size) / threads_per_block;

    // Device pointers for input and output matrices
    float *device_A, *device_B, *device_C;
    cudaMalloc((void**)&device_A, matrix_size * matrix_size * sizeof(float));
    cudaMalloc((void**)&device_B, matrix_size * matrix_size * sizeof(float));
    cudaMalloc((void**)&device_C, matrix_size * matrix_size * sizeof(float));
    
    // Copy input matrices from host to device
    cudaMemcpy(device_A, matrix_A, matrix_size * matrix_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, matrix_B, matrix_size * matrix_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(device_C, 0, matrix_size * matrix_size * sizeof(float)); // Initialize the output matrix to zero

    // Launch the CUDA kernel
    matmul_kernel<<<num_blocks, threads_per_block>>>(device_A, device_B, device_C, matrix_size);
    cudaDeviceSynchronize(); // Ensure all threads complete execution
    
    // Copy the result back from device to host
    cudaMemcpy(matrix_C, device_C, matrix_size * matrix_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free the allocated device memory
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);
}
