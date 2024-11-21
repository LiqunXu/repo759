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
