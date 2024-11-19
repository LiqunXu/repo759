#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include "matmul.cuh"

// Function to fill a matrix with random values in the range [-1, 1]
void fillMatrix(float* matrix, int n) {
    for (int i = 0; i < n * n; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: ./task1 n threads_per_block\n";
        return -1;
    }

    // Parse command-line arguments
    int n = std::atoi(argv[1]);
    int threads_per_block = std::atoi(argv[2]);

    if (n <= 0 || threads_per_block <= 0) {
        std::cerr << "Matrix size and threads per block must be positive integers.\n";
        return -1;
    }

    // Allocate host memory
    float *A = new float[n * n];
    float *B = new float[n * n];
    float *C = new float[n * n];

    // Fill matrices with random values
    fillMatrix(A, n);
    fillMatrix(B, n);

    // Measure execution time using CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing
    cudaEventRecord(start);

    // Call the matrix multiplication function
    matmul(A, B, C, n, threads_per_block);

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print the last element of the resulting matrix
    std::cout << "Last element of the resulting matrix: " << C[n * n - 1] << "\n";

    // Print the execution time
    std::cout << "Execution time: " << milliseconds << " ms\n";

    // Clean up
    delete[] A;
    delete[] B;
    delete[] C;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
