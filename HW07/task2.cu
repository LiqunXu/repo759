#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include "reduce.cuh"

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: ./task2 N threads_per_block" << std::endl;
        return 1;
    }

    // Parse command-line arguments
    unsigned int N = atoi(argv[1]);
    unsigned int threads_per_block = atoi(argv[2]);

    if (N == 0 || threads_per_block == 0) {
        std::cerr << "Error: N and threads_per_block must be positive integers." << std::endl;
        return 1;
    }

    // Allocate and initialize host memory
    std::vector<float> h_data(N);
    for (unsigned int i = 0; i < N; ++i) {
        h_data[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f; // Random values in [-1, 1]
    }

    // Allocate device memory
    float *d_input, *d_output;
    unsigned int output_size = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_input, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Timing setup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Perform reduction
    cudaEventRecord(start);
    reduce(&d_input, &d_output, N, threads_per_block);
    cudaEventRecord(stop);

    // Copy the result back to host
    float result;
    cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // Timing
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print the result and timing
    std::cout << "Sum: " << result << std::endl;
    std::cout << "Time taken (ms): " << milliseconds << std::endl;

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
