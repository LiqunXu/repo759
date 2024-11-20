#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include "stencil.cuh"

// Function to fill an array with random values in the range [-1, 1]
void fillArray(float* array, unsigned int size) {
    for (unsigned int i = 0; i < size; i++) {
        array[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    }
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: ./task2 n R threads_per_block\n";
        return -1;
    }

    // Parse command-line arguments
    unsigned int n = static_cast<unsigned int>(std::atoi(argv[1]));
    unsigned int R = static_cast<unsigned int>(std::atoi(argv[2]));
    unsigned int threads_per_block = static_cast<unsigned int>(std::atoi(argv[3]));

    if (n <= 0 || R <= 0 || threads_per_block <= 0) {
        std::cerr << "All input parameters must be positive integers.\n";
        return -1;
    }

    // Allocate host memory
    float* image = new float[n];
    float* mask = new float[2 * R + 1];
    float* output = new float[n];

    // Fill input arrays with random values
    fillArray(image, n);
    fillArray(mask, 2 * R + 1);

    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing
    cudaEventRecord(start);

    // Call the stencil function
    stencil(image, mask, output, n, R, threads_per_block);

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print the last element of the output array
    std::cout << "Last element of the output array: " << output[n - 1] << "\n";

    // Print the execution time
    std::cout << "Execution time: " << milliseconds << " ms\n";

    // Cleanup
    delete[] image;
    delete[] mask;
    delete[] output;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
