#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "vscale.cuh"

// Function to initialize arrays with random values
void initializeArray(float *arr, unsigned int n, float lower, float upper) {
    for (unsigned int i = 0; i < n; ++i) {
        arr[i] = lower + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (upper - lower)));
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: ./task3 <n>\n");
        return EXIT_FAILURE;
    }

    unsigned int n = atoi(argv[1]);
    size_t size = n * sizeof(float);

    // Allocate memory on the host
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);

    // Initialize host arrays with random numbers
    initializeArray(h_a, n, -10.0f, 10.0f); // Range for array a: [-10.0, 10.0]
    initializeArray(h_b, n, 0.0f, 1.0f);    // Range for array b: [0.0, 1.0]

    // Allocate memory on the device
    float *d_a, *d_b;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(512);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start recording time
    cudaEventRecord(start);

    // Launch the kernel
    vscale<<<gridDim, blockDim>>>(d_a, d_b, n);

    // Stop recording time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy results back to the host
    cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost);

    // Print the elapsed time and the first and last elements of the result
    printf("Time taken to execute the kernel: %f ms\n", milliseconds);
    printf("First element of resulting array: %f\n", h_b[0]);
    printf("Last element of resulting array: %f\n", h_b[n - 1]);

    // Free device and host memory
    cudaFree(d_a);
    cudaFree(d_b);
    free(h_a);
    free(h_b);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
