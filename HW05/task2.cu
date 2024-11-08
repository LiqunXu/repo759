#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Kernel to compute ax + y and store results in dA
__global__ void computeArray(int *dA, int a) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Calculate global index
    dA[idx] = a * threadIdx.x + blockIdx.x; // Compute ax + y
}

int main() {
    const int numElements = 16; // Total elements (2 blocks * 8 threads)
    int hA[numElements];        // Host array to store the result

    // Allocate memory on the device
    int *dA;
    cudaMalloc(&dA, numElements * sizeof(int));

    // Generate a random integer for 'a'
    int a = rand() % 10 + 1; // Random number between 1 and 10


    // Launch kernel with 2 blocks, 8 threads each
    computeArray<<<2, 8>>>(dA, a);

    // Check for errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        cudaFree(dA);
        return EXIT_FAILURE;
    }

    // Copy results back to the host
    cudaMemcpy(hA, dA, numElements * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result

    for (int i = 0; i < numElements; ++i) {
        printf("%d ", hA[i]);
        printf("\n");
    }


    // Free device memory
    cudaFree(dA);

    return 0;
}
