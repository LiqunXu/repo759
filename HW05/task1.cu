#include <cstdio>
#include <cuda_runtime.h>

// CUDA kernel to compute factorials
__global__ void computeFactorial() {
    int idx = threadIdx.x; // Thread index from 0 to 7
    if (idx < 8) {
        int factorial = 1;
        for (int i = 1; i <= (idx + 1); i++) {
            factorial *= i;
        }
        printf("%d!=%d\n", idx + 1, factorial);
    }
}

int main() {
    // Launch the kernel with 1 block and 8 threads
    computeFactorial<<<1, 8>>>();

    // Synchronize to ensure the kernel has finished execution
    cudaDeviceSynchronize();

    // Return success
    return 0;
}
