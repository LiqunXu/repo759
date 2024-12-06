// #include <iostream>
// #include <cuda_runtime.h>
// #include "matmul.cuh"

// template <typename T>
// void fill_matrix(T *matrix, unsigned int n) {
//     for (unsigned int i = 0; i < n * n; i++) {
//         matrix[i] = static_cast<T>(rand() % 100); // Fill with random values between 0 and 99
//     }
// }

// template <typename T>
// void test_matmul(void (*matmul_func)(const T *, const T *, T *, unsigned int, unsigned int),
//                  const std::string &func_name,
//                  unsigned int n, unsigned int block_dim) {
//     // Allocate host memory
//     T *A = new T[n * n];
//     T *B = new T[n * n];
//     T *C = new T[n * n];

//     // Fill host matrices
//     fill_matrix(A, n);
//     fill_matrix(B, n);

//     // Allocate device memory
//     T *d_A, *d_B, *d_C;
//     cudaMallocManaged(&d_A, n * n * sizeof(T));
//     cudaMallocManaged(&d_B, n * n * sizeof(T));
//     cudaMallocManaged(&d_C, n * n * sizeof(T));

//     // Copy data to device
//     cudaMemcpy(d_A, A, n * n * sizeof(T), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_B, B, n * n * sizeof(T), cudaMemcpyHostToDevice);

//     // Record start and stop events
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);

//     // Launch kernel and measure time
//     cudaEventRecord(start);
//     matmul_func(d_A, d_B, d_C, n, block_dim);
//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);

//     // Copy result back to host
//     cudaMemcpy(C, d_C, n * n * sizeof(T), cudaMemcpyDeviceToHost);

//     // Calculate elapsed time
//     float elapsed_time;
//     cudaEventElapsedTime(&elapsed_time, start, stop);

//     // Print results
//     std::cout << "Results for " << func_name << ":\n";
//     std::cout << "First element of C: " << C[0] << "\n";
//     std::cout << "Last element of C: " << C[n * n - 1] << "\n";
//     std::cout << "Elapsed time: " << elapsed_time << " ms\n";

//     // Free device memory
//     cudaFree(d_A);
//     cudaFree(d_B);
//     cudaFree(d_C);

//     // Free host memory
//     delete[] A;
//     delete[] B;
//     delete[] C;

//     // Destroy events
//     cudaEventDestroy(start);
//     cudaEventDestroy(stop);
// }

// int main(int argc, char **argv) {
//     if (argc != 3) {
//         std::cerr << "Usage: ./task1 <n> <block_dim>\n";
//         return 1;
//     }

//     unsigned int n = std::stoi(argv[1]);
//     unsigned int block_dim = std::stoi(argv[2]);

//     std::cout << "Matrix size: " << n << "x" << n << "\n";
//     std::cout << "Block dimension: " << block_dim << "\n";

//     // Test matmul_1 with int
//     test_matmul<int>(matmul_1, "matmul_1 (int)", n, block_dim);

//     // Test matmul_2 with float
//     test_matmul<float>(matmul_2, "matmul_2 (float)", n, block_dim);

//     // Test matmul_3 with double
//     test_matmul<double>(matmul_3, "matmul_3 (double)", n, block_dim);

//     return 0;
// }


#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "matmul.cuh"

// Utility function to initialize matrices
template <typename T>
void initialize_matrix(std::vector<T>& matrix, unsigned int n) {
    for (size_t i = 0; i < matrix.size(); ++i) {
        matrix[i] = static_cast<T>(rand()) / RAND_MAX; // Random values between 0 and 1
    }
}

// Utility function to print timing and matrix results
template <typename T>
void print_results(const T* C, unsigned int n, float time_ms) {
    std::cout << "First element: " << C[0] << std::endl;
    std::cout << "Last element: " << C[n * n - 1] << std::endl;
    std::cout << "Execution time (ms): " << time_ms << std::endl;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: ./task1 n block_dim" << std::endl;
        return 1;
    }

    unsigned int n = atoi(argv[1]);         // Matrix size
    unsigned int block_dim = atoi(argv[2]); // Block dimension

    if (n <= 0 || block_dim <= 0) {
        std::cerr << "Matrix size and block dimension must be positive integers." << std::endl;
        return 1;
    }

    // Allocate and initialize matrices
    std::vector<int> A_int(n * n), B_int(n * n), C_int(n * n);
    std::vector<float> A_float(n * n), B_float(n * n), C_float(n * n);
    std::vector<double> A_double(n * n), B_double(n * n), C_double(n * n);

    initialize_matrix(A_int, n);
    initialize_matrix(B_int, n);
    initialize_matrix(A_float, n);
    initialize_matrix(B_float, n);
    initialize_matrix(A_double, n);
    initialize_matrix(B_double, n);

    // Timing variables
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Perform int matrix multiplication
    cudaEventRecord(start);
    matmul_1(A_int.data(), B_int.data(), C_int.data(), n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_int = 0.0f;
    cudaEventElapsedTime(&time_int, start, stop);
    print_results(C_int.data(), n, time_int);

    // Perform float matrix multiplication
    cudaEventRecord(start);
    matmul_2(A_float.data(), B_float.data(), C_float.data(), n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_float = 0.0f;
    cudaEventElapsedTime(&time_float, start, stop);
    print_results(C_float.data(), n, time_float);

    // Perform double matrix multiplication
    cudaEventRecord(start);
    matmul_3(A_double.data(), B_double.data(), C_double.data(), n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_double = 0.0f;
    cudaEventElapsedTime(&time_double, start, stop);
    print_results(C_double.data(), n, time_double);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
