#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>
#include "matmul.h" // Header for mmul function

int main(int argc, char* argv[]) {
    // Ensure we have enough command-line arguments
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix dimension n> <number of threads t>" << std::endl;
        return 1;
    }

    // Parse command-line arguments
    std::size_t n = std::stoul(argv[1]); // matrix dimension n
    int num_threads = std::stoi(argv[2]); // number of threads t

    // Set number of threads for OpenMP
    omp_set_num_threads(num_threads);

    // Initialize random number generator for generating random float numbers
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // Create matrices A, B, and C as 1D arrays in row-major order
    std::vector<float> A(n * n), B(n * n), C(n * n, 0.0f);

    // Fill matrices A and B with random float numbers between 0.0 and 1.0
    for (std::size_t i = 0; i < n * n; ++i) {
        A[i] = static_cast<float>(dis(gen));
        B[i] = static_cast<float>(dis(gen));
    }

    // Measure the time before matrix multiplication
    auto start_time = std::chrono::high_resolution_clock::now();

    // Perform matrix multiplication C = A * B using the parallel mmul function
    mmul(A.data(), B.data(), C.data(), n);

    // Measure the time after matrix multiplication
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // Print the first and last element of matrix C
    std::cout << C[0] << std::endl;
    std::cout << C[n * n - 1] << std::endl;

    // Print the time taken in milliseconds
    std::cout << duration << std::endl;

    return 0;
}

