// #include <iostream>
// #include <vector>
// #include <random>
// #include <chrono>
// #include <omp.h>
// #include "matmul.h" // Header for mmul function

// int main(int argc, char* argv[]) {
//     // Ensure we have enough command-line arguments
//     if (argc != 3) {
//         std::cerr << "Usage: " << argv[0] << " <matrix dimension n> <number of threads t>" << std::endl;
//         return 1;
//     }

//     // Parse command-line arguments
//     std::size_t n = std::stoul(argv[1]); // matrix dimension n
//     int num_threads = std::stoi(argv[2]); // number of threads t

//     // Set number of threads for OpenMP
//     omp_set_num_threads(num_threads);

//     // Initialize random number generator for generating random float numbers
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_real_distribution<> dis(0.0, 1.0);

//     // Create matrices A, B, and C as 1D arrays in row-major order
//     std::vector<float> A(n * n), B(n * n), C(n * n, 0.0f);

//     // Fill matrices A and B with random float numbers between 0.0 and 1.0
//     for (std::size_t i = 0; i < n * n; ++i) {
//         A[i] = static_cast<float>(dis(gen));
//         B[i] = static_cast<float>(dis(gen));
//     }

//     // Measure the time before matrix multiplication
//     auto start_time = std::chrono::high_resolution_clock::now();

//     // Perform matrix multiplication C = A * B using the parallel mmul function
//     mmul(A.data(), B.data(), C.data(), n);

//     // Measure the time after matrix multiplication
//     auto end_time = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

//     // Print the first and last element of matrix C
//     std::cout << C[0] << std::endl;
//     std::cout << C[n * n - 1] << std::endl;

//     // Print the time taken in milliseconds
//     std::cout << duration << std::endl;

//     return 0;
// }

#include <iostream>
#include <cstdlib>
#include <chrono>
#include "matmul.h"

int main(int argc, char* argv[]) {
    // Ensure we have the correct number of arguments
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size> <num_threads>" << std::endl;
        return 1;
    }

    // Parse command line arguments
    std::size_t n = std::stoi(argv[1]);       // Matrix size (n x n)
    int num_threads = std::stoi(argv[2]);     // Number of threads

    // Allocate memory for matrices A, B, and C (row-major order)
    float* A = new float[n * n];
    float* B = new float[n * n];
    float* C = new float[n * n];

    // Seed for random number generation
    std::srand(std::time(0));

    // Initialize matrices A and B with random float values, and C with zeros
    for (std::size_t i = 0; i < n; i++) {
        for (std::size_t j = 0; j < n; j++) {
            A[i * n + j] = static_cast<float>(std::rand()) / RAND_MAX;  // Fill A with random floats
            B[i * n + j] = static_cast<float>(std::rand()) / RAND_MAX;  // Fill B with random floats
            C[i * n + j] = 0.0f;                                        // Initialize C to zero
        }
    }

    // Set the number of threads for OpenMP
    omp_set_num_threads(num_threads);

    // Start timing using std::chrono
    auto start = std::chrono::high_resolution_clock::now();

    // Perform matrix multiplication C = A * B using the parallel mmul function
    mmul(A, B, C, n);

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the elapsed time in milliseconds
    std::chrono::duration<double, std::milli> elapsed_time = end - start;

    // Output the first element, last element of C, and the elapsed time
    std::cout << C[0] << std::endl;               // First element of matrix C
    std::cout << C[n * n - 1] << std::endl;       // Last element of matrix C
    std::cout << elapsed_time.count() << std::endl;  // Time taken

    // Clean up memory
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
