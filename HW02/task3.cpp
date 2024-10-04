#include "matmul.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib> // For std::rand() and std::srand()
#include <ctime>   // For std::time()

using std::cout;
using std::endl;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main() {
    const unsigned int n = 1024; // Size of matrices (at least 1000x1000)

    // Allocate memory for matrices A, B, and C
    std::vector<double> A(n * n), B(n * n);
    double *C1 = new double[n * n];
    double *C2 = new double[n * n];
    double *C3 = new double[n * n];
    double *C4 = new double[n * n];

    // Seed the random number generator
    std::srand(static_cast<unsigned int>(std::time(0)));

    // Fill A and B with random values
    for (unsigned int i = 0; i < n * n; ++i) {
        A[i] = static_cast<double>(std::rand()) / RAND_MAX;
        B[i] = static_cast<double>(std::rand()) / RAND_MAX;
    }

    // Measure and execute mmul1
    auto start = high_resolution_clock::now();
    mmul1(A.data(), B.data(), C1, n);
    auto end = high_resolution_clock::now();
    duration<double, std::milli> duration_sec = end - start;
    cout << n << endl;
    cout << duration_sec.count() << endl;
    cout << C1[n * n - 1] << endl;

    // Measure and execute mmul2
    start = high_resolution_clock::now();
    mmul2(A.data(), B.data(), C2, n);
    end = high_resolution_clock::now();
    duration_sec = end - start;
    cout << duration_sec.count() << endl;
    cout << C2[n * n - 1] << endl;

    // Measure and execute mmul3
    start = high_resolution_clock::now();
    mmul3(A.data(), B.data(), C3, n);
    end = high_resolution_clock::now();
    duration_sec = end - start;
    cout << duration_sec.count() << endl;
    cout << C3[n * n - 1] << endl;

    // Measure and execute mmul4
    start = high_resolution_clock::now();
    mmul4(A, B, C4, n);
    end = high_resolution_clock::now();
    duration_sec = end - start;
    cout << duration_sec.count() << endl;
    cout << C4[n * n - 1] << endl;

    // Deallocate memory
    delete[] C1;
    delete[] C2;
    delete[] C3;
    delete[] C4;

    return 0;
}

