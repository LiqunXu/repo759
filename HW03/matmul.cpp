#include "matmul.h"
#include <omp.h>
#include <cstddef>

void mmul(const float* A, const float* B, float* C, const std::size_t n) {
    // Initialize the output matrix C to zero
    for (std::size_t i = 0; i < n * n; ++i) {
        C[i] = 0.0f;
    }

    // Parallelize the outer loop using OpenMP
    #pragma omp parallel for collapse(2)
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < n; ++k) {
            for (std::size_t j = 0; j < n; ++j) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}
