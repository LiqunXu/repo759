// #include "matmul.h"

// // mmul1: Standard order of loops: (i, j, k)
// void mmul1(const double* A, const double* B, double* C, const unsigned int n) {
//     for (unsigned int i = 0; i < n; ++i) {
//         for (unsigned int j = 0; j < n; ++j) {
//             C[i * n + j] = 0.0; // Initialize C[i][j] to zero
//             for (unsigned int k = 0; k < n; ++k) {
//                 C[i * n + j] += A[i * n + k] * B[k * n + j];
//             }
//         }
//     }
// }

// // mmul2: Change the order of inner loops: (i, k, j)
// void mmul2(const double* A, const double* B, double* C, const unsigned int n) {
//     for (unsigned int i = 0; i < n; ++i) {
//         for (unsigned int k = 0; k < n; ++k) {
//             for (unsigned int j = 0; j < n; ++j) {
//                 C[i * n + j] += A[i * n + k] * B[k * n + j];
//             }
//         }
//     }
// }

// // mmul3: Change the order of the loops: (j, k, i)
// void mmul3(const double* A, const double* B, double* C, const unsigned int n) {
//     for (unsigned int j = 0; j < n; ++j) {
//         for (unsigned int k = 0; k < n; ++k) {
//             for (unsigned int i = 0; i < n; ++i) {
//                 C[i * n + j] += A[i * n + k] * B[k * n + j];
//             }
//         }
//     }
// }

// // mmul4: Uses std::vector<double> for A and B
// void mmul4(const std::vector<double>& A, const std::vector<double>& B, double* C, const unsigned int n) {
//     for (unsigned int i = 0; i < n; ++i) {
//         for (unsigned int j = 0; j < n; ++j) {
//             C[i * n + j] = 0.0; // Initialize C[i][j] to zero
//             for (unsigned int k = 0; k < n; ++k) {
//                 C[i * n + j] += A[i * n + k] * B[k * n + j];
//             }
//         }
//     }
// }


#include "matmul.h"
#include <iostream>

void mmul1(const double* A, const double* B, double* C, const unsigned int n) {
	unsigned int i, j, k;
	for (i = 0; i < n; i++){
		for (j = 0; j < n; j++) {
			for (k = 0; k < n; k++) {
				C[i * n + j] += A[i * n + k] * B[k * n + j];
			}
		}
	}
}


void mmul2(const double* A, const double* B, double* C, const unsigned int n) {
	unsigned int i, j, k;
	for (i = 0; i < n; i++){
		for (k = 0; k < n; k++) {
			for (j = 0; j < n; j++) {
				C[i * n + j] += A[i * n + k] * B[k * n + j];
			}
		}
	}
}


void mmul3(const double* A, const double* B, double* C, const unsigned int n) {
	unsigned int i, j, k;
	for (j = 0; j < n; j++) {
		for (k = 0; k < n; k++) {
			for (i = 0; i < n; i++){
				C[i * n + j] += A[i * n + k] * B[k * n + j];
			}
		}
	}
}


void mmul4(const std::vector<double>& A, const std::vector<double>& B, double* C, const unsigned int n) {
	unsigned int i, j, k;
	for (i = 0; i < n; i++){
		for (j = 0; j < n; j++) {
			for (k = 0; k < n; k++) {
				
				C[i * n + j] += A[i * n + k] * B[k * n + j];
			}
		}
	}
}
