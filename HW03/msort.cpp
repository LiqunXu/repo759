// #include "msort.h"
// #include <algorithm> // For std::copy
// #include <cstring>   // For memcpy

// // Function to merge two sorted sub-arrays
// void merge(int* arr, int* left, std::size_t leftSize, int* right, std::size_t rightSize) {
//     std::size_t i = 0, j = 0, k = 0;
//     while (i < leftSize && j < rightSize) {
//         if (left[i] <= right[j]) {
//             arr[k++] = left[i++];
//         } else {
//             arr[k++] = right[j++];
//         }
//     }

//     // Copy remaining elements from left array
//     while (i < leftSize) {
//         arr[k++] = left[i++];
//     }

//     // Copy remaining elements from right array
//     while (j < rightSize) {
//         arr[k++] = right[j++];
//     }
// }

// // Serial merge sort (for when the array size is below the threshold)
// void serialSort(int* arr, const std::size_t n) {
//     if (n < 2) return;

//     std::size_t mid = n / 2;

//     // Create temporary arrays for left and right subarrays
//     int* left = new int[mid];
//     int* right = new int[n - mid];

//     // Copy data to the subarrays
//     std::copy(arr, arr + mid, left);
//     std::copy(arr + mid, arr + n, right);

//     // Recursively sort both halves
//     serialSort(left, mid);
//     serialSort(right, n - mid);

//     // Merge sorted halves
//     merge(arr, left, mid, right, n - mid);

//     // Clean up
//     delete[] left;
//     delete[] right;
// }

// // Parallel merge sort
// void msort(int* arr, const std::size_t n, const std::size_t threshold) {
//     if (n < threshold) {
//         serialSort(arr, n); // Base case: Use serial sort for small arrays
//         return;
//     }

//     std::size_t mid = n / 2;

//     // Create temporary arrays for left and right subarrays
//     int* left = new int[mid];
//     int* right = new int[n - mid];

//     // Copy data to the subarrays
//     std::copy(arr, arr + mid, left);
//     std::copy(arr + mid, arr + n, right);

//     // Parallel tasks to sort the two halves
//     #pragma omp parallel
//     {
//         #pragma omp single nowait
//         {
//             #pragma omp task
//             msort(left, mid, threshold);

//             #pragma omp task
//             msort(right, n - mid, threshold);

//             #pragma omp taskwait
//         }
//     }

//     // Merge the two sorted halves
//     merge(arr, left, mid, right, n - mid);

//     // Clean up
//     delete[] left;
//     delete[] right;
// }


// msort.cpp
#include "msort.h"
#include <algorithm> // for std::sort
#include <vector>

// Helper function to merge two sorted halves
void merge(int* arr, std::size_t left, std::size_t mid, std::size_t right) {
    std::size_t n1 = mid - left + 1;
    std::size_t n2 = right - mid;

    // Create temporary arrays for left and right halves
    std::vector<int> L(n1), R(n2);
    for (std::size_t i = 0; i < n1; ++i) L[i] = arr[left + i];
    for (std::size_t i = 0; i < n2; ++i) R[i] = arr[mid + 1 + i];

    // Merge the temporary arrays back into arr[l..r]
    std::size_t i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k++] = L[i++];
        } else {
            arr[k++] = R[j++];
        }
    }

    // Copy the remaining elements of L[], if any
    while (i < n1) arr[k++] = L[i++];

    // Copy the remaining elements of R[], if any
    while (j < n2) arr[k++] = R[j++];
}

// Recursive parallel merge sort function
void parallel_merge_sort(int* arr, std::size_t left, std::size_t right, std::size_t threshold) {
    if (left < right) {
        std::size_t mid = left + (right - left) / 2;

        // If the size of the subarray is less than the threshold, use serial sort
        if ((right - left + 1) <= threshold) {
            std::sort(arr + left, arr + right + 1);
        } else {
            // Parallel recursive sorting of left and right halves
            #pragma omp task shared(arr)
            parallel_merge_sort(arr, left, mid, threshold);

            #pragma omp task shared(arr)
            parallel_merge_sort(arr, mid + 1, right, threshold);

            // Wait for both halves to be sorted
            #pragma omp taskwait

            // Merge the sorted halves
            merge(arr, left, mid, right);
        }
    }
}

// Public msort function as defined in the header
void msort(int* arr, const std::size_t n, const std::size_t threshold) {
    // Start parallel region
    #pragma omp parallel
    {
        #pragma omp single
        {
            parallel_merge_sort(arr, 0, n - 1, threshold);
        }
    }
}
