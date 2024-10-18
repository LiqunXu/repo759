#include "msort.h"
#include <algorithm> // For std::copy
#include <cstring>   // For memcpy

// Function to merge two sorted sub-arrays
void merge(int* arr, int* left, std::size_t leftSize, int* right, std::size_t rightSize) {
    std::size_t i = 0, j = 0, k = 0;
    while (i < leftSize && j < rightSize) {
        if (left[i] <= right[j]) {
            arr[k++] = left[i++];
        } else {
            arr[k++] = right[j++];
        }
    }

    // Copy remaining elements from left array
    while (i < leftSize) {
        arr[k++] = left[i++];
    }

    // Copy remaining elements from right array
    while (j < rightSize) {
        arr[k++] = right[j++];
    }
}

// Serial merge sort (for when the array size is below the threshold)
void serialSort(int* arr, const std::size_t n) {
    if (n < 2) return;

    std::size_t mid = n / 2;

    // Create temporary arrays for left and right subarrays
    int* left = new int[mid];
    int* right = new int[n - mid];

    // Copy data to the subarrays
    std::copy(arr, arr + mid, left);
    std::copy(arr + mid, arr + n, right);

    // Recursively sort both halves
    serialSort(left, mid);
    serialSort(right, n - mid);

    // Merge sorted halves
    merge(arr, left, mid, right, n - mid);

    // Clean up
    delete[] left;
    delete[] right;
}

// Parallel merge sort
void msort(int* arr, const std::size_t n, const std::size_t threshold) {
    if (n < threshold) {
        serialSort(arr, n); // Base case: Use serial sort for small arrays
        return;
    }

    std::size_t mid = n / 2;

    // Create temporary arrays for left and right subarrays
    int* left = new int[mid];
    int* right = new int[n - mid];

    // Copy data to the subarrays
    std::copy(arr, arr + mid, left);
    std::copy(arr + mid, arr + n, right);

    // Parallel tasks to sort the two halves
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            #pragma omp task
            msort(left, mid, threshold);

            #pragma omp task
            msort(right, n - mid, threshold);

            #pragma omp taskwait
        }
    }

    // Merge the two sorted halves
    merge(arr, left, mid, right, n - mid);

    // Clean up
    delete[] left;
    delete[] right;
}
