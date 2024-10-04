#include "scan.h"

void scan(const float *arr, float *output, std::size_t n) {
    if (n == 0) return; // Handle the case where the input array is empty

    // Start with the first element
    output[0] = arr[0];
    
    // Iterate through the array to compute the inclusive scan
    for (std::size_t i = 1; i < n; ++i) {
        output[i] = output[i - 1] + arr[i];
    }
}
