#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <chrono>
#include "msort.h"

int main(int argc, char* argv[]) {
    // Check for the correct number of command-line arguments
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " n t ts" << std::endl;
        return 1;
    }

    // Parse command-line arguments
    std::size_t n = std::stoul(argv[1]);  // Size of the array
    int t = std::stoi(argv[2]);           // Number of threads
    std::size_t ts = std::stoul(argv[3]); // Threshold

    // Set the number of threads
    omp_set_num_threads(t);

    // Create and fill the array with random integers in the range [-1000, 1000]
    int* arr = new int[n];
    srand(time(0));  // Seed for random number generator
    for (std::size_t i = 0; i < n; ++i) {
        arr[i] = rand() % 2001 - 1000; // Generates numbers between [-1000, 1000]
    }

    // Record start time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Apply msort to the array
    msort(arr, n, ts);

    // Record end time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_time = end_time - start_time;

    // Print the first and last elements of the sorted array
    std::cout << arr[0] << std::endl;         // First element
    std::cout << arr[n - 1] << std::endl;     // Last element
    std::cout << elapsed_time.count() << std::endl; // Time in milliseconds

    // Clean up
    delete[] arr;

    return 0;
}

