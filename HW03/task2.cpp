#include <iostream>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include "convolution.h"

// Function to generate a random float between a given range
float random_float(float min, float max) {
    return min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max - min)));
}

int main(int argc, char *argv[]) {
    // Make sure the correct number of arguments is provided
    if (argc != 3) {
        std::cerr << "Usage: ./task2 <n> <t>" << std::endl;
        return 1;
    }

    // Read command line arguments
    std::size_t n = std::stoul(argv[1]); // Matrix size n x n
    int t = std::stoi(argv[2]); // Number of threads

    // Initialize random seed
    srand(static_cast<unsigned int>(time(nullptr)));

    // Allocate memory for image and output (n x n)
    float *image = new float[n * n];
    float *output = new float[n * n];

    // Create random n x n image with values between -10.0 and 10.0
    for (std::size_t i = 0; i < n * n; ++i) {
        image[i] = random_float(-10.0, 10.0);
    }

    // Create a 3x3 mask with random values between -1.0 and 1.0
    std::size_t m = 3; // mask size is 3x3
    float mask[9];
    for (std::size_t i = 0; i < m * m; ++i) {
        mask[i] = random_float(-1.0, 1.0);
    }

    // Set the number of OpenMP threads
    omp_set_num_threads(t);

    // Start timing the convolution process
    double start_time = omp_get_wtime();

    // Call the parallelized convolve function
    convolve(image, output, n, mask, m);

    // End timing
    double end_time = omp_get_wtime();

    // Print the first element of the resulting output array
    std::cout << output[0] << std::endl;

    // Print the last element of the resulting output array
    std::cout << output[n * n - 1] << std::endl;

    // Print the time taken to run the convolve function in milliseconds
    double time_taken = (end_time - start_time) * 1000.0; // Convert to milliseconds
    std::cout << time_taken << std::endl;

    // Clean up memory
    delete[] image;
    delete[] output;

    return 0;
}
