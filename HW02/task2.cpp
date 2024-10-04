#include "convolution.h"
#include <iostream>
#include <chrono>
#include <cstdlib> // For std::rand() and std::srand()
#include <ctime>   // For std::time()

using std::cout;
using std::endl;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char **argv) {
    // Ensure correct number of arguments are provided
    if (argc != 3) {
        cout << "Usage: ./task2 n m" << endl;
        return 1;
    }

    // Parse command line arguments
    std::size_t n = std::stoi(argv[1]);
    std::size_t m = std::stoi(argv[2]);

    // Ensure m is an odd number
    if (m % 2 == 0) {
        cout << "Error: m must be an odd number." << endl;
        return 1;
    }

    // Allocate memory for the input image and mask, and the output result
    float *image = new float[n * n];
    float *mask = new float[m * m];
    float *output = new float[n * n];

    // Seed the random number generator
    std::srand(static_cast<unsigned int>(std::time(0)));

    // Generate n x n image matrix with random float numbers between -10.0 and 10.0
    for (std::size_t i = 0; i < n * n; ++i) {
        image[i] = static_cast<float>(std::rand()) / RAND_MAX * 20.0f - 10.0f;
    }

    // Generate m x m mask matrix with random float numbers between -1.0 and 1.0
    for (std::size_t i = 0; i < m * m; ++i) {
        mask[i] = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;
    }

    // Measure the time taken by the convolve function
    high_resolution_clock::time_point start = high_resolution_clock::now();
    
    // Apply the convolution
    convolve(image, output, n, mask, m);
    
    high_resolution_clock::time_point end = high_resolution_clock::now();

    // Calculate the duration in milliseconds
    duration<double, std::milli> duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    
    // Print the results
    cout << duration_sec.count() << endl;         // iv) Print time taken by the convolve function in milliseconds
    cout << output[0] << endl;                    // v) Print the first element of the output array
    cout << output[n * n - 1] << endl;            // vi) Print the last element of the output array

    // Deallocate memory
    delete[] image;
    delete[] mask;
    delete[] output;

    return 0;
}
