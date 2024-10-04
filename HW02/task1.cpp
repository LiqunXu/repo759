#include "scan.h"
#include <iostream>
#include <chrono>
#include <cstdlib> // For std::rand() and std::srand()
#include <ctime>   // For std::time()

using std::cout;
using std::endl;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "Usage: ./task1 n" << endl;
        return 1;
    }

    // Parse the command line argument to get n
    std::size_t n = std::stoi(argv[1]);

    // Allocate memory for the input and output arrays
    float *arr = new float[n];
    float *output = new float[n];

    // Seed the random number generator
    std::srand(static_cast<unsigned int>(std::time(0)));

    // Generate n random float numbers between -1.0 and 1.0
    for (std::size_t i = 0; i < n; ++i) {
        arr[i] = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;
    }

    // Measure the time taken by the scan function
    high_resolution_clock::time_point start = high_resolution_clock::now();
    
    // Call the scan function
    scan(arr, output, n);
    
    high_resolution_clock::time_point end = high_resolution_clock::now();
    
    // Calculate the duration in milliseconds
    duration<double, std::milli> duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    
    // Print the results
    cout << duration_sec.count() << endl;         // iii) Print time taken by the scan function in milliseconds
    cout << output[0] << endl;                    // iv) Print the first element of the output array
    cout << output[n - 1] << endl;                // v) Print the last element of the output array

    // Deallocate memory
    delete[] arr;
    delete[] output;

    return 0;
}
