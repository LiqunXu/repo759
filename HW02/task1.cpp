// #include <cstdlib>      // For rand(), srand(), and atoi()
// #include <ctime>        // For clock()
// #include "scan.h"
// // The std::chrono namespace provides timer functions in C++
// #include <chrono>

// // std::ratio provides easy conversions between metric units
// #include <ratio>

// // iostream is not needed for timers, but we need it for cout
// #include <iostream>

// // not needed for timers, provides std::pow function
// #include <cmath>

// // Provide some namespace shortcuts
// using std::cout;
// using std::chrono::high_resolution_clock;
// using std::chrono::duration;

// int main(int argc, char* argv[]) {
//     high_resolution_clock::time_point start;
//     high_resolution_clock::time_point end;
//     duration<double, std::milli> duration_sec;
//     // Check if the correct number of arguments is provided
//     if (argc != 2) {
//         std::cerr << "Usage: " << argv[0] << " n" << std::endl;
//         return 1;
//     }

//     // Read the size of the array from the first command line argument
//     std::size_t n = std::atoi(argv[1]);
//     if (n <= 0) {
//         std::cerr << "Error: n must be a positive integer" << std::endl;
//         return 1;
//     }

//     // Create an array of n random float numbers between -1.0 and 1.0
//     float* arr = new float[n];
//     float* output = new float[n];
//     srand(static_cast<unsigned int>(time(nullptr))); // Seed the random number generator

//     for (std::size_t i = 0; i < n; ++i) {
//         arr[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f; // Generates random float between -1.0 and 1.0
//     }

//     // Get the starting timestamp
//     start = high_resolution_clock::now();
//     scan(arr, output, n);
//     // Get the ending timestamp
//     end = high_resolution_clock::now();

//     // Convert the calculated duration to a double using the standard library
//     duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    
//     // Print the results
//     // Durations are converted to milliseconds already thanks to std::chrono::duration_cast
//     cout << "Total time: " << duration_sec.count() << "ms\n";
//     std::cout << output[0] << std::endl;          // Print the first element of the scanned array
//     std::cout << output[n - 1] << std::endl;      // Print the last element of the scanned array

//     // Deallocate memory
//     delete[] arr;
//     delete[] output;

//     return 0;
// }

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
