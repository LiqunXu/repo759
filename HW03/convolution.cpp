#include "convolution.h"

// Function to apply a mask to an image using convolution in parallel with OpenMP
void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m) {
    // Calculate the offset, which is half the size of the mask
    std::size_t offset = m / 2;

    // Parallelize the outer loop over x using OpenMP
    #pragma omp parallel for collapse(2) // Collapse 2 loops for better performance
    for (std::size_t x = 0; x < n; ++x) {
        for (std::size_t y = 0; y < n; ++y) {
            float result = 0.0;

            // Apply the mask to the current position (x, y)
            for (std::size_t i = 0; i < m; ++i) {
                for (std::size_t j = 0; j < m; ++j) {
                    // Calculate the corresponding coordinates in the original image
                    std::size_t image_x = x + i - offset;
                    std::size_t image_y = y + j - offset;

                    // Handle boundary conditions by padding with zeros or ones
                    float image_value;
                    if (image_x < n && image_y < n) {
                        image_value = image[image_x * n + image_y];
                    } else if ((image_x < n) || (image_y < n)) {
                        image_value = 1.0;
                    } else {
                        image_value = 0.0;
                    }

                    // Apply the mask value
                    result += mask[i * m + j] * image_value;
                }
            }

            // Store the result in the output
            output[x * n + y] = result;
        }
    }
}
