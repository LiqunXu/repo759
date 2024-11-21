#include "stencil.cuh"
#include <cuda_runtime.h>
#include <cstdio>

// CUDA kernel for stencil computation
__global__ void stencil_kernel(const float* input_image, const float* filter_mask, float* result_output, unsigned int data_size, unsigned int filter_radius) {
    // Declare shared memory for temporary storage
    extern __shared__ float shared_memory[];

    // Divide shared memory into sections for image, mask, and output
    float* local_image = shared_memory; // Shared memory for the image segment
    float* local_mask = &shared_memory[blockDim.x + 2 * filter_radius]; // Shared memory for the mask
    float* local_output = &shared_memory[blockDim.x + 2 * filter_radius + (2 * filter_radius + 1)]; // Shared memory for output

    // Thread and global index
    unsigned int thread_index = threadIdx.x;
    unsigned int global_idx = blockIdx.x * blockDim.x + thread_index;

    // Load the mask into shared memory
    if (thread_index < 2 * filter_radius + 1) {
        local_mask[thread_index] = filter_mask[thread_index];
    }

    // Load the input image into shared memory with halo regions
    if (global_idx < data_size) {
        local_image[thread_index + filter_radius] = input_image[global_idx];
    } else {
        local_image[thread_index + filter_radius] = 1.0f; // Default value for out-of-bound indices
    }

    // Load halo elements on the left of the block
    if (thread_index < filter_radius) {
        unsigned int left_idx = global_idx < filter_radius ? 0 : global_idx - filter_radius;
        local_image[thread_index] = (global_idx < filter_radius) ? 1.0f : input_image[left_idx];
    }

    // Load halo elements on the right of the block
    if (thread_index >= blockDim.x - filter_radius) {
        unsigned int right_idx = global_idx + filter_radius >= data_size ? data_size - 1 : global_idx + filter_radius;
        local_image[thread_index + 2 * filter_radius] = (global_idx + filter_radius >= data_size) ? 1.0f : input_image[right_idx];
    }

    // Ensure all threads have completed memory operations
    __syncthreads();

    // Perform the stencil computation
    if (global_idx < data_size) {
        float temp_result = 0.0f;
        for (int offset = -filter_radius; offset <= filter_radius; ++offset) {
            temp_result += local_image[thread_index + filter_radius + offset] * local_mask[offset + filter_radius];
        }
        local_output[thread_index] = temp_result;
    }

    // Synchronize threads before writing back the results
    __syncthreads();
    
    // Write the result back to global memory
    if (global_idx < data_size) {
        result_output[global_idx] = local_output[thread_index];
    }
}

// Host function for stencil computation
void stencil(const float* input_image, const float* filter_mask, float* result_output, unsigned int data_size, unsigned int filter_radius, unsigned int threads_per_block) {
    float *device_image, *device_mask, *device_output;

    // Allocate memory on the device
    cudaMalloc((void**)&device_image, data_size * sizeof(float));
    cudaMalloc((void**)&device_mask, (2 * filter_radius + 1) * sizeof(float));
    cudaMalloc((void**)&device_output, data_size * sizeof(float));

    // Copy input data from host to device
    cudaMemcpy(device_image, input_image, data_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_mask, filter_mask, (2 * filter_radius + 1) * sizeof(float), cudaMemcpyHostToDevice);

    // Set up the grid and block dimensions
    dim3 block_dim(threads_per_block);
    dim3 grid_dim((data_size + threads_per_block - 1) / threads_per_block);

    // Calculate shared memory size
    size_t shared_mem_size = (threads_per_block + 2 * filter_radius) * sizeof(float) +  // For local_image
                             (2 * filter_radius + 1) * sizeof(float) +                 // For local_mask
                             threads_per_block * sizeof(float);                        // For local_output

    // Launch the kernel
    stencil_kernel<<<grid_dim, block_dim, shared_mem_size>>>(device_image, device_mask, device_output, data_size, filter_radius);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Copy the results back from device to host
    cudaMemcpy(result_output, device_output, data_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free allocated memory on the device
    cudaFree(device_image);
    cudaFree(device_mask);
    cudaFree(device_output);
}
