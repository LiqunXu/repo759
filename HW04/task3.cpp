// #include <iostream>
// #include <cmath>
// #include <fstream>
// #include <vector>
// #include <omp.h>  // Include OpenMP library

// // Constants
// const double G = 1.0;          // Gravitational constant
// const double softening = 0.1;  // Softening length
// const double dt = 0.01;        // Time step
// const double board_size = 4.0; // Size of the board

// void getAcc(const double pos[][3], const double mass[], double acc[][3], int N, int num_threads) {
//     // Initialize acceleration array to zero
//     #pragma omp parallel for num_threads(num_threads)  // Parallelize the outer loop
//     for (int i = 0; i < N; i++) {
//         acc[i][0] = acc[i][1] = acc[i][2] = 0.0;
//         for (int j = 0; j < N; j++) {
//             if (i != j) {
//                 double dx = pos[j][0] - pos[i][0];
//                 double dy = pos[j][1] - pos[i][1];
//                 double dz = pos[j][2] - pos[i][2];
//                 double distSqr = dx * dx + dy * dy + dz * dz + softening * softening;
//                 double invDist3 = pow(distSqr, -1.5);

//                 acc[i][0] += G * dx * invDist3 * mass[j];
//                 acc[i][1] += G * dy * invDist3 * mass[j];
//                 acc[i][2] += G * dz * invDist3 * mass[j];
//             }
//         }
//     }
// }

// void savePositionsToCSV(const double pos[][3], int N, int step, const std::string& filename) {
//     std::ofstream file;
//     if (step == 0) {
//         file.open(filename, std::ofstream::out | std::ofstream::trunc);
//         file << "step,positions\n";
//     } else {
//         file.open(filename, std::ofstream::out | std::ofstream::app);
//     }
//     file << step << ",[";
//     for (int i = 0; i < N; i++) {
//         file << "[" << pos[i][0] << "," << pos[i][1] << "," << pos[i][2] << "]";
//         if (i < N - 1) file << ",";
//     }
//     file << "]\n";
//     file.close();
// }

// int main(int argc, char *argv[]) {
//     if (argc != 5) {
//         std::cerr << "Usage: " << argv[0] << " number_of_particles simulation_end_time num_threads" << std::endl;
//         return 1;
//     }
    
//     int N = std::stoi(argv[1]);
//     double tEnd = std::stod(argv[2]);
//     int num_threads = std::stoi(argv[3]);
//     std::string filename = "positions.csv";

//     // Allocate memory for positions, velocities, masses, and accelerations
//     double pos[N][3], vel[N][3], acc[N][3], mass[N];

//     // Initialize random positions, velocities, and masses
//     for (int i = 0; i < N; i++) {
//         pos[i][0] = (double)rand() / RAND_MAX * board_size - board_size / 2;
//         pos[i][1] = (double)rand() / RAND_MAX * board_size - board_size / 2;
//         pos[i][2] = (double)rand() / RAND_MAX * board_size - board_size / 2;

//         vel[i][0] = (double)rand() / RAND_MAX - 0.5;
//         vel[i][1] = (double)rand() / RAND_MAX - 0.5;
//         vel[i][2] = (double)rand() / RAND_MAX - 0.5;

//         mass[i] = (double)rand() / RAND_MAX + 0.1;
//     }

//     // Calculate initial accelerations
//     getAcc(pos, mass, acc, N, num_threads);

//     // Main simulation loop
//     int Nt = int(tEnd / dt);
//     double t = 0.0;
//     for (int step = 0; step < Nt; step++) {
        
//         // (1/2) kick: Update velocities by half-step
//         #pragma omp parallel for num_threads(num_threads)
//         for (int i = 0; i < N; i++) {
//             vel[i][0] += 0.5 * acc[i][0] * dt;
//             vel[i][1] += 0.5 * acc[i][1] * dt;
//             vel[i][2] += 0.5 * acc[i][2] * dt;
//         }

//         // Drift: Update positions with the new velocities
//         #pragma omp parallel for num_threads(num_threads)
//         for (int i = 0; i < N; i++) {
//             pos[i][0] += vel[i][0] * dt;
//             pos[i][1] += vel[i][1] * dt;
//             pos[i][2] += vel[i][2] * dt;
//         }

//         // Ensure particles stay within the board limits
//         #pragma omp parallel for num_threads(num_threads)
//         for (int i = 0; i < N; i++) {
//             for (int j = 0; j < 3; j++) {
//                 if (pos[i][j] > board_size) pos[i][j] = board_size;
//                 if (pos[i][j] < -board_size) pos[i][j] = -board_size;
//             }
//         }

//         // Update accelerations
//         getAcc(pos, mass, acc, N, num_threads);

//         // (1/2) kick: Update velocities by another half-step
//         #pragma omp parallel for num_threads(num_threads)
//         for (int i = 0; i < N; i++) {
//             vel[i][0] += 0.5 * acc[i][0] * dt;
//             vel[i][1] += 0.5 * acc[i][1] * dt;
//             vel[i][2] += 0.5 * acc[i][2] * dt;
//         }

//         // Update time
//         t += dt;

//         // For debug: save positions to CSV at each step
//         savePositionsToCSV(pos, N, step, filename);
//     }
//     return 0;
// }


#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include <omp.h>  // Include OpenMP library
#include <chrono> // For timing

const double G = 1.0;       // Gravitational constant
const double dt = 0.01;     // Time step
const double softening = 0.1;  // Softening parameter
const double board_size = 4.0; // Boundary size

void getAcc(const double pos[][3], const double mass[], double acc[][3], int N, int num_threads, std::string schedule_policy) {
    // Set schedule type
    omp_sched_t sched_type = (schedule_policy == "static") ? omp_sched_static : 
                             (schedule_policy == "dynamic") ? omp_sched_dynamic : omp_sched_guided;

    omp_set_schedule(sched_type, 1);  // Set schedule type with chunk size 1

    // Initialize acceleration array to zero
    #pragma omp parallel for num_threads(num_threads) schedule(runtime) // Enable runtime scheduling
    for (int i = 0; i < N; i++) {
        acc[i][0] = acc[i][1] = acc[i][2] = 0.0;
        for (int j = 0; j < N; j++) {
            if (i != j) {
                double dx = pos[j][0] - pos[i][0];
                double dy = pos[j][1] - pos[i][1];
                double dz = pos[j][2] - pos[i][2];
                double distSqr = dx * dx + dy * dy + dz * dz + softening * softening;
                double invDist3 = pow(distSqr, -1.5);

                acc[i][0] += G * dx * invDist3 * mass[j];
                acc[i][1] += G * dy * invDist3 * mass[j];
                acc[i][2] += G * dz * invDist3 * mass[j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " number_of_particles simulation_end_time num_threads schedule_policy" << std::endl;
        return 1;
    }
    
    int N = std::stoi(argv[1]);
    double tEnd = std::stod(argv[2]);
    int num_threads = std::stoi(argv[3]);
    std::string schedule_policy = argv[4];
    std::string filename = "positions.csv";

    // Allocate memory for positions, velocities, masses, and accelerations
    double pos[N][3], vel[N][3], acc[N][3], mass[N];

    // Initialize random positions, velocities, and masses
    for (int i = 0; i < N; i++) {
        pos[i][0] = (double)rand() / RAND_MAX * board_size - board_size / 2;
        pos[i][1] = (double)rand() / RAND_MAX * board_size - board_size / 2;
        pos[i][2] = (double)rand() / RAND_MAX * board_size - board_size / 2;

        vel[i][0] = (double)rand() / RAND_MAX - 0.5;
        vel[i][1] = (double)rand() / RAND_MAX - 0.5;
        vel[i][2] = (double)rand() / RAND_MAX - 0.5;

        mass[i] = (double)rand() / RAND_MAX + 0.1;
    }

    // Calculate initial accelerations
    getAcc(pos, mass, acc, N, num_threads, schedule_policy);

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Main simulation loop
    int Nt = int(tEnd / dt);
    double t = 0.0;
    for (int step = 0; step < Nt; step++) {
        
        // (1/2) kick: Update velocities by half-step
        #pragma omp parallel for num_threads(num_threads) schedule(runtime)
        for (int i = 0; i < N; i++) {
            vel[i][0] += 0.5 * acc[i][0] * dt;
            vel[i][1] += 0.5 * acc[i][1] * dt;
            vel[i][2] += 0.5 * acc[i][2] * dt;
        }

        // Drift: Update positions with the new velocities
        #pragma omp parallel for num_threads(num_threads) schedule(runtime)
        for (int i = 0; i < N; i++) {
            pos[i][0] += vel[i][0] * dt;
            pos[i][1] += vel[i][1] * dt;
            pos[i][2] += vel[i][2] * dt;
        }

        // Ensure particles stay within the board limits
        #pragma omp parallel for num_threads(num_threads) schedule(runtime)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < 3; j++) {
                if (pos[i][j] > board_size) pos[i][j] = board_size;
                if (pos[i][j] < -board_size) pos[i][j] = -board_size;
            }
        }

        // Update accelerations
        getAcc(pos, mass, acc, N, num_threads, schedule_policy);

        // (1/2) kick: Update velocities by another half-step
        #pragma omp parallel for num_threads(num_threads) schedule(runtime)
        for (int i = 0; i < N; i++) {
            vel[i][0] += 0.5 * acc[i][0] * dt;
            vel[i][1] += 0.5 * acc[i][1] * dt;
            vel[i][2] += 0.5 * acc[i][2] * dt;
        }

        // Update time
        t += dt;
    }

    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}
