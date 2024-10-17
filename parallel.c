#include <mpi.h>
#include <stdio.h>
#include <math.h>

// Function to evaluate the curve (y = f(x))
float f(float x) {
    return x * x;  // Example: y = x^2
}

// Function to compute the area of a trapezoid and count function evaluations
float trapezoid_area(float a, float b, float d, int *eval_count) {
    float area = 0.0f;
    *eval_count = 0;  // Initialize the step counter

    for (float x = a; x < b; x += d) {
        area += f(x)+ f(x + d); 
         (*eval_count)++;
         (*eval_count)++;       // Increment the counter
   
    }
    return area * d / 2.0f;
}


int main(int argc, char** argv) {
    int rank, size;
    float a = 0.0f, b = 1.0f;  // Limits of integration
    int n;
    float local_area, total_area;
    int local_eval_count = 0;  // Local counter for function evaluations

    MPI_Init(&argc, &argv); // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get rank of the process
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get number of processes

    double time_start, time_end;  // Variables to store start and end time

    if (rank == 0) {
        // Get the number of intervals from the user
        printf("Enter the number of intervals: ");
        scanf("%d", &n);
    }
    
    // Broadcast the number of intervals to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Calculate the interval size for each process
    float d = (b - a) / n;  // delta
    float region = (b - a) / size;
    
    // Calculate local bounds for each process
    float start = a + rank * region; 
    float end = start + region;       
    // Measure the start time
       MPI_Barrier(MPI_COMM_WORLD);
    time_start = MPI_Wtime();
    
    // Each process calculates the area of its subinterval
    local_area = trapezoid_area(start, end, d, &local_eval_count);
   // Measure the end time
    time_end = MPI_Wtime();
    // Reduce all local areas and evaluation counts to the total area and count on the root process
       
    int total_eval_count = 0;  // Variable to hold the total evaluation count on the root
    MPI_Reduce(&local_area, &total_area, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD); 
    MPI_Reduce(&local_eval_count, &total_eval_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
     
  
    if (rank == 0) {
        printf("The total area under the curve is: %f\n", total_area);
        printf("Parallel execution time: %f seconds\n", time_end - time_start);
        printf("Total computational steps (function evaluations): %d\n", total_eval_count);
    }
    
    MPI_Finalize(); // Finalize MPI
    return 0;
}