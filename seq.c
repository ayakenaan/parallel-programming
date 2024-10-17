#include <stdio.h>
#include <time.h>  //For clock()
// Function to evaluate the curve (y = f(x))
double f(double x) {
    return x * x;  // Example: y = x^2
}

int main() {
    double a = 0.0, b = 1.0;  //Limits of integration
    int n;                    //Number of intervals
    double h, area, x;
    int computational_steps = 0;  //Counter for computational steps
    clock_t start, end;      

    printf("Enter the number of intervals (n): ");
    scanf("%d", &n);

    // Start timing
    start = clock();

    //Step size (width of each trapezoid)
    h = (b - a) / n;
    computational_steps++;  
    //Initial area (sum of the first and last terms)
    area = (f(a) + f(b)) / 2.0;
    computational_steps += 4; 
    for (int i = 1; i < n; i++) {
        x = a + i * h; 
        double fx = f(x);  
        area += fx; 
        computational_steps += 3;  
    }
    area *= h;
    computational_steps++;  
    
    end = clock();
    printf("The area under the curve is: %f\n", area);
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time taken: %f seconds\n", time_taken);
    printf("Total number of computational steps: %d\n", computational_steps);
    return 0;
}
