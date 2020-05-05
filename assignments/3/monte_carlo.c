#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Compute approximate integral of function on interval [a, b] using Monte-carlo integration.
 *
 * Parameters: 
 *   The parameters a and b represent the lower and upper integration bounds respectively.
 *   The parameter n_samp specifies the number of samples used in the integration.
 *   The parameter fun represents the function to be numerically integrated.
 *   The parameter fun_rand represents the function used to sample the uniform distribution.
 *   The parameter max_fun_value represents the maximum value of the function on the integration interval.
 * */
double monte_carlo(double a, double b, long n_samp, double(*fun)(double), double(*fun_rand)(double, double), double max_fun_val)
{

    // declare variables for sample x and y values.
    double x_samp, y_samp;

    // Declare and initialize variable for counting hits.
    long num_in = 0;
    
    // Perform sampling.
    for (int i = 0; i < n_samp; i++)
    {
        // Sample x and y coordinates.
        // Use Julia function as rand() is problematic.
        x_samp = fun_rand(a, b);
        y_samp = fun_rand(0, max_fun_val);
        if (y_samp <= fun(x_samp))
        {
            num_in++;
        }
    }
    
    // Get proportion of samples below curve.
    return (b-a)*max_fun_val*((double)num_in/(double)n_samp);
}

