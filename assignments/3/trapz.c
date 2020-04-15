#include <math.h>

/* Compute approximate integral of function on interval [a, b] using n steps.
 *
 * Parameters: 
 *   The parameters a and b represent the lower and upper integration bounds respectively.
 *   The parameter n specifies the number of steps used in the approximation.
 *   The last parameter represents the function to be numerically integrated.
 * */
double trapz(double a, double b, int n, double(*fun)(double))
{
    // Set step size.
    double step = (b - a)/(double)n;

    // Initialize result accumulator and accumulate.
    double res_acc = (fun(a) + fun(b))/2.0;
    for (int i = 1; i <= n-1; i++)
    {
        res_acc += fun(a + i*step);
    }

    // Multiply with step size to get result.
    return res_acc*step;
}

