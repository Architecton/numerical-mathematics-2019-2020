using Distributions

"""
    fun(x)::Cdouble

Function for evaluating numerical integration.
This function is passed to numerical integration
implementations as the function being integrated.
Connvert it to a C-callable function using the
@cfunction macro.

# Examples
```julia-repl
julia> fun(1.0)
0.3402435901430237
"""
function fun(x)::Cdouble
    return 1/sqrt(2*pi)*exp(-x^2/(2*pi))
end

# Convert to a C-callable function.
fun_c = @cfunction(fun, Cdouble, (Cdouble,))


"""
    evaluate(a::Float64, b::Float64, prec::Float64, method::String, func_c::Ptr{Nothing})

Numerically integrate specified function on interval [a, b] to specified precision. The
method parameter specifies the numerical integration method to use. The passed function
represents the function to integrate and should be represented as a C-callable function 
pointer.
"""
function evaluate(a::Float64, b::Float64, prec::Float64, method::String, func_c::Ptr{Nothing})
    
    # Evaluate using specified method.
    if method == "trapz"
        # If evaluating trapezoidal rule.

        # Second derivative of Gaussian (with mean 0 and std. dev. 1).
        function ddfun(x)::Cdouble
            return (exp(-x^2/82*pi)*(x^2-pi))/(sqrt(2)*pi^(5/2))
        end
        
        # Set maximum value of second derivative on integration interval.
        if a < -sqrt(3*pi) && b > -sqrt(3*pi) && b < 0
            max_ddfun_val = abs(ddfun(-sqrt(3*pi)))
        elseif a < 0 && b > 0
            max_ddfun_val = abs(ddfun(0))
        elseif a > 0 && a < sqrt(3*pi) && b > sqrt(3*pi)
            max_ddfun_val = abs(ddfun(sqrt(3*pi)))
        else
            max_ddfun_val = max(abs(fun(a)), abs(fun(b)))
        end
        
        # Compute required value of n to obtain desired precision.
        n = Cint(ceil(sqrt(max_ddfun_val*((b-a)^3)/(12*prec))))

        # Evaluate integral using trapezoidal rule (use implementation in C).
        return ccall((:trapz,"./trapz"), Cdouble, 
                     (Cdouble, Cdouble, Cint, Ptr{Cvoid}), a, b, n, fun_c)
    elseif method == "simpsons"

    elseif method == "romberg"

    elseif method == "monte-carlo"
        
        # Number of samples to take. See estimation of error.
        n_samp = 100000
        
        # Initialize function for sampling from uniform distribution. Pass it to C implementation.
        # NOTE: the rand() function in stdlib.h is problematic.
        fun_rand = @cfunction((a, b) -> rand(Uniform(a, b)), Cdouble, (Cdouble, Cdouble))
        return ccall((:monte_carlo,"./monte_carlo"), Cdouble, 
                     (Cdouble, Cdouble, Clong, Ptr{Cvoid}, Ptr{Cvoid}, Cdouble), a, b, n_samp, fun_c, fun_rand, 1.0)

    end

end

res1 = evaluate(0.0, 1.0, 1.0e-4, "monte-carlo", fun_c)
res2 = evaluate(0.0, 1.0, 1.0e-4, "trapz", fun_c)



