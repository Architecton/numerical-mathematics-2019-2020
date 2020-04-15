function fun(x)::Cdouble
    return 1/sqrt(2*pi)*exp(-x^2/(2*pi))
end
fun_c = @cfunction(fun, Cdouble, (Cdouble,))

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

        # Evaluate integral using trapezoidal rule.
        return ccall((:trapz,"./trapz"), Cdouble, 
                     (Cdouble, Cdouble, Cint, Ptr{Cvoid}), a, b, n, fun_c)
    end

end

res = evaluate(0.0, 1.0, 1.0e-4, "trapz", fun_c)
