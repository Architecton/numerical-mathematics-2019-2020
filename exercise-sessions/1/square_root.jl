struct TaylorSeries end
struct NewtonMethod end

"Compute the square root of the given number using the default method"
koren(x) = koren(x, NewtonMethod())

"Compute the square root of the given number using Newton's method"
function koren(x, method::NewtonMethod)
    y = 1 + (x-1)/2 # Initial estimate
    maxit = 100
    prec = 1e-10
    for i = 1:maxit
        y = (y + x/y)/2
        if abs(x-y^2) < prec
            return y, i
        end 
    end
    return y, maxit
end


"Compute the square root of the given number using the Taylor series"
function koren(x, method::TaylorSeries)
    # Set required precision
    prec = 1e-2
    y = 1 + (x-1)/2

    maxit = 1000
    power = x-1
    term = 0.5

    for n = 2:maxit
        term = -term*(2*n-3)/2
        power = power*(x-1)
        corr = term*power
        y = y + corr
        err = corr*y^(2*n-1)
        if abs(err) < prec
            return y, n
        end
    end
    return y, maxit
end

