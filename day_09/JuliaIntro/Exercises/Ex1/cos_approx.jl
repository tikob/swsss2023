using BenchmarkTools

function cos_approx(x, N)
    return sum((-1)^n*x^(2n)/factorial(2n) for n in 1:N)
    # approximation of cosine via power series expansion
    # inputs:
    #       - x : argument of cosine 
    #       - N : truncation order of the power series approximation
    # outputs:
    #       - cos_val : approximation of cos(x)
end

@btime cos_approx($(π/3),$(10)) 
results = @btime cos($(π/3))
@btime cos(π/3)


println("The time it took $(results)" * string(results))S