##################################
#### assignment and unicode symbols 🔥🔥🔥
##################################
using LinearAlgebra
# silly example 

🥔 = 43
🍨 = 16

# Golden ratio (nice vs. old)
φ = (1+√5)/2
phi = (1+sqrt(5))/2
# pi 
π

# Navier stokes 
∇ = randn(10,10)
u = randn(10)
p = 0
ρ = 1000.

ρ*(u⋅(∇*u)) + p

##################################
#### Arrays/Vectors/Matrices
##################################

# defining a vector
powers_of_two = [1, 2, 4.0, "🍨"]
some_random_stuff = 

# appending stuff/mutating vectors: push!, append!

# defining a matrix
vandermonde = [1 2 4 8;  # first row
                 1 3 9 27] # second row

# concatenating 
# adding rows 
add_a_row = [vandermonde;
                1 4 16 47]

# adding columns

add_a_column = [vandermonde [1; 5]]

# indexing starts at 1!
add_a_column[1,2]

# slicing
add_a_column[:, 1:4]

# last element is indexed by end keyword

##################################
#### loops + printing
##################################
for i in 1:8:10
    println(i)
end


for power in powers_of_two
    println(power)
end

push!(powers_of_two, 7)
append!(powers_of_two, [5,"🍷"])
i = 0
while i < 10
    println(i)
    i +=1
end
#in particular ranges are written with : instead of range function
#range(5) in python <=> 0:4 in julia 

##################################
#### if-elseif-else 
##################################
a = 5.0

if a < 2.5
    println("a < 2.5")
elseif a < 3.5
    println("a < 3.5")
else
    println("a is not less than 3.5")
end
##################################
#### functions
##################################

# functional programming style
function my_add(a::Float64, b)
    c = a + b
    return [a,b]
    
end

function my_add(a, b)
    c = a + b
    return a
    
end

Σ = my_add(5, 3)

# for simple functions we may prefer the assignment form 
# to resemble standard math notation more closely
f(x::Float64) = 1/(2π)*exp(-1/2*x^2)

# evaluation 
p = f(2)

f.([1.0, 2.0, 0.5, 0.7])

# vectorization/(map-reduce)
# evaluates our function at every element of the supplied 
# vector/array and returns the result in the same shape!


# differences between python and julia
# Why Julia was created
# https://julialang.org/blog/2012/02/why-we-created-julia/# 
# Julia documentation: Noteworthy differences to other common languages
# https://docs.julialang.org/en/v1/manual/noteworthy-differences/
# Julia for data science
# https://www.imaginarycloud.com/blog/julia-vs-python/#future