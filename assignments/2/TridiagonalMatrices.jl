module TridiagonalMatrices

import Base:*,\,convert,copy,size,getindex,setindex
import LinearAlgebra:dot,lu
using Printf
using InteractiveUtils  # Prevent subtypes not defined error.

export TridiagonalMatrix, UpperTridiagonalMatrix, LowerTridiagonalMatrix

"""
    TridiagonalMatrix{T} <: AbstractArray{T, 2}

Type representing a tridiagonal matrix where only the specified diagonals are explicitly stored.
The matrix is initialized by specifying the band diagonals as an array of arrays.

# Examples
```julia-repl
julia> M = TridiagonalMatrix{Int64}([[1, 4, 3], [4, 2, 3, 4], [4, 7, 6]])
4×4 TridiagonalMatrix{Int64}:
 4  4  0  0
 1  2  7  0
 0  4  3  6
 0  0  3  4
```
"""
struct TridiagonalMatrix{T} <: AbstractArray{T,2}

    # Explicitly stored values on the diagonals.
    diagonals::Array{Array{T,1},1}

    # Constructor
    function TridiagonalMatrix{T}(diagonals::Array{Array{T,1},1}) where {T}

        # Check if diagonals specified correctly.
        if length(diagonals) != 3
            error("diagonals not specified correctly")
        else
            if length(diagonals[1]) + 1 != length(diagonals[2]) || 
                length(diagonals[3]) + 1 != length(diagonals[2])
                error("diagonals not specified correctly")
            else
                new(diagonals)
            end
        end
    end
end


struct UpperTridiagonalMatrix{T} <: AbstractArray{T,2}

    # Explicitly stored values on the diagonals.
    diagonals::Array{Array{T,1},1}

    # Constructor
    function UpperTridiagonalMatrix{T}(diagonals::Array{Array{T,1},1}) where {T}

        # Check if diagonals specified correctly.
        if length(diagonals) != 2
            error("diagonals not specified correctly")
        else
            if length(diagonals[2]) + 1 != length(diagonals[1])
                error("diagonals not specified correctly")
            else
                new(diagonals)
            end
        end
    end
end


struct LowerTridiagonalMatrix{T} <: AbstractArray{T,2}

    # Explicitly stored values on the diagonals.
    diagonals::Array{Array{T,1},1}

    # Constructor
    function LowerTridiagonalMatrix{T}(diagonals::Array{Array{T,1},1}) where {T}

        # Check if diagonals specified correctly.
        if length(diagonals) != 2
            error("diagonals not specified correctly")
        else
            if length(diagonals[1]) + 1 != length(diagonals[2])
                error("diagonals not specified correctly")
            else
                new(diagonals)
            end
        end
    end
end


"""
    Base.size(M::TridiagonalMatrix)::Tuple{Int64,Int64}

Get size of matrix represented by tridiagonal matrix type.

# Examples
```julia-repl
julia> M = TridiagonalMatrix{Int64}([[1, 4, 3], [4, 2, 3, 4], [4, 7, 6]])
4×4 TridiagonalMatrix{Int64}:
 4  4  0  0
 1  2  7  0
 0  4  3  6
 0  0  3  4

julia> size(M)
(4, 4)
```
"""
function Base.size(M::TridiagonalMatrix)::Tuple{Int64,Int64}
    return (length(M.diagonals[2]), length(M.diagonals[2]))
end

function Base.size(M::UpperTridiagonalMatrix)::Tuple{Int64,Int64}
    return (length(M.diagonals[1]), length(M.diagonals[1]))
end

function Base.size(M::LowerTridiagonalMatrix)::Tuple{Int64,Int64}
    return (length(M.diagonals[2]), length(M.diagonals[2]))
end


"""
    convert(::Type{TridiagonalMatrix{T}}, M::TridiagonalMatrix) where {T}

Convert type of values of matrix represented by tridiagonal matrix type.
"""
function convert(::Type{TridiagonalMatrix{T}}, M::TridiagonalMatrix) where {T}
    res = TridiagonalMatrix{T}(convert(Array{Array{T, 1},1}, M.diagonals))
    return res
end

function convert(::Type{UpperTridiagonalMatrix{T}}, M::UpperTridiagonalMatrix) where {T}
    res = UpperTridiagonalMatrix{T}(convert(Array{Array{T, 1},1}, M.diagonals))
    return res
end

function convert(::Type{LowerTridiagonalMatrix{T}}, M::LowerTridiagonalMatrix) where {T}
    res = LowerTridiagonalMatrix{T}(convert(Array{Array{T, 1},1}, M.diagonals))
    return res
end


"""
    Base.copy(M::Union{TridiagonalMatrix, UpperTridiagonalMatrix, LowerTridiagonalMatrix})

Make a copy of a tridiagonal matrix type instance.
"""
function Base.copy(M::Union{TridiagonalMatrix, UpperTridiagonalMatrix, LowerTridiagonalMatrix})
    return typeof(M)(deepcopy(M.diagonals))
end


"""
    Base.getindex(M::TridiagonalMatrix, idx::Int)

Perform linear indexing of matrix represented by tridiagonal matrix type instance.

# Examples
```julia-repl
julia> M = TridiagonalMatrix{Int64}([[1, 4, 3], [4, 2, 3, 4], [4, 7, 6]])
4×4 TridiagonalMatrix{Int64}:
 4  4  0  0
 1  2  7  0
 0  4  3  6
 0  0  3  4

julia> M[6]
2

julia> M[3]
0
```
"""
function Base.getindex(M::TridiagonalMatrix, idx::Int)
    
    # If linear index out of bounds, throw error.
    if idx > length(M.diagonals[2])^2
        throw(BoundsError())
    end

    # Convert to Cartesian index.
    cart_idx = CartesianIndices(size(M))[idx]
    
    # If difference between values in cartesian indices greater than 1 return 0.
    if abs(cart_idx[2] - cart_idx[1]) > 1
        return 0
    else
        # Compute difference between values in cartesian indices.
        cart_idx_diff = cart_idx[2] - cart_idx[1]

        # Return appropriate element on appropriate diagonal.
        return cart_idx_diff < 0 ? M.diagonals[1][cart_idx[2]] : M.diagonals[2 + cart_idx_diff][cart_idx[1]]
    end
end


function Base.getindex(M::UpperTridiagonalMatrix, idx::Int)
    
    # If linear index out of bounds, throw error.
    if idx > length(M.diagonals[1])^2
        throw(BoundsError())
    end

    # Convert to Cartesian index.
    cart_idx = CartesianIndices(size(M))[idx]
    
    # If difference between values in cartesian indices greater than 1 
    # or if element below main diagonal, return 0.
    if abs(cart_idx[2] - cart_idx[1]) > 1 || cart_idx[2] - cart_idx[1] < 0
        return 0
    else
        # Compute difference between values in cartesian indices.
        cart_idx_diff = cart_idx[2] - cart_idx[1]

        # Return appropriate element on appropriate diagonal.
        return M.diagonals[1 + cart_idx_diff][cart_idx[1]]
    end
end


function Base.getindex(M::LowerTridiagonalMatrix, idx::Int)
    
    # If linear index out of bounds, throw error.
    if idx > length(M.diagonals[2])^2
        throw(BoundsError())
    end

    # Convert to Cartesian index.
    cart_idx = CartesianIndices(size(M))[idx]
    
    # If difference between values in cartesian indices greater than 1
    # or if element above main diagonal, return 0.
    if abs(cart_idx[2] - cart_idx[1]) > 1 || cart_idx[2] - cart_idx[1] > 0
        return 0
    else
        # Compute difference between values in cartesian indices.
        cart_idx_diff = cart_idx[2] - cart_idx[1]

        # Return appropriate element on appropriate diagonal.
        return M.diagonals[2+cart_idx_diff][cart_idx[2]]
    end
end


"""
    Base.getindex(M::TridiagonalMatrix, idx::Vararg{Int, 2})

Perform cartesian indexing of matrix represented by tridiagonal matrix type instance.

# Examples
```julia-repl
julia> M = TridiagonalMatrix{Int64}([[1, 4, 3], [4, 2, 3, 4], [4, 7, 6]])
4×4 TridiagonalMatrix{Int64}:
 4  4  0  0
 1  2  7  0
 0  4  3  6
 0  0  3  4

julia> M[2, 2]
2

julia> M[3, 1]
0
```
"""
function Base.getindex(M::TridiagonalMatrix, idx::Vararg{Int, 2})
    
    # Build cartesian index.
    cart_idx = CartesianIndex(idx[1], idx[2])
    
    # If cartesian index out of bounds, throw error.
    if cart_idx[1] > length(M.diagonals[2]) || cart_idx[2] > length(M.diagonals[2])
        throw(BoundsError())
    end
    
    # If difference between values in cartesian indices greater than 1, return 0.
    if abs(cart_idx[2] - cart_idx[1]) > 1
        return 0
    else
        
        # Compute difference between values in cartesian indices.
        cart_idx_diff = cart_idx[2] - cart_idx[1]
        
        # Return appropriate element on appropriate diagonal.
        return cart_idx_diff < 0 ? M.diagonals[1][cart_idx[2]] : M.diagonals[2 + cart_idx_diff][cart_idx[1]]
    end
end


function Base.getindex(M::UpperTridiagonalMatrix, idx::Vararg{Int, 2})
    
    # Build cartesian index.
    cart_idx = CartesianIndex(idx[1], idx[2])
    
    # If cartesian index out of bounds, throw error.
    if cart_idx[1] > length(M.diagonals[1]) || cart_idx[2] > length(M.diagonals[1])
        throw(BoundsError())
    end
    
    # If difference between values in cartesian indices greater than 1
    # or if element below main diagonal, return 0.
    if abs(cart_idx[2] - cart_idx[1]) > 1 || cart_idx[2] - cart_idx[1] < 0
        return 0
    else
        # Compute difference between values in cartesian indices.
        cart_idx_diff = cart_idx[2] - cart_idx[1]

        # Return appropriate element on appropriate diagonal.
        return M.diagonals[1 + cart_idx_diff][cart_idx[1]]
    end
end


function Base.getindex(M::LowerTridiagonalMatrix, idx::Vararg{Int, 2})
    
    # Build cartesian index.
    cart_idx = CartesianIndex(idx[1], idx[2])
    
    # If cartesian index out of bounds, throw error.
    if cart_idx[1] > length(M.diagonals[2]) || cart_idx[2] > length(M.diagonals[2])
        throw(BoundsError())
    end
    
    # If difference between values in cartesian indices greater than 1 
    # or if element above main diagonal, return 0.
    if abs(cart_idx[2] - cart_idx[1]) > 1 || cart_idx[2] - cart_idx[1] > 0
        return 0
    else
        # Compute difference between values in cartesian indices.
        cart_idx_diff = cart_idx[2] - cart_idx[1]

        # Return appropriate element on appropriate diagonal.
        return M.diagonals[2+cart_idx_diff][cart_idx[2]]
    end
    
end


"""
    Base.setindex!(M::TridiagonalMatrix, val, idx::Int)

Set element at specified linear index in tridiagonal matrix type instance.

# Examples
```julia-repl
julia> M = TridiagonalMatrix{Int64}([[1, 4, 3], [4, 2, 3, 4], [4, 7, 6]])
4×4 TridiagonalMatrix{Int64}:
 4  4  0  0
 1  2  7  0
 0  4  3  6
 0  0  3  4

julia> M[7] = 11
11

julia> M
4×4 TridiagonalMatrix{Int64}:
 4   4  0  0
 1   2  7  0
 0  11  3  6
 0   0  3  4
```
"""
function Base.setindex!(M::TridiagonalMatrix, val, idx::Int)

    # If linear index out of bounds, throw error.
    if idx > length(M.diagonals[2])^2
        throw(BoundsError())
    end

    # Convert to Cartesian index.
    cart_idx = CartesianIndices(size(M))[idx]
    
    # If difference between values in cartesian indices greater than 1, throw error.
    if abs(cart_idx[2] - cart_idx[1]) > 1
        throw(DomainError(idx, "only elements on diagonals can be set"))
    else
        
        # Compute difference between values in cartesian indices.
        cart_idx_diff = cart_idx[2] - cart_idx[1]

        # Set appropriate value on appropriate diagonal.
        if cart_idx_diff < 0
            M.diagonals[1][cart_idx[2]] = val
        else
            M.diagonals[2 + cart_idx_diff][cart_idx[1]] = val
        end
    end
end


"""
    Base.setindex!(M::TridiagonalMatrix, val, idx::Int)

Set element at specified linear index in tridiagonal matrix type instance.

# Examples
```julia-repl
```
"""
function Base.setindex!(M::UpperTridiagonalMatrix, val, idx::Int)

    # If linear index out of bounds, throw error.
    if idx > length(M.diagonals[1])^2
        throw(BoundsError())
    end

    # Convert to Cartesian index.
    cart_idx = CartesianIndices(size(M))[idx]
    
    # If difference between values in cartesian indices greater than 1
    # or if index below main diagonal, throw error.
    if abs(cart_idx[2] - cart_idx[1]) > 1 || cart_idx[2] - cart_idx[1] < 0
        throw(DomainError(idx, "only elements on diagonals can be set"))
    else
        
        # Compute difference between values in cartesian indices.
        cart_idx_diff = cart_idx[2] - cart_idx[1]

        # Set appropriate value on appropriate diagonal.
        M.diagonals[1+cart_idx_diff][cart_idx[1]] = val
    end
end


"""
    Base.setindex!(M::TridiagonalMatrix, val, idx::Int)

Set element at specified linear index in tridiagonal matrix type instance.

# Examples
```julia-repl
```
"""
function Base.setindex!(M::LowerTridiagonalMatrix, val, idx::Int)

    # If linear index out of bounds, throw error.
    if idx > length(M.diagonals[2])^2
        throw(BoundsError())
    end

    # Convert to Cartesian index.
    cart_idx = CartesianIndices(size(M))[idx]
    
    # If difference between values in cartesian indices greater than 1
    # or if index above main diagonal, throw error.
    if abs(cart_idx[2] - cart_idx[1]) > 1 || cart_idx[2] - cart_idx[1] > 0
        throw(DomainError(idx, "only elements on diagonals can be set"))
    else
        
        # Compute difference between values in cartesian indices.
        cart_idx_diff = cart_idx[2] - cart_idx[1]

        # Set appropriate value on appropriate diagonal.
        M.diagonals[2 + cart_idx_diff][cart_idx[2]] = val
    end
end

"""
    Base.setindex!(M::TridiagonalMatrix, val, idx::Vararg{Int, 2})

Set element at specified cartesian index in tridiagonal matrix type instance.

# Examples
```julia-repl
```
"""
function Base.setindex!(M::TridiagonalMatrix, val, idx::Vararg{Int, 2})
     
    # Build cartesian index.
    cart_idx = CartesianIndex(idx[1], idx[2])

    # If cartesian index out of bounds, throw error.
    if cart_idx[1] > length(M.diagonals[2]) || cart_idx[2] > length(M.diagonals[2])
        throw(BoundsError())
    end

    # If difference between values in cartesian indices greater than 1, throw error.
    if abs(cart_idx[2] - cart_idx[1]) > 1
        throw(DomainError(idx, "only elements on diagonals can be set"))
    else
        
        # Compute difference between values in cartesian indices.
        cart_idx_diff = cart_idx[2] - cart_idx[1]
        
        # Set appropriate value on appropriate diagonal.
        cart_idx_diff < 0 ? M.diagonals[1][cart_idx[2]] : M.diagonals[2 + cart_idx_diff][cart_idx[1]]
        if cart_idx_diff < 0
            M.diagonals[1][cart_idx[2]] = val
        else
            M.diagonals[2 + cart_idx_diff][cart_idx[1]] = val
        end
    end
end

"""
    Base.setindex!(M::TridiagonalMatrix, val, idx::Vararg{Int, 2})

Set element at specified cartesian index in tridiagonal matrix type instance.

# Examples
```julia-repl
```
"""
function Base.setindex!(M::UpperTridiagonalMatrix, val, idx::Vararg{Int, 2})
     
    # Build cartesian index.
    cart_idx = CartesianIndex(idx[1], idx[2])

    # If cartesian index out of bounds, throw error.
    if cart_idx[1] > length(M.diagonals[1]) || cart_idx[2] > length(M.diagonals[1])
        throw(BoundsError())
    end

    # If difference between values in cartesian indices greater than 1
    # or if index below main diagonal, throw error.
    if abs(cart_idx[2] - cart_idx[1]) > 1 || cart_idx[2] - cart_idx[1] < 0
        throw(DomainError(idx, "only elements on diagonals can be set"))
    else
        
        # Compute difference between values in cartesian indices.
        cart_idx_diff = cart_idx[2] - cart_idx[1]

        # Set appropriate value on appropriate diagonal.
        M.diagonals[1+cart_idx_diff][cart_idx[1]] = val
    end

end

"""
    Base.setindex!(M::TridiagonalMatrix, val, idx::Vararg{Int, 2})

Set element at specified cartesian index in tridiagonal matrix type instance.

# Examples
```julia-repl
```
"""
function Base.setindex!(M::LowerTridiagonalMatrix, val, idx::Vararg{Int, 2})
     
    # Build cartesian index.
    cart_idx = CartesianIndex(idx[1], idx[2])

    # If cartesian index out of bounds, throw error.
    if cart_idx[1] > length(M.diagonals[2]) || cart_idx[2] > length(M.diagonals[2])
        throw(BoundsError())
    end
    
    # If difference between values in cartesian indices greater than 1
    # or if index above main diagonal, throw error.
    if abs(cart_idx[2] - cart_idx[1]) > 1 || cart_idx[2] - cart_idx[1] > 0
        throw(DomainError(idx, "only elements on diagonals can be set"))
    else
        
        # Compute difference between values in cartesian indices.
        cart_idx_diff = cart_idx[2] - cart_idx[1]

        # Set appropriate value on appropriate diagonal.
        M.diagonals[2 + cart_idx_diff][cart_idx[2]] = val
    end

end


"""
    *(M::TridiagonalMatrix, v::Vector)
    
Perform multiplication of tridiagonal matrix type instance with vector.

# Examples
```julia-repl

```
"""
function *(M::TridiagonalMatrix, v::Vector)
    
    # Dimensions match test.
    if length(v) != length(M.diagonals[2])
        throw(DimensionMismatch(@sprintf("Matrix A has dimensions %s, vector B has length %d", size(M), length(v))))
    end

    # Allocate vector for storing results.
    type_M_param = typeof(M).parameters[1]
    res = Vector{type_M_param in subtypes(Signed) ? Float64 : type_M_param}(undef, length(v))
    
    # Set start and end indices for computing the result elements only using explicitly stored elements.
    end_idx = 2
    start_idx = 0

    # Compute product by shifting indices over explicitly stored elements.
    for idx = 1:length(v)
        res[idx] = dot(M[idx, max(start_idx, 1):min(end_idx, length(M.diagonals[2]))], v[max(start_idx, 1):min(end_idx, length(M.diagonals[2]))])
        start_idx += 1
        end_idx += 1
    end

    # Return result.
    return res
end


function *(M::UpperTridiagonalMatrix, v::Vector)
    
    # Dimensions match test.
    if length(v) != length(M.diagonals[1])
        throw(DimensionMismatch(@sprintf("Matrix A has dimensions %s, vector B has length %d", size(M), length(v))))
    end

    # Allocate vector for storing results.
    type_M_param = typeof(M).parameters[1]
    res = Vector{type_M_param in subtypes(Signed) ? Float64 : type_M_param}(undef, length(v))
    
    # Set start and end indices for computing the result elements only using explicitly stored elements.
    end_idx = 2
    start_idx = 1

    # Compute product by shifting indices over explicitly stored elements.
    for idx = 1:length(v)
        res[idx] = dot(M[idx, start_idx:min(end_idx, length(M.diagonals[1]))], v[start_idx:min(end_idx, length(M.diagonals[1]))])
        start_idx += 1
        end_idx += 1
    end

    # Return result.
    return res
end


function *(M::LowerTridiagonalMatrix, v::Vector)
    
    # Dimensions match test.
    if length(v) != length(M.diagonals[2])
        throw(DimensionMismatch(@sprintf("Matrix A has dimensions %s, vector B has length %d", size(M), length(v))))
    end

    # Allocate vector for storing results.
    type_M_param = typeof(M).parameters[1]
    res = Vector{type_M_param in subtypes(Signed) ? Float64 : type_M_param}(undef, length(v))
    
    # Set start and end indices for computing the result elements only using explicitly stored elements.
    end_idx = 1
    start_idx = 0

    # Compute product by shifting indices over explicitly stored elements.
    for idx = 1:length(v)
        res[idx] = dot(M[idx, max(start_idx, 1):min(end_idx, length(M.diagonals[2]))], v[max(start_idx, 1):min(end_idx, length(M.diagonals[2]))])
        start_idx += 1
        end_idx += 1
    end

    # Return result.
    return res
end


function lu(M::TridiagonalMatrix)

end

end
