module TridiagonalMatrices

import Base:*,\,convert,copy,size,getindex,setindex!
import LinearAlgebra:dot,lu,I,norm,diagind
using Printf

export TridiagonalMatrix, UpperTridiagonalMatrix, LowerTridiagonalMatrix, tridiag, inv_eigen


"""
    TridiagonalMatrix{T} <: AbstractArray{T, 2}

Type representing a tridiagonal matrix where only the specified diagonals are explicitly stored.
The matrix is initialized by specifying the band diagonals as an array of arrays.

# Examples
```julia-repl
julia> td = TridiagonalMatrix{Int64}([[1, 4, 3], [4, 2, 3, 4], [4, 7, 6]])
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


"""
    UpperTridiagonalMatrix{T} <: AbstractArray{T,2}

Type representing an upper tridiagonal matrix where only the specified diagonals are explicitly stored.
The matrix is initialized by specifying the band diagonals as an array of arrays.

# Examples
```julia-repl
julia> utd = UpperTridiagonalMatrix{Int64}([[4, 2, 3, 4], [4, 7, 6]])
4×4 UpperTridiagonalMatrix{Int64}:
 4  4  0  0
 0  2  7  0
 0  0  3  6
 0  0  0  4
```
"""
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


"""
    LowerTridiagonalMatrix{T} <: AbstractArray{T,2}

Type representing a lower tridiagonal matrix where only the specified diagonals are explicitly stored.
The matrix is initialized by specifying the band diagonals as an array of arrays.

# Examples
```julia-repl
julia> ltd = LowerTridiagonalMatrix{Int64}([[4, 2, 3], [4, 7, 6, 1]])
4×4 LowerTridiagonalMatrix{Int64}:
 4  0  0  0
 4  7  0  0
 0  2  6  0
 0  0  3  1
```
"""
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
    size(M::TridiagonalMatrix)::Tuple{Int64,Int64}

Get size of matrix represented by tridiagonal matrix type.
"""
function size(M::TridiagonalMatrix)::Tuple{Int64,Int64}
    return (length(M.diagonals[2]), length(M.diagonals[2]))
end


"""
    size(M::UpperTridiagonalMatrix)::Tuple{Int64,Int64}

Get size of matrix represented by upper tridiagonal matrix type.
"""
function size(M::UpperTridiagonalMatrix)::Tuple{Int64,Int64}
    return (length(M.diagonals[1]), length(M.diagonals[1]))
end


"""
    size(M::LowerTridiagonalMatrix)::Tuple{Int64,Int64}

Get size of matrix represented by lower tridiagonal matrix type.
"""
function size(M::LowerTridiagonalMatrix)::Tuple{Int64,Int64}
    return (length(M.diagonals[2]), length(M.diagonals[2]))
end


"""
    convert(::Type{TridiagonalMatrix{T}}, M::TridiagonalMatrix) where {T}

Convert type of values of matrix represented by tridiagonal matrix type.
"""
function convert(::Type{TridiagonalMatrix{T}}, M::TridiagonalMatrix) where {T}
    res = TridiagonalMatrix{T}(convert(Array{Array{T, 1},1}, deepcopy(M.diagonals)))
    return res
end


"""
    convert(::Type{UpperTridiagonalMatrix{T}}, M::UpperTridiagonalMatrix) where {T}

Convert type of values of matrix represented by upper tridiagonal matrix type.
"""
function convert(::Type{UpperTridiagonalMatrix{T}}, M::UpperTridiagonalMatrix) where {T}
    res = UpperTridiagonalMatrix{T}(convert(Array{Array{T, 1},1}, deepcopy(M.diagonals)))
    return res
end


"""
    convert(::Type{LowerTridiagonalMatrix{T}}, M::LowerTridiagonalMatrix) where {T}

Convert type of values of matrix represented by lower tridiagonal matrix type.
"""
function convert(::Type{LowerTridiagonalMatrix{T}}, M::LowerTridiagonalMatrix) where {T}
    res = LowerTridiagonalMatrix{T}(convert(Array{Array{T, 1},1}, deepcopy(M.diagonals)))
    return res
end


"""
    copy(M::TridiagonalMatrix)

Make a copy of a tridiagonal matrix type instance.
"""
function copy(M::TridiagonalMatrix)
    return typeof(M)(deepcopy(M.diagonals))
end


"""
    copy(M::UpperTridiagonalMatrix)

Make a copy of a tridiagonal matrix type instance.
"""
function copy(M::UpperTridiagonalMatrix)
    return typeof(M)(deepcopy(M.diagonals))
end


"""
    copy(M::LowerTridiagonalMatrix)

Make a copy of a tridiagonal matrix type instance.
"""
function copy(M::LowerTridiagonalMatrix)
    return typeof(M)(deepcopy(M.diagonals))
end


"""
    getindex(M::TridiagonalMatrix, idx::Int)

Perform linear indexing of matrix represented by tridiagonal matrix type instance.

# Examples
```julia-repl
julia> td = TridiagonalMatrix{Int64}([[1, 4, 3], [4, 2, 3, 4], [4, 7, 6]])
4×4 TridiagonalMatrix{Int64}:
 4  4  0  0
 1  2  7  0
 0  4  3  6
 0  0  3  4

julia> td[6]
2

julia> td[3]
0
```
"""
function getindex(M::TridiagonalMatrix{T}, idx::Int) where T
    
    # If linear index out of bounds, throw error.
    if idx > length(M.diagonals[2])^2
        throw(BoundsError())
    end

    # Convert to Cartesian index.
    cart_idx = CartesianIndices(size(M))[idx]
    
    # If difference between values in cartesian indices greater than 1 return 0.
    if abs(cart_idx[2] - cart_idx[1]) > 1
        return zero(T)
    else
        # Compute difference between values in cartesian indices.
        cart_idx_diff = cart_idx[2] - cart_idx[1]

        # Return appropriate element on appropriate diagonal.
        return cart_idx_diff < 0 ? M.diagonals[1][cart_idx[2]] : M.diagonals[2 + cart_idx_diff][cart_idx[1]]
    end
end


"""
    getindex(M::UpperTridiagonalMatrix, idx::Int)

Perform linear indexing of matrix represented by upper tridiagonal matrix type instance.

# Examples
```julia-repl
julia> utd = UpperTridiagonalMatrix{Int64}([[4, 2, 3, 4], [4, 7, 6]])
4×4 UpperTridiagonalMatrix{Int64}:
 4  4  0  0
 0  2  7  0
 0  0  3  6
 0  0  0  4

julia> utd[5]
4

julia> utd[7]
0
```
"""
function getindex(M::UpperTridiagonalMatrix{T}, idx::Int) where T
    
    # If linear index out of bounds, throw error.
    if idx > length(M.diagonals[1])^2
        throw(BoundsError())
    end

    # Convert to Cartesian index.
    cart_idx = CartesianIndices(size(M))[idx]
    
    # If difference between values in cartesian indices greater than 1 
    # or if element below main diagonal, return 0.
    if abs(cart_idx[2] - cart_idx[1]) > 1 || cart_idx[2] - cart_idx[1] < 0
        return zero(T)
    else
        # Compute difference between values in cartesian indices.
        cart_idx_diff = cart_idx[2] - cart_idx[1]

        # Return appropriate element on appropriate diagonal.
        return M.diagonals[1 + cart_idx_diff][cart_idx[1]]
    end
end


"""
    getindex(M::LowerTridiagonalMatrix, idx::Int)

Perform linear indexing of matrix represented by lower tridiagonal matrix type instance.

# Examples
```julia-repl
julia> ltd = LowerTridiagonalMatrix{Int64}([[4, 2, 3], [4, 7, 6, 1]])
4×4 LowerTridiagonalMatrix{Int64}:
 4  0  0  0
 4  7  0  0
 0  2  6  0
 0  0  3  1

julia> ltd[6]
7

julia> ltd[8]
0
```
"""
function getindex(M::LowerTridiagonalMatrix{T}, idx::Int) where T
    
    # If linear index out of bounds, throw error.
    if idx > length(M.diagonals[2])^2
        throw(BoundsError())
    end

    # Convert to Cartesian index.
    cart_idx = CartesianIndices(size(M))[idx]
    
    # If difference between values in cartesian indices greater than 1
    # or if element above main diagonal, return 0.
    if abs(cart_idx[2] - cart_idx[1]) > 1 || cart_idx[2] - cart_idx[1] > 0
        return zero(T)
    else
        # Compute difference between values in cartesian indices.
        cart_idx_diff = cart_idx[2] - cart_idx[1]

        # Return appropriate element on appropriate diagonal.
        return M.diagonals[2+cart_idx_diff][cart_idx[2]]
    end
end


"""
    getindex(M::TridiagonalMatrix, idx::Vararg{Int, 2})

Perform cartesian indexing of matrix represented by tridiagonal matrix type instance.

# Examples
```julia-repl
julia> td = TridiagonalMatrix{Int64}([[1, 4, 3], [4, 2, 3, 4], [4, 7, 6]])
4×4 TridiagonalMatrix{Int64}:
 4  4  0  0
 1  2  7  0
 0  4  3  6
 0  0  3  4

julia> td[2, 2]
2

julia> td[3, 1]
0
```
"""
function getindex(M::TridiagonalMatrix{T}, idx::Vararg{Int, 2}) where T
    
    # Build cartesian index.
    cart_idx = CartesianIndex(idx[1], idx[2])
    
    # If cartesian index out of bounds, throw error.
    if cart_idx[1] > length(M.diagonals[2]) || cart_idx[2] > length(M.diagonals[2])
        throw(BoundsError())
    end
    
    # If difference between values in cartesian indices greater than 1, return 0.
    if abs(cart_idx[2] - cart_idx[1]) > 1
        return zero(T)
    else
        
        # Compute difference between values in cartesian indices.
        cart_idx_diff = cart_idx[2] - cart_idx[1]
        
        # Return appropriate element on appropriate diagonal.
        return cart_idx_diff < 0 ? M.diagonals[1][cart_idx[2]] : M.diagonals[2 + cart_idx_diff][cart_idx[1]]
    end
end


"""
    getindex(M::UpperTridiagonalMatrix, idx::Vararg{Int, 2})

Perform cartesian indexing of matrix represented by upper tridiagonal matrix type instance.

# Examples
```julia-repl
julia> utd = UpperTridiagonalMatrix{Int64}([[4, 2, 3, 4], [4, 7, 6]])
4×4 UpperTridiagonalMatrix{Int64}:
 4  4  0  0
 0  2  7  0
 0  0  3  6
 0  0  0  4

julia> utd[2, 3]
7

julia> utd[3, 2]
0
```
"""
function getindex(M::UpperTridiagonalMatrix{T}, idx::Vararg{Int, 2}) where T
    
    # Build cartesian index.
    cart_idx = CartesianIndex(idx[1], idx[2])
    
    # If cartesian index out of bounds, throw error.
    if cart_idx[1] > length(M.diagonals[1]) || cart_idx[2] > length(M.diagonals[1])
        throw(BoundsError())
    end
    
    # If difference between values in cartesian indices greater than 1
    # or if element below main diagonal, return 0.
    if abs(cart_idx[2] - cart_idx[1]) > 1 || cart_idx[2] - cart_idx[1] < 0
        return zero(T)
    else
        # Compute difference between values in cartesian indices.
        cart_idx_diff = cart_idx[2] - cart_idx[1]

        # Return appropriate element on appropriate diagonal.
        return M.diagonals[1 + cart_idx_diff][cart_idx[1]]
    end
end


"""
    getindex(M::LowerTridiagonalMatrix, idx::Vararg{Int, 2})

Perform cartesian indexing of matrix represented by lower tridiagonal matrix type instance.

# Examples
```julia-repl
julia> ltd = LowerTridiagonalMatrix{Int64}([[4, 2, 3], [4, 7, 6, 1]])
4×4 LowerTridiagonalMatrix{Int64}:
 4  0  0  0
 4  7  0  0
 0  2  6  0
 0  0  3  1

julia> ltd[2, 1]
4

julia> ltd[1, 3]
0
```
"""
function getindex(M::LowerTridiagonalMatrix{T}, idx::Vararg{Int, 2}) where T
    
    # Build cartesian index.
    cart_idx = CartesianIndex(idx[1], idx[2])
    
    # If cartesian index out of bounds, throw error.
    if cart_idx[1] > length(M.diagonals[2]) || cart_idx[2] > length(M.diagonals[2])
        throw(BoundsError())
    end
    
    # If difference between values in cartesian indices greater than 1 
    # or if element above main diagonal, return 0.
    if abs(cart_idx[2] - cart_idx[1]) > 1 || cart_idx[2] - cart_idx[1] > 0
        return zero(T)
    else
        # Compute difference between values in cartesian indices.
        cart_idx_diff = cart_idx[2] - cart_idx[1]

        # Return appropriate element on appropriate diagonal.
        return M.diagonals[2+cart_idx_diff][cart_idx[2]]
    end
    
end


"""
    setindex!(M::TridiagonalMatrix, val, idx::Int)

Set element at specified linear index in tridiagonal matrix type instance.

# Examples
```julia-repl
julia> td = TridiagonalMatrix{Int64}([[1, 4, 3], [4, 2, 3, 4], [4, 7, 6]])
4×4 TridiagonalMatrix{Int64}:
 4  4  0  0
 1  2  7  0
 0  4  3  6
 0  0  3  4

julia> td[7] = 11
11

julia> td
4×4 TridiagonalMatrix{Int64}:
 4   4  0  0
 1   2  7  0
 0  11  3  6
 0   0  3  4
```
"""
function setindex!(M::TridiagonalMatrix, val, idx::Int)

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
    setindex!(M::UpperTridiagonalMatrix, val, idx::Int)

Set element at specified linear index in upper tridiagonal matrix type instance.

# Examples
```julia-repl
julia> utd = UpperTridiagonalMatrix{Int64}([[4, 2, 3, 4], [4, 7, 6]])
4×4 UpperTridiagonalMatrix{Int64}:
 4  4  0  0
 0  2  7  0
 0  0  3  6
 0  0  0  4

julia> utd[10] = -12
-12

julia> utd
4×4 UpperTridiagonalMatrix{Int64}:
 4  4    0  0
 0  2  -12  0
 0  0    3  6
 0  0    0  4
```
"""
function setindex!(M::UpperTridiagonalMatrix, val, idx::Int)

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
    setindex!(M::LowerTridiagonalMatrix, val, idx::Int)

Set element at specified linear index in lower tridiagonal matrix type instance.

# Examples
```julia-repl
julia> ltd = LowerTridiagonalMatrix{Int64}([[4, 2, 3], [4, 7, 6, 1]])
4×4 LowerTridiagonalMatrix{Int64}:
 4  0  0  0
 4  7  0  0
 0  2  6  0
 0  0  3  1

julia> ltd[7] = -3
-3

julia> ltd
4×4 LowerTridiagonalMatrix{Int64}:
 4   0  0  0
 4   7  0  0
 0  -3  6  0
 0   0  3  1
```
"""
function setindex!(M::LowerTridiagonalMatrix, val, idx::Int)

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
    setindex!(M::TridiagonalMatrix, val, idx::Vararg{Int, 2})

Set element at specified cartesian index in tridiagonal matrix type instance.

# Examples
```julia-repl
julia> td = TridiagonalMatrix{Int64}([[1, 4, 3], [4, 2, 3, 4], [4, 7, 6]])
4×4 TridiagonalMatrix{Int64}:
 4  4  0  0
 1  2  7  0
 0  4  3  6
 0  0  3  4

julia> td[3, 2]= -3
-3

julia> td
4×4 TridiagonalMatrix{Int64}:
 4   4  0  0
 1   2  7  0
 0  -3  3  6
 0   0  3  4
```
"""
function setindex!(M::TridiagonalMatrix, val, idx::Vararg{Int, 2})
     
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
    setindex!(M::UpperTridiagonalMatrix, val, idx::Vararg{Int, 2})

Set element at specified cartesian index in upper tridiagonal matrix type instance.

# Examples
```julia-repl
julia> utd = UpperTridiagonalMatrix{Int64}([[4, 2, 3, 4], [4, 7, 6]])
4×4 UpperTridiagonalMatrix{Int64}:
 4  4  0  0
 0  2  7  0
 0  0  3  6
 0  0  0  4

julia> utd[3, 4] = -5
-5

julia> utd
4×4 UpperTridiagonalMatrix{Int64}:
 4  4  0   0
 0  2  7   0
 0  0  3  -5
 0  0  0   4
```
"""
function setindex!(M::UpperTridiagonalMatrix, val, idx::Vararg{Int, 2})
     
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
    setindex!(M::LowerTridiagonalMatrix, val, idx::Vararg{Int, 2})

Set element at specified cartesian index in lower tridiagonal matrix type instance.

# Examples
```julia-repl
julia> ltd = LowerTridiagonalMatrix{Int64}([[4, 2, 3], [4, 7, 6, 1]])
4×4 LowerTridiagonalMatrix{Int64}:
 4  0  0  0
 4  7  0  0
 0  2  6  0
 0  0  3  1

julia> ltd[4, 3] = -2
-2

julia> ltd
4×4 LowerTridiagonalMatrix{Int64}:
 4  0   0  0
 4  7   0  0
 0  2   6  0
 0  0  -2  1
```
"""
function setindex!(M::LowerTridiagonalMatrix, val, idx::Vararg{Int, 2})
     
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
    *(M::TridiagonalMatrix{T}, v::Vector{S})
    
Perform multiplication of tridiagonal matrix type instance with vector.

# Examples
```julia-repl
julia> td = TridiagonalMatrix{Int64}([[1, 4, 3], [4, 2, 3, 4], [4, 7, 6]])
4×4 TridiagonalMatrix{Int64}:
 4  4  0  0
 1  2  7  0
 0  4  3  6
 0  0  3  4

julia> v = [1, 2, 3, 2]
4-element Array{Int64,1}:
 1
 2
 3
 2

julia> td*v
4-element Array{Float64,1}:
 12.0
 26.0
 29.0
 17.0
```
"""
function *(M::TridiagonalMatrix{T}, v::Vector{S}) where {T, S}
    
    # Dimensions match test.
    if length(v) != length(M.diagonals[2])
        throw(DimensionMismatch(@sprintf("Matrix A has dimensions %s, vector B has length %d", size(M), length(v))))
    end
    
    # Allocate vector of promoted type resulting from multiplication of value types
    # in matrix and vector for storing results.
    R = typeof(oneunit(T) * oneunit(S))
    res = Vector{R}(undef, length(v))
    
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


"""
    *(M::UpperTridiagonalMatrix{T}, v::Vector{S})
    
Perform multiplication of upper tridiagonal matrix type instance with vector.

# Examples
```julia-repl
julia> utd = UpperTridiagonalMatrix{Int64}([[4, 2, 3, 4], [4, 7, 6]])
4×4 UpperTridiagonalMatrix{Int64}:
 4  4  0  0
 0  2  7  0
 0  0  3  6
 0  0  0  4

julia> v = [1, 2, 3, 2]
4-element Array{Int64,1}:
 1
 2
 3
 2

julia> utd*v
4-element Array{Float64,1}:
 12.0
 25.0
 21.0
  8.0
```
"""
function *(M::UpperTridiagonalMatrix{T}, v::Vector{S}) where {T, S}
    
    # Dimensions match test.
    if length(v) != length(M.diagonals[1])
        throw(DimensionMismatch(@sprintf("Matrix A has dimensions %s, vector B has length %d", size(M), length(v))))
    end

    # Allocate vector of promoted type resulting from multiplication of value types
    # in matrix and vector for storing results.
    R = typeof(oneunit(T) * oneunit(S))
    res = Vector{R}(undef, length(v))
    
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


"""
    *(M::LowerTridiagonalMatrix{T}, v::Vector{S})
    
Perform multiplication of lower tridiagonal matrix type instance with vector.

# Examples
```julia-repl
julia> ltd = LowerTridiagonalMatrix{Int64}([[4, 2, 3], [4, 7, 6, 1]])
4×4 LowerTridiagonalMatrix{Int64}:
 4  0  0  0
 4  7  0  0
 0  2  6  0
 0  0  3  1

julia> v = [1, 2, 3, 2]
4-element Array{Int64,1}:
 1
 2
 3
 2

julia> ltd*v
4-element Array{Float64,1}:
  4.0
 18.0
 22.0
 11.0
```
"""
function *(M::LowerTridiagonalMatrix{T}, v::Vector{S}) where {T, S}
    
    # Dimensions match test.
    if length(v) != length(M.diagonals[2])
        throw(DimensionMismatch(@sprintf("Matrix A has dimensions %s, vector B has length %d", size(M), length(v))))
    end

    # Allocate vector of promoted type resulting from multiplication of value types
    # in matrix and vector for storing results.
    R = typeof(oneunit(T) * oneunit(S))
    res = Vector{R}(undef, length(v))
    
    # Set start and end indices for computing the result elements only using explicitly stored elements.
    end_idx = 1
    start_idx = 0

    # Compute product by shifting indices over explicitly stored elements.
    for idx = 1:length(v)
        res[idx] = dot(M[idx, max(start_idx, 1):min(end_idx, length(M.diagonals[2]))], 
                       v[max(start_idx, 1):min(end_idx, length(M.diagonals[2]))])
        start_idx += 1
        end_idx += 1
    end

    # Return result.
    return res
end


"""
    lu(M::TridiagonalMatrix{T})

Perform LU decomposition of tridiagonal matrix type instance.

# Examples
```julia-repl
julia> td = TridiagonalMatrix{Int64}([[1, 4, 3], [5, 4, 6, 4], [4, 2, 1]])
4×4 TridiagonalMatrix{Int64}:
 5  4  0  0
 1  4  2  0
 0  4  6  1
 0  0  3  4

julia> l, u = lu(td);

julia> display(l)
4×4 LowerTridiagonalMatrix{Float64}:
 1.0  0     0         0
 0.2  1.0   0         0
 0    1.25  1.0       0
 0    0     0.857143  1.0

julia> display(u)
4×4 UpperTridiagonalMatrix{Float64}:
 5.0  4.0  0    0
 0    3.2  2.0  0
 0    0    3.5  1.0
 0    0    0    3.14286

julia> display(l*u)
4×4 Array{Float64,2}:
 5.0  4.0  0.0  0.0
 1.0  4.0  2.0  0.0
 0.0  4.0  6.0  1.0
 0.0  0.0  3.0  4.0
```
"""
function lu(M::TridiagonalMatrix{T}) where T
    
    # Initialize matrix on which to perform elimination.
    # Convert to promoted type resulting from division and multiplication.
    S = typeof(oneunit(T) * (oneunit(T) / oneunit(T)))
    M_el = convert(TridiagonalMatrix{S}, M)

    # Eliminate elements in column below main diagonal. Build elimination matrix in-place.
    for idx = 1:length(M_el.diagonals[2])-1
        if M_el[idx+1, idx] != zero(T)
            if M_el[idx, idx] == zero(T)
                throw(SingularException(idx))
            end
            mult = -M_el[idx+1, idx]/M_el[idx, idx]
            M_el[idx+1, idx:idx+1] += mult*M_el[idx, idx:idx+1]
            M_el[idx+1, idx] = -mult
        end
    end

    # Initialize lower and upper tridiagonal matrix type instances as results of decomposition and return.
    l = LowerTridiagonalMatrix{S}(vcat([M_el.diagonals[1]], [ones(length(M.diagonals[2]))]))
    u = UpperTridiagonalMatrix{S}(M_el.diagonals[2:end])
    return l, u
end


"""
    lu(M::UpperTridiagonalMatrix{T})

Perform LU decomposition of upper tridiagonal matrix type instance.

# Examples
```julia-repl
julia> utd = UpperTridiagonalMatrix{Int64}([[5, 7, 6, 5], [4, 6, 4]])
4×4 UpperTridiagonalMatrix{Int64}:
 5  4  0  0
 0  7  6  0
 0  0  6  4
 0  0  0  5

julia> l, u = lu(utd);

julia> display(l)
4×4 LowerTridiagonalMatrix{Float64}:
 1.0  0    0    0  
 0.0  1.0  0    0  
 0    0.0  1.0  0  
 0    0    0.0  1.0

julia> display(u)
4×4 UpperTridiagonalMatrix{Float64}:
 5.0  4.0  0    0  
 0    7.0  6.0  0  
 0    0    6.0  4.0
 0    0    0    5.0

julia> display(l*u)
4×4 Array{Float64,2}:
 5.0  4.0  0.0  0.0
 0.0  7.0  6.0  0.0
 0.0  0.0  6.0  4.0
 0.0  0.0  0.0  5.0
```
"""
function lu(M::UpperTridiagonalMatrix{T}) where T
    
    # Initialize matrix on which to perform elimination.
    # Convert to promoted type resulting from division and multiplication.
    S = typeof(oneunit(T) * (oneunit(T) / oneunit(T)))
    M_el = convert(UpperTridiagonalMatrix{S}, M)
    
    # Initialize lower and upper tridiagonal matrix type instances as results of decomposition and return.
    l = LowerTridiagonalMatrix{S}(vcat([zeros(length(M_el.diagonals[1])-1)], [ones(length(M_el.diagonals[1]))]))
    u = UpperTridiagonalMatrix{S}(M_el.diagonals)
    return l, u
end


"""
    lu(M::LowerTridiagonalMatrix{T})

Perform LU decomposition of lower tridiagonal matrix type instance.

# Examples
```julia-repl
julia> ltd = LowerTridiagonalMatrix{Int64}([[4, 2, 3], [4, 7, 6, 5]])
4×4 LowerTridiagonalMatrix{Int64}:
 4  0  0  0
 4  7  0  0
 0  2  6  0
 0  0  3  5

julia> l, u = lu(utd);

julia> display(l)
4×4 LowerTridiagonalMatrix{Float64}:
 1.0  0    0    0  
 0.0  1.0  0    0  
 0    0.0  1.0  0  
 0    0    0.0  1.0

julia> display(u)
4×4 UpperTridiagonalMatrix{Float64}:
 5.0  4.0  0    0  
 0    7.0  6.0  0  
 0    0    6.0  4.0
 0    0    0    5.0

julia> display(l*u)
4×4 Array{Float64,2}:
 5.0  4.0  0.0  0.0
 0.0  7.0  6.0  0.0
 0.0  0.0  6.0  4.0
 0.0  0.0  0.0  5.0
```
"""
function lu(M::LowerTridiagonalMatrix{T}) where T
    
    # Initialize matrix on which to perform elimination.
    # Convert to promoted type resulting from division and multiplication.
    S = typeof(oneunit(T) * (oneunit(T) / oneunit(T)))
    M_el = convert(LowerTridiagonalMatrix{S}, M)
    
    # Eliminate elements in column below main diagonal. Build elimination matrix in-place.
    for idx = 1:length(M_el.diagonals[2])-1
        if M_el[idx, idx] == zero(T)
            throw(SingularException(idx))
        end
        mult = -M_el[idx+1, idx]/M_el[idx, idx]
        M_el[idx+1, idx] += mult*M_el[idx, idx]
        M_el[idx+1, idx] = -mult
    end

    # Initialize lower and upper tridiagonal matrix type instances as results of decomposition and return.
    l = LowerTridiagonalMatrix{S}(vcat([M_el.diagonals[1]], [ones(length(M.diagonals[2]))]))
    u = UpperTridiagonalMatrix{S}([M_el.diagonals[2], zeros(length(M.diagonals[2])-1)])
    return l, u
end


"""
    \\(M::TridiagonalMatrix{T}, b::Vector{S}) where {T, S}

Solve systems of linear equations Ax = B for x where A is a tridiagonal matrix.

# Examples
```julia-repl
julia> td = TridiagonalMatrix{Int64}([[1, 4, 3], [4, 2, 3, 4], [4, 7, 6]])
4×4 TridiagonalMatrix{Int64}:
 4  4  0  0
 1  2  7  0
 0  4  3  6
 0  0  3  4

julia> v = [1, 2, 3, 4]
4-element Array{Int64,1}:
 1
 2
 3
 4

julia> x = td\v
4-element Array{Float64,1}:
  0.8728813559322037 
 -0.6228813559322037 
  0.33898305084745767
  0.7457627118644068 

julia> td*x
4-element Array{Float64,1}:
 1.0              
 2.0              
 2.999999999999999
 4.0          
```
"""
function \(M::TridiagonalMatrix{T}, b::Vector{S}) where {T, S}
    
    # Dimensions match test.
    if length(b) != length(M.diagonals[2])
        throw(DimensionMismatch(@sprintf("B has leading dimension %d, but needs %d", length(b), length(M.diagonals[2]))))
    end
    
    # Perform LU decomposition.
    l, u = lu(M)
    
    ### Solve Ly = b for y. ###

    # Initialize y vector of promoted type resulting from performed operations 
    # with value types in matrix and vector b.
    R = typeof((oneunit(S) - oneunit(T) * oneunit(S)) / oneunit(T))
    y = convert(Vector{R}, copy(b))

    # Perform forward substitions to compute y vector.
    y[1] = y[1]/l[1, 1]
    for idx = 2:length(y)
        y[idx] = (y[idx] - l[idx, idx-1]*y[idx-1])/l[idx, idx]
    end

    ### Solve Ux = y for x. ###
    
    # Initialize x vector.
    x = copy(y)

    # Perform backward substitions to compute x vector.
    x[end] = x[end]/u[end, end]
    for idx = length(x)-1:-1:1
        x[idx] = (x[idx] - u[idx, idx+1]*x[idx+1])/u[idx, idx]
    end

    # Return solution.
    return x
end


"""
    \\(M::UpperTridiagonalMatrix{T}, b::Vector{S}) where {T, S}

Solve systems of linear equations Ax = B for x where A is an upper tridiagonal matrix.

# Examples
```julia-repl
julia> utd = UpperTridiagonalMatrix{Int64}([[4, 2, 3, 4], [4, 7, 6]])
4×4 UpperTridiagonalMatrix{Int64}:
 4  4  0  0
 0  2  7  0
 0  0  3  6
 0  0  0  4

julia> v = [1, 2, 3, 4]
4-element Array{Int64,1}:
 1
 2
 3
 4

julia> x = td\v
4-element Array{Float64,1}:
  0.8728813559322037 
 -0.6228813559322037 
  0.33898305084745767
  0.7457627118644068 

julia> td*x
4-element Array{Float64,1}:
 1.0              
 2.0              
 2.999999999999999
 4.0        
```
"""
function \(M::UpperTridiagonalMatrix{T}, b::Vector{S}) where {T, S}
    
    # Dimensions match test.
    if length(b) != length(M.diagonals[1])
        throw(DimensionMismatch(@sprintf("B has leading dimension %d, but needs %d", length(b), length(M.diagonals[2]))))
    end
    
   
    # Initialize x vector of promoted type resulting from performed operations 
    # with value types in matrix and vector b.
    R = typeof((oneunit(S) - oneunit(T) * oneunit(S)) / oneunit(T))
    x = convert(Vector{R}, copy(b))

    # Perform backward substitions to compute x vector.
    x[end] = x[end]/M[end, end]
    for idx = length(x)-1:-1:1
        x[idx] = (x[idx] - M[idx, idx+1]*x[idx+1])/M[idx, idx]
    end

    # Return solution.
    return x
end


"""
    \\(M::LowerTridiagonalMatrix{T}, b::Vector{S}) where {T, S}

Solve systems of linear equations Ax = B for x where A is a lower tridiagonal matrix.

# Examples
```julia-repl
julia> ltd = LowerTridiagonalMatrix{Int64}([[4, 2, 3], [4, 7, 6, 1]])
4×4 LowerTridiagonalMatrix{Int64}:
 4  0  0  0
 4  7  0  0
 0  2  6  0
 0  0  3  1

julia> v = [1, 2, 3, 4]
4-element Array{Int64,1}:
 1
 2
 3
 4

julia> x = td\v
4-element Array{Float64,1}:
  0.8728813559322037
 -0.6228813559322037
  0.33898305084745767
  0.7457627118644068

julia> td*x
4-element Array{Float64,1}:
 1.0
 2.0
 2.999999999999999
 4.0
```
"""
function \(M::LowerTridiagonalMatrix{T}, b::Vector{S}) where {T, S}
    
    # Dimensions match test.
    if length(b) != length(M.diagonals[2])
        throw(DimensionMismatch(@sprintf("B has leading dimension %d, but needs %d", length(b), length(M.diagonals[2]))))
    end
    
    # Initialize x vector of promoted type resulting from performed operations 
    # with value types in matrix and vector b.
    R = typeof((oneunit(S) - oneunit(T) * oneunit(S)) / oneunit(T))
    x = convert(Vector{R}, copy(b))

    # Perform forward substitions to compute x vector.
    x[1] = x[1]/M[1, 1]
    for idx = 2:length(x)
        x[idx] = (x[idx] - M[idx, idx-1]*x[idx-1])/M[idx, idx]
    end

    # Return solution.
    return x
end


"""
    householder_matrix(u::Vector{T}, padding::Int=0) where T

Construct Householder matrix using vector u. Pad matrix with specified
number of rows and columns to get form required for matrix element
eliminations.
"""
function householder_matrix(u::Vector{T}, padding::Int=0) where T

    # Create Householder matrix.
    H = Matrix{T}(I, length(u), length(u)) - 2*(u*u')/(u'*u)

    # Pad matrix to get form required for matrix element eliminations.
    if padding > 0
        padded_l = hcat(zeros(size(H, 1), padding), H)
        padded_t = vcat(zeros(padding, size(padded_l, 2)), padded_l)
        padded_t[diagind(padded_t)[1:padding]] .= oneunit(T)
        return padded_t
    end

    return H
end


"""
    tridiag(M::Array{T,2}) where T
    
Construct tridiagonal matrix that is similar to specified real symmetric matrix.
Return tridiagonal matrix type instance representing the result and the matrix
Q representing the transformation matrix.

# Examples
```julia-repl
julia> M = rand(4, 4)
4×4 Array{Float64,2}:
 0.901219  0.50817    0.817667   0.309866
 0.521819  0.78728    0.0261446  0.452818
 0.433704  0.0943157  0.499424   0.575889
 0.255353  0.542199   0.846987   0.804865

julia> M_symm = M + M'
4×4 Array{Float64,2}:
 1.80244   1.02999   1.25137   0.565219
 1.02999   1.57456   0.12046   0.995017
 1.25137   0.12046   0.998847  1.42288 
 0.565219  0.995017  1.42288   1.60973 

julia> td, Q = tridiag(M_symm);

julia> display(td)
4×4 TridiagonalMatrix{Float64}:
  1.80244  -1.71647   0.0        0.0      
 -1.71647   2.45417  -1.45699    0.0      
  0.0      -1.45699   0.44968    0.0976526
  0.0       0.0       0.0976526  1.27929  

julia> display(Q)
4×4 Array{Float64,2}:
 1.0   0.0         0.0        0.0     
 0.0  -0.600061   -0.729036  -0.329291
 0.0  -0.0771091  -0.357007   0.930914
 0.0  -0.796229    0.583996   0.158011
```
"""
function tridiag(M::Array{T,2}) where T
    
    # Check if matrix symmetric.
    if M != M'
        throw(DomainError(M, "matrix must be symmetric"))
    end

    # Create copy of matrix.
    M_nxt = copy(M)

    # Initialize matrix for constructing the transformation matrix Q.
    Q = Matrix{T}(I, size(M))

    # Go over rows and eliminate elements below and above
    # the tridiagonal band using householder transformations.
    for idx = 1:size(M,1) - 2
        u = M_nxt[idx+1:end, idx]
        u[1] += sign(u[1])*norm(u)

        # Construct Householder matrix with padding.
        H = householder_matrix(u, idx)

        # Multiply with previous transformation matrix Q to get 
        # next transformation matrix Q.
        Q = H*Q

        # Use the Householder matrix to eliminate elements below
        # and above the tridiagonal band.
        M_nxt = H*M_nxt*H
    end
    
    # Construct tridiagonal matrix type instance from result.
    trid_res = TridiagonalMatrix{Float64}([M_nxt[2:size(M_nxt,1)+1:prod(size(M))], 
                                           M_nxt[1:size(M_nxt,1)+1:prod(size(M))], 
                                           M_nxt[size(M_nxt,1)+1:size(M_nxt,1)+1:prod(size(M))]])

    return trid_res, Q
end


"""
    inv_eigen(M::Array{T, 2}, eig_approx, thresh_conv=1.0e-4, max_it=1000) where T

Perform shifted inverse power method to estimate eigenvalue closest to provided estimate
as well as the corresponding eigenvector of the similar tridiagonal matrix.

# Examples
```julia-repl

julia> M = rand(4, 4);

julia> M = M + M'
4×4 Array{Float64,2}:
 0.567611  0.894417  0.469564  1.30955
 0.894417  0.976549  1.07652   1.0006
 0.469564  1.07652   1.9806    1.17661
 1.30955   1.0006    1.17661   0.77642


julia> e, x = inv_eigen(M, 1.0);

julia> display(e)
0.9089243683508385

julia> x
4-element Array{Float64,1}:
  0.7300715741712198
 -0.15066351908938347
  1.0
 -0.3459300293265851

julia> e, x = inv_eigen(M, 3.0);

julia> display(e)
4.121324137256185

julia> x
4-element Array{Float64,1}:
 -0.4653932197634543
  1.0
  0.47845258384333117

julia> eigvals(M)
4-element Array{Float64,1}:
 -0.741419917210247
  0.012357613193294056
  0.9089245103804795
  4.121317072870224
```
"""
function inv_eigen(M::Array{T, 2}, eig_approx, thresh_conv=1.0e-4, max_it=1000) where T
    
    # Check if matrix symmetric.
    if M != M'
        throw(DomainError(M, "matrix must be symmetric"))
    end
    
    # Get tridiagonal matrix similar to M.
    td, Q = tridiag(M)

    # Shift matrix.
    td.diagonals[2] .-= eig_approx

    # Perform LU decomposition.
    l, u = lu(td)

    # Initialize convergence flag.
    conv_flag = false

    # Set initial eigenvector guess.
    x_prev = ones(size(M, 1))
    
    # Initialize eigenvector and eigenvalue estimates.
    eigval_est = nothing
    eigvec_est = nothing
    
    # Initialize iteration counter.
    it_count = 0

    # Iterate until convergence or until maximum number of iterations reached.
    while !conv_flag && it_count < max_it

        # Increment iteration counter.
        it_count += 1

        # Solve system of equations for next value of eigenvector x.
        y = l\x_prev
        x_nxt = u\y

        # Get approximated eigenvalue of inverse matrix
        # by taking maximum entry by absolute value in eigenvector.
        max_el, min_el = maximum(x_nxt), minimum(x_nxt)
        eig_inv_approx = abs(max_el) > abs(min_el) ? max_el : min_el

        # Divide eigenvector estimate by approximated eigenvalue.
        x_nxt /= eig_inv_approx

        # If change in eigenvector estimation small enough,
        # declare convergence.
        if norm(x_nxt - x_prev) < thresh_conv || it_count == max_it
            conv_flag = true
            eigval_est = 1/eig_inv_approx + eig_approx
            eigvec_est = x_nxt
        else
            x_prev = x_nxt
        end
    end

    # Return estimate of eigenvector of similar tridiagonal matrix and
    # estimated eigenvalue closest to provided estimate.
    return eigval_est, Q'*eigvec_est
end

end

