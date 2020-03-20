module BandMatrices

import Base:*,\,convert, copy
import LinearAlgebra:dot,lu
using Printf
using InteractiveUtils  # Prevent subtypes not defined error.

export BandMatrix, UpperBandMatrix, LowerBandMatrix, lu, *, \


"""
    BandMatrix{T} <: AbstractArray{T, 2}

Type representing a center-band band matrix where only the specified diagonals are explicitly stored.
The matrix is initialized by specifying the band diagonals as an array of arrays.

# Examples
```julia-repl
julia> M = BandMatrix{Int64}([[3, 2, 7], [1, 5, 3, 2], [8, 4, 6]])
4×4 BandMatrix{Int64}:
 1  8  0  0
 3  5  4  0
 0  2  3  6
 0  0  7  2
```
"""
struct BandMatrix{T} <: AbstractArray{T, 2}
    
    # Band diagonals that are explicitly stored by the matrix
    diagonals::Array{Array{T,1},1}
    
    # Constructor
    function BandMatrix{T}(diagonals::Array{Array{T,1},1}) where {T}
        
        # Check if diagonals specified and if and odd number of diagonals given.
        if length(diagonals) == 0 || mod(length(diagonals), 2) != 1
            error("diagonals not specified correctly")
        else

            # Get length of center diagonal.
            center_idx = div(length(diagonals), 2) + 1
            center_len = length(diagonals[center_idx])

            # Check if diagonal lengths decreasing by one when moving away from central diagonal.
            for offset = 1:center_idx-1
                if length(diagonals[center_idx-offset]) != center_len - offset || length(diagonals[center_idx+offset]) != center_len - offset
                    error("diagonals not specified correctly")
                end
            end

            # Construct matrix.
            new(diagonals)
        end
    end
end


"""
    UpperBandMatrix{T} <: AbstractArray{T, 2}

Type representing a upper-band band matrix where only the specified diagonals are explicitly stored.
The matrix is initialized by specifying the band diagonals as an array of arrays.

# Examples
```julia-repl
julia> M = UpperBandMatrix{Int64}([[1, 5, 3, 2], [8, 4, 6], [2, 2]])
4×4 UpperBandMatrix{Int64}:
 1  8  2  0
 0  5  4  2
 0  0  3  6
 0  0  0  2
```
"""
struct UpperBandMatrix{T} <: AbstractArray{T, 2}
    
    # Band diagonals that are explicitly stored by the matrix
    diagonals::Array{Array{T,1},1}

    # Constructor
    function UpperBandMatrix{T}(diagonals::Array{Array{T,1}, 1}) where {T}

        # Check if diagonals specified.
        if length(diagonals) == 0
            error("diagonals not specified correctly")
        else
            
            # Check if diagonal lengths decreasing by one when moving away from central diagonal.
            center_len = length(diagonals[1])
            for offset = 1:length(diagonals)-1
                if length(diagonals[1+offset]) != center_len - offset
                    error("diagonals not specified correctly")
                end
            end
        end
        
        # Construct matrix.
        new(diagonals)
    end
end


"""
    LowerBandMatrix{T} <: AbstractArray{T, 2}

Type representing a lower-band band matrix where only the specified diagonals are explicitly stored.
The matrix is initialized by specifying the band diagonals as an array of arrays.

# Examples
```julia-repl
julia> M = LowerBandMatrix{Int64}([[2, 1], [3, 3, 2], [4, 5, 5, 3]])
4×4 LowerBandMatrix{Int64}:
 4  0  0  0
 3  5  0  0
 2  3  5  0
 0  1  2  3
```
"""
struct LowerBandMatrix{T} <: AbstractArray{T, 2}
    
    # Band diagonals that are explicitly stored by the matrix
    diagonals::Array{Array{T,1},1}

    # Constructor
    function LowerBandMatrix{T}(diagonals::Array{Array{T,1}, 1}) where {T}

        # Check if diagonals specified.
        if length(diagonals) == 0
            error("diagonals not specified correctly")
        else
            # Check if diagonal lengths decreasing by one when moving away from central diagonal.
            center_len = length(diagonals[end])
            for offset = 1:length(diagonals)-1
                if length(diagonals[end-offset]) != center_len - offset
                    error("diagonals not specified correctly")
                end
            end
        end
        
        # Construct matrix.
        new(diagonals)
    end
end


"""
    Base.copy(M::Union{BandMatrix, UpperBandMatrix, LowerBandMatrix})

Make a deep copy of a band matrix type instance.
"""
function Base.copy(M::Union{BandMatrix, UpperBandMatrix, LowerBandMatrix})
    return typeof(M)(deepcopy(M.diagonals))
end


"""
    Base.size(M::Union{BandMatrix, UpperBandMatrix, LowerBandMatrix})

Get size of matrix represented by band matrix type.

# Examples
```julia-repl
julia> M = BandMatrix{Int64}([[3, 2, 7], [1, 5, 3, 2], [8, 4, 6]])
4×4 BandMatrix{Int64}:
 1  8  0  0
 3  5  4  0
 0  2  3  6
 0  0  7  2

julia> size(M)
(4, 4)
```
"""
function Base.size(M::Union{BandMatrix, UpperBandMatrix, LowerBandMatrix})

    # Get size by inspecting main diagonal length.
    if string(Base.typename(typeof(M))) == "BandMatrix"
        center_idx = div(length(M.diagonals), 2) + 1
        return (length(M.diagonals[center_idx]), length(M.diagonals[center_idx]))
    end
    if string(Base.typename(typeof(M))) == "UpperBandMatrix"
        return (length(M.diagonals[1]), length(M.diagonals[1]))
    end
    if string(Base.typename(typeof(M))) == "LowerBandMatrix"
        return (length(M.diagonals[end]), length(M.diagonals[end]))
    end
end


"""
    Base.getindex(M::Union{BandMatrix, UpperBandMatrix, LowerBandMatrix}, idx::Int)

Perform linear indexing of matrix represented by band matrix type instance.

# Examples
```julia-repl
julia> M = BandMatrix{Int64}([[3, 2, 7], [1, 5, 3, 2], [8, 4, 6]])
4×4 BandMatrix{Int64}:
 1  8  0  0
 3  5  4  0
 0  2  3  6
 0  0  7  2

julia> M[5]
8
```
"""
function Base.getindex(M::Union{BandMatrix, UpperBandMatrix, LowerBandMatrix}, idx::Int)

    # If linear index out of bounds, throw error.
    if idx > size(M, 1)^2
        throw(BoundsError())
    end

    # Convert to Cartesian index.
    cart_idx = CartesianIndices(size(M))[idx]

    # Compute offset of diagonal on which the element lies.
    # Negative indices mean diagonals below the main diagonal.
    # Positive indices mean diagonals above the main diagonal.
    # Note that if diagonal below main diagonal, we need to check the second
    # value in the Cartesian index to get the index in that diagonal.
    # If diagonal above main diagonal, we need to check the first value
    # in the Cartesian index.
    #
    offset = cart_idx[2] - cart_idx[1]
    
    if string(Base.typename(typeof(M))) == "BandMatrix"
        
        # Check if index on explicitly stored diagonal.
        if abs(offset) > div(length(M.diagonals), 2)
            return 0
        else
            return M.diagonals[div(length(M.diagonals), 2)+1+offset][offset >= 0 ? cart_idx[1] : cart_idx[2]]
        end
    end
    if string(Base.typename(typeof(M))) == "UpperBandMatrix"
        
        # Check if index on explicitly stored diagonal.
        if offset < 0 || offset > length(M.diagonals) - 1
            return 0
        else
            return M.diagonals[offset+1][cart_idx[1]]
        end
    end
    if string(Base.typename(typeof(M))) == "LowerBandMatrix"
        
        # Check if index on explicitly stored diagonal.
        if offset > 0 || -offset > length(M.diagonals) - 1
            return 0
        else
            return M.diagonals[end+offset][cart_idx[2]]
        end
    end
end


"""
    Base.getindex(M::Union{BandMatrix, UpperBandMatrix, LowerBandMatrix}, idx::Vararg{Int, 2})

Perform cartesian indexing of matrix represented by band matrix type instance.

# Examples
```julia-repl
julia> M = BandMatrix{Int64}([[3, 2, 7], [1, 5, 3, 2], [8, 4, 6]])
4×4 BandMatrix{Int64}:
 1  8  0  0
 3  5  4  0
 0  2  3  6
 0  0  7  2

julia> M[3, 2]
2
```
"""
function Base.getindex(M::Union{BandMatrix, UpperBandMatrix, LowerBandMatrix}, idx::Vararg{Int, 2})
        
    # Get Cartesian index.
    cart_idx = CartesianIndex(idx[1], idx[2])
    
    # If index out of bounds, throw error.
    if any(Tuple(cart_idx) .> size(M, 1))
        throw(BoundsError())
    end
    
    # See method for resolving linear indexing for explanation.

    offset = cart_idx[2] - cart_idx[1]
    if string(Base.typename(typeof(M))) == "BandMatrix"
        if abs(offset) > div(length(M.diagonals), 2)
            return 0
        else
            return M.diagonals[div(length(M.diagonals), 2)+1+offset][offset >= 0 ? cart_idx[1] : cart_idx[2]]
        end
    end
    if string(Base.typename(typeof(M))) == "UpperBandMatrix"
        if offset < 0 || offset > length(M.diagonals) - 1
            return 0
        else
            return M.diagonals[offset+1][cart_idx[1]]
        end
    end
    if string(Base.typename(typeof(M))) == "LowerBandMatrix"
        if offset > 0 || -offset > length(M.diagonals) - 1
            return 0
        else
            return M.diagonals[end+offset][cart_idx[2]]
        end
    end
end


"""
    Base.setindex!(M::Union{BandMatrix, UpperBandMatrix, LowerBandMatrix}, val, idx::Int)

Set element at specified linear index in band matrix type instance.

# Examples
```julia-repl
julia> M = BandMatrix{Int64}([[3, 2, 7], [1, 5, 3, 2], [8, 4, 6]])
4×4 BandMatrix{Int64}:
 1  8  0  0
 3  5  4  0
 0  2  3  6
 0  0  7  2

julia> M[5] = 2
2

julia> display(M)
4×4 BandMatrix{Int64}:
 1  2  0  0
 3  5  4  0
 0  2  3  6
 0  0  7  2
```
"""
function Base.setindex!(M::Union{BandMatrix, UpperBandMatrix, LowerBandMatrix}, val, idx::Int)


    # If linear index out of bounds, throw error.
    if idx > size(M, 1)^2
        throw(BoundsError())
    end
    
    # Convert to Cartesian index.
    cart_idx = CartesianIndices(size(M))[idx]
   
    # See method for resolving linear indexing for explanation.
    
    offset = cart_idx[2] - cart_idx[1]
    if string(Base.typename(typeof(M))) == "BandMatrix"
        if abs(offset) > div(length(M.diagonals), 2)
            throw(DomainError(idx, "only elements on diagonals can be set"))
        else
            M.diagonals[div(length(M.diagonals), 2)+1+offset][offset >= 0 ? cart_idx[1] : cart_idx[2]] = val
        end
    end
    if string(Base.typename(typeof(M))) == "UpperBandMatrix"
        if offset < 0 || offset > length(M.diagonals) - 1
            throw(DomainError(idx, "only elements on diagonals can be set"))
        else
            M.diagonals[offset+1][cart_idx[1]] = val
        end
    end
    if string(Base.typename(typeof(M))) == "LowerBandMatrix"
        if offset > 0 || -offset > length(M.diagonals) - 1
            throw(DomainError(idx, "only elements on diagonals can be set"))
        else
            M.diagonals[end+offset][cart_idx[2]] = val
        end
    end
end 


"""
    Base.setindex!(M::Union{BandMatrix, UpperBandMatrix, LowerBandMatrix}, val, idx::Vararg{Int, 2})

Set element at specified cartesian index in band matrix type instance.

# Examples
```julia-repl
julia> M = BandMatrix{Int64}([[3, 2, 7], [1, 5, 3, 2], [8, 4, 6]])
4×4 BandMatrix{Int64}:
 1  8  0  0
 3  5  4  0
 0  2  3  6
 0  0  7  2

julia> M[3, 2] = 9
9

julia> display(M)
4×4 BandMatrix{Int64}:
 1  8  0  0
 3  5  4  0
 0  9  3  6
 0  0  7  2
```
"""
function Base.setindex!(M::Union{BandMatrix, UpperBandMatrix, LowerBandMatrix}, val, idx::Vararg{Int, 2})
    
    # Get Cartesian index.
    cart_idx = CartesianIndex(idx[1], idx[2])

    # If index out of bounds, throw error.
    if any(Tuple(cart_idx) .> size(M, 1))
        throw(BoundsError())
    end
   
    # See method for resolving linear indexing for explanation.
    
    offset = cart_idx[2] - cart_idx[1]

    if string(Base.typename(typeof(M))) == "BandMatrix"
        if abs(offset) > div(length(M.diagonals), 2)
            throw(DomainError(idx, "only elements on diagonals can be set"))
        else
            M.diagonals[div(length(M.diagonals), 2)+1+offset][offset >= 0 ? cart_idx[1] : cart_idx[2]] = val
        end
    end
    if string(Base.typename(typeof(M))) == "UpperBandMatrix"
        if offset < 0 || offset > length(M.diagonals) - 1
            throw(DomainError(idx, "only elements on diagonals can be set"))
        else
            M.diagonals[offset+1][cart_idx[1]] = val
        end
    end
    if string(Base.typename(typeof(M))) == "LowerBandMatrix"
        if offset > 0 || -offset > length(M.diagonals) - 1
            throw(DomainError(idx, "only elements on diagonals can be set"))
        else
            M.diagonals[end+offset][cart_idx[2]] = val
        end
    end
end


"""
    *(M::BandMatrix, v::Vector)
    
Perform multiplication of center band matrix type instance with vector.

# Examples
```julia-repl
julia> M = BandMatrix{Int64}([[3, 2, 7], [1, 5, 3, 2], [8, 4, 6]])
4×4 BandMatrix{Int64}:
 1  8  0  0
 3  5  4  0
 0  2  3  6
 0  0  7  2

julia> v = [2, 1, 1, 2]
4-element Array{Int64,1}:
 2
 1
 1
 2

julia> M*v
4-element Array{Any,1}:
 10
 15
 17
 11
```
"""
function *(M::BandMatrix, v::Vector)

    # Inspect center diagonal to get matrix dimensions.
    center_idx = div(length(M.diagonals), 2) + 1
    mat_dim = length(M.diagonals[center_idx])

    # Dimensions match test.
    if length(v) != mat_dim
        throw(DimensionMismatch(@sprintf("Matrix A has dimensions %s, vector B has length %d", (mat_dim, mat_dim), length(v))))
    end

    # Allocate vector for storing results.
    res = Vector{}(undef, length(v))
    
    # Compute start index offset and ending index.
    offset = -div(length(M.diagonals), 2) + 1
    end_idx = div(length(M.diagonals), 2) + 1
     
    # Compute product by shifting indices over explicitly stored elements.
    for idx = 1:length(v)
        res[idx] = dot(M[idx, max(offset, 1):min(end_idx, mat_dim)], v[max(offset, 1):min(end_idx, mat_dim)]) 
        offset += 1
        end_idx += 1
    end

    # Return result.
    return res
end

# TODO: result of multiplication Any -> type of matrix.
"""
    *(M::UpperBandMatrix, v::Vector)
    
Perform multiplication of upper band matrix type instance with vector.

# Examples
```julia-repl
julia> M = UpperBandMatrix{Int64}([[1, 5, 3, 2], [8, 4, 6], [2, 1]])
4×4 UpperBandMatrix{Int64}:
 1  8  2  0
 0  5  4  1
 0  0  3  6
 0  0  0  2

julia> v = [2, 1, 2, 1]
4-element Array{Int64,1}:
 2
 1
 2
 1

julia> M*v
4-element Array{Any,1}:
 14
 14
 12
  2
```
"""
function *(M::UpperBandMatrix, v::Vector)
    
    # Dimensions match test.
    if length(v) != length(M.diagonals[1])
        throw(DimensionMismatch(@sprintf("Matrix A has dimensions %s, vector B has length %d", (length(M.diagonals[1]), length(M.diagonals[1])), length(v))))
    end
    
    # Allocate vector for storing results.
    res = Vector{}(undef, length(v))

    # Compute start index offset and ending index.
    offset = 1
    end_idx = length(M.diagonals)

    # Compute product by shifting indices over explicitly stored elements.
    for idx = 1:length(v)
        res[idx] = dot(M[idx, offset:min(end_idx, length(M.diagonals[1]))], v[offset:min(end_idx, length(M.diagonals[1]))]) 
        offset += 1
        end_idx += 1
    end

    # Return result.
    return res
end


"""
    *(M::LowerBandMatrix, v::Vector)
    
Perform multiplication of lower band matrix type instance with vector.

# Examples
```julia-repl
julia> M = LowerBandMatrix{Int64}([[2, 1], [3, 3, 1], [1, 5, 3, 2]])
4×4 LowerBandMatrix{Int64}:
 1  0  0  0
 3  5  0  0
 2  3  3  0
 0  1  1  2

julia> v = [1, 2, 1, 3]
4-element Array{Int64,1}:
 1
 2
 1
 3

julia> M*v
4-element Array{Any,1}:
  1
 13
 11
  9
```
"""
function *(M::LowerBandMatrix, v::Vector)
    
    # Dimensions match test.
    if length(v) != length(M.diagonals[end])
        throw(DimensionMismatch(@sprintf("Matrix A has dimensions %s, vector B has length %d", (length(M.diagonals[end]), length(M.diagonals[end])), length(v))))
    end
    
    # Allocate vector for storing results.
    res = Vector{}(undef, length(v))

    # Compute start index offset.
    offset = length(M.diagonals[1]) - length(M.diagonals[end]) + 1

    # Compute product by shifting indices over explicitly stored elements.
    for idx = 1:length(v)
        res[idx] = dot(M[idx, max(offset, 1):idx], v[max(offset, 1):idx]) 
        offset += 1
    end

    # Return result.
    return res
end


"""
    convert(::Type{BandMatrix{T}}, M::BandMatrix) where {T}

Convert type of values of matrix represented by center band matrix type.
"""
function convert(::Type{BandMatrix{T}}, M::BandMatrix) where {T}
    res = BandMatrix{T}(convert(Array{Array{T, 1},1}, M.diagonals))
    return res
end


"""
    convert(::Type{UpperBandMatrix{T}}, M::UpperBandMatrix) where {T}

Convert type of values of matrix represented by center band matrix type.
"""
function convert(::Type{UpperBandMatrix{T}}, M::UpperBandMatrix) where {T}
    res = UpperBandMatrix{T}(convert(Array{Array{T, 1},1}, M.diagonals))
    return res
end


"""
    convert(::Type{LowerBandMatrix{T}}, M::LowerBandMatrix) where {T}

Convert type of values of matrix represented by center band matrix type.
"""
function convert(::Type{LowerBandMatrix{T}}, M::LowerBandMatrix) where {T}
    res = LowerBandMatrix{T}(convert(Array{Array{T, 1},1}, M.diagonals))
    return res
end


"""
    function lu(M::BandMatrix)

Perform LU decomposition of center-band matrix type instance and return L and U matrices
(lower-band and upper-band respectively). The matrix must be diagonally dominant to ensure numerical
stability of the gaussian elimination process.

# Examples
```julia-repl
julia> M = BandMatrix{Int64}([[3, 2, 1], [5, 5, 7, 8], [1, 1, 3]])
4×4 BandMatrix{Int64}:
 5  1  0  0
 3  5  1  0
 0  2  7  3
 0  0  1  8

julia> l, u = lu(M);

julia> display(l)
4×4 LowerBandMatrix{Float64}:
 1.0  0         0         0
 0.6  1.0       0         0
 0    0.454545  1.0       0
 0    0         0.152778  1.0

julia> display(l)
4×4 LowerBandMatrix{Float64}:
 1.0  0         0         0
 0.6  1.0       0         0
 0    0.454545  1.0       0
 0    0         0.152778  1.0

julia> display(l*u)
4×4 Array{Float64,2}:
 5.0  1.0  0.0  0.0
 3.0  5.0  1.0  0.0
 0.0  2.0  7.0  3.0
 0.0  0.0  1.0  8.0
```
"""
function lu(M::BandMatrix)
    
    # If matrix values of signed integer type, convert to Float.
    if typeof(M).parameters[1] in subtypes(Signed)
        M_el = convert(BandMatrix{Float64}, M)
    else
        M_el = copy(M)
    end
    
    # Inspect center diagonal to get matrix dimensions.
    center_idx = div(length(M_el.diagonals), 2) + 1
    mat_dim = length(M_el.diagonals[center_idx])

    # Get explicitly stored columns range for elimination.
    col_range = div(length(M.diagonals), 2) + 1

    # Compute start index offset and ending index.
    offset = -div(length(M.diagonals), 2) + 1
    end_idx = div(length(M.diagonals), 2) + 1
    
    # Perform elimination.
    for idx_col = 1:mat_dim-1
        
        # perform diagonal dominance check for next row.
        if sum(M[idx_col, max(offset, 1):min(end_idx, mat_dim)]) - M[idx_col, idx_col] > M[idx_col, idx_col]
            throw(DomainError(M, "Matrix not diagonally dominant"))
        end
        offset += 1
        end_idx += 1
        
        # Eliminate elements in column below main diagonal. Build elimination matrix in-place.
        piv = M_el[idx_col, idx_col]
        for idx_row = idx_col+1:min(idx_col+div(length(M.diagonals), 2), mat_dim)
            div = -M_el[idx_row, idx_col]/piv
            M_el[idx_row, idx_col+1:min(idx_col+col_range, mat_dim)] += div*M_el[idx_col, idx_col+1:min(idx_col+col_range, mat_dim)]
            M_el[idx_row, idx_col] = -div
        end
    end

    # perform diagonal dominance check for last row.
    if sum(M[mat_dim, max(offset, 1):min(end_idx, mat_dim)]) - M[end, end] > M[end, end]
       throw(DomainError(M, "Matrix not diagonally dominant"))
    end

    # Get L and U matrices and return them.
    l = LowerBandMatrix{Float64}(vcat(M_el.diagonals[1:div(length(M_el.diagonals), 2)], [ones(mat_dim)]))
    u = UpperBandMatrix{Float64}(M_el.diagonals[center_idx:end])
    return l, u
end


"""
    function lu(M::UpperBandMatrix)

Perform LU decomposition of upper-band matrix type instance and return L and U matrices
(lower-band and upper-band respectively). The matrix must be diagonally dominant to ensure numerical
stability of the gaussian elimination process.

# Examples
```julia-repl
julia> M = UpperBandMatrix{Int64}([[7, 5, 3, 2], [4, 3, 2], [2, 1]])
4×4 UpperBandMatrix{Int64}:
 7  4  2  0
 0  5  3  1
 0  0  3  2
 0  0  0  2

julia> l, u = lu(M);

julia> display(l)
4×4 BandMatrix{Float64}:
 1.0  0    0    0
 0    1.0  0    0
 0    0    1.0  0
 0    0    0    1.0

julia> display(u)
4×4 UpperBandMatrix{Float64}:
 7.0  4.0  2.0  0
 0    5.0  3.0  1.0
 0    0    3.0  2.0
 0    0    0    2.0

julia> display(l*u)
4×4 Array{Float64,2}:
 7.0  4.0  2.0  0.0
 0.0  5.0  3.0  1.0
 0.0  0.0  3.0  2.0
 0.0  0.0  0.0  2.0
```
"""
# TODO center band -> center-band
function lu(M::UpperBandMatrix)

    # If matrix values of signed integer type, convert to Float.
    if typeof(M).parameters[1] in subtypes(Signed)
        M_el = convert(UpperBandMatrix{Float64}, M)
    else
        M_el = copy(M)
    end

    # Compute start index offset and ending index.
    offset = 1
    end_idx = length(M_el.diagonals)

    # Compute product by shifting indices over explicitly stored elements.
    for idx = 1:length(M_el.diagonals[1])

        # perform diagonal dominance check for next row.
        if sum(M_el[idx, offset:min(end_idx, length(M_el.diagonals[1]))]) - M_el[idx, idx] > M_el[idx, idx]
            throw(DomainError(M, "Matrix not diagonally dominant"))
        end
        offset += 1
        end_idx += 1
    end

    # Get L and U matrices and return them.
    l = BandMatrix{Float64}([ones(length(M_el.diagonals[1]))])
    u = M_el
    return l, u
end


"""
    function lu(M::LowerBandMatrix)

Perform LU decomposition of lower-band matrix type instance and return L and U matrices
(lower-band and upper-band respectively). The matrix must be diagonally dominant to ensure numerical
stability of the gaussian elimination process.

# Examples
```julia-repl
julia> M = LowerBandMatrix{Int64}([[2, 1], [3, 2, 1], [1, 5, 6, 3]])
4×4 LowerBandMatrix{Int64}:
 1  0  0  0
 3  5  0  0
 2  2  6  0
 0  1  1  3

julia> l, u = lu(M);

julia> display(l)
4×4 LowerBandMatrix{Float64}:
 1.0  0    0         0
 3.0  1.0  0         0
 2.0  0.4  1.0       0
 0    0.2  0.166667  1.0

julia> display(u)
d4×4 UpperBandMatrix{Float64}:
 1.0  0    0    0
 0    5.0  0    0
 0    0    6.0  0
 0    0    0    3.0

julia> display(l*u)
4×4 Array{Float64,2}:
 1.0  0.0  0.0  0.0
 3.0  5.0  0.0  0.0
 2.0  2.0  6.0  0.0
 0.0  1.0  1.0  3.0
```
"""
function lu(M::LowerBandMatrix)

    # If matrix values of signed integer type, convert to Float.
    if typeof(M).parameters[1] in subtypes(Signed)
        M_el = convert(LowerBandMatrix{Float64}, M)
    else
        M_el = copy(M)
    end
    
    # Inspect last diagonal to get matrix dimensions.
    mat_dim = length(M_el.diagonals[end])
    
    # Get explicitly stored columns range for elimination.
    col_range = div(length(M.diagonals), 2) + 1

    # Compute start index offset.
    offset = length(M.diagonals[1]) - length(M.diagonals[end]) + 1
    
    for idx_col = 1:mat_dim-1

        # perform diagonal dominance check for next row.
        if sum(M[idx_col, max(offset, 1):idx_col]) - M[idx_col, idx_col] > M[idx_col, idx_col]
            throw(DomainError(M, "Matrix not diagonally dominant"))
        end
        offset += 1

        # Eliminate elements in column below main diagonal. Build elimination matrix in-place.
        piv = M_el[idx_col, idx_col]
        for idx_row = idx_col+1:min(idx_col+length(M.diagonals)-1, mat_dim)
            div = -M_el[idx_row, idx_col]/piv
            M_el[idx_row, idx_col] = -div
        end
    end

    # perform diagonal dominance check for last row.
    if sum(M[mat_dim, max(offset, 1):mat_dim]) - M[end, end] > M[end, end]
        throw(DomainError(M, "Matrix not diagonally dominant"))
    end
    
    # Get L and U matrices and return them.
    l = LowerBandMatrix{Float64}(vcat(M_el.diagonals[1:end-1], [ones(mat_dim)]))
    u = UpperBandMatrix{Float64}([M_el.diagonals[end]])
    return l, u
end


"""
    \\(M::Union{BandMatrix, UpperBandMatrix, LowerBandMatrix}, b::Vector)

Solve systems of linear equations Ax = B for x where A is a band matrix.

# Examples
```julia-repl
julia> M = BandMatrix{Int64}([[2, 2], [4, 2, 2], [7, 9, 8, 7], [4, 3, 2], [2, 1]])
4×4 BandMatrix{Int64}:
 7  4  2  0
 4  9  3  1
 2  2  8  2
 0  2  2  7

julia> b = [3, 2, 4, 2]
4-element Array{Int64,1}:
 3
 2
 4
 2

julia> x = M\b
4-element Array{Float64,1}:
  0.37324602432179604
 -0.09260991580916746
  0.3788587464920487
  0.2039289055191768

julia> M*x
4-element Array{Any,1}:
 3.0
 2.0
 4.0
 2.0
```
"""
function \(M::Union{BandMatrix, UpperBandMatrix, LowerBandMatrix}, b::Vector)
    
    # Dimensions match test.
    if length(b) != maximum(map(length, M.diagonals))
        throw(DimensionMismatch(@sprintf("B has leading dimension %d, but needs %d", length(b), maximum(map(length, M.diagonals)))))
    end
    
    # If vector values of signed integer type, convert to Float.
    if typeof(b).parameters[1] in subtypes(Signed)
        b = convert(Array{Float64}, b)
    end
    
    # Perform LU decomposition.
    l, u = lu(M)    

    
    ### Solve Ly = b for y. ###
    
    # Initialize y vector.
    y = copy(b)

    # Set columns range for row when performing substitutions.
    col_range1 = length(l.diagonals) - 1

    # Perform forward substitions to compute y vector.
    y[1] = y[1]/l[1, 1]
    for idx = 2:length(y)
        start_idx = max(1, idx-col_range1)
        coeffs = l[idx, start_idx:idx-1]
        y[idx] = (y[idx] + dot(-coeffs, y[start_idx:idx-1]))/l[idx, idx]
    end


    ### Solve Ux = y for x. ###
    
    # Initialize x vector.
    x = copy(y)
    
    # Set columns range for row when performing substitutions.
    col_range2 = length(u.diagonals) - 1

    # Perform backward substitions to compute x vector.
    x[end] = x[end]/u[end, end]
    for idx = length(x)-1:-1:1
        end_idx = min(length(u.diagonals[1]), idx+col_range2)
        coeffs = u[idx, idx+1:end_idx]
        x[idx] = (x[idx] + dot(-coeffs, x[idx+1:end_idx]))/u[idx, idx]
    end
    
    # Return solution.
    return x
end

end
