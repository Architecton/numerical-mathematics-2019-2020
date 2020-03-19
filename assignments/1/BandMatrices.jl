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

# TODO
function Base.copy(M::Union{BandMatrix, UpperBandMatrix, LowerBandMatrix})
    return typeof(M)(deepcopy(M.diagonals))
end


# TODO
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


# TODO
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


# TODO
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


# TODO
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


# TODO
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


# TODO
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


# TODO
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


# TODO
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


# TODO
function convert(::Type{BandMatrix{T}}, M::BandMatrix) where {T}
    res = BandMatrix{T}(convert(Array{Array{T, 1},1}, M.diagonals))
    return res
end


# TODO
function convert(::Type{UpperBandMatrix{T}}, M::UpperBandMatrix) where {T}
    res = UpperBandMatrix{T}(convert(Array{Array{T, 1},1}, M.diagonals))
    return res
end


# TODO
function convert(::Type{LowerBandMatrix{T}}, M::LowerBandMatrix) where {T}
    res = LowerBandMatrix{T}(convert(Array{Array{T, 1},1}, M.diagonals))
    return res
end


# TODO
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


# TODO
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


# TODO
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


# TODO
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
