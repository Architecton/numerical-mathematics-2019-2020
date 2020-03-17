import Base:*,\,convert
import LinearAlgebra:dot,lu


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


function *(M::BandMatrix, v::Vector)
    # TODO bounds check
    #

    # Allocate vector for storing results.
    res = Vector{}(undef, length(v))
    
    # Inspect center diagonal to get matrix dimensions.
    center_idx = div(length(M.diagonals), 2) + 1
    mat_dim = length(M.diagonals[center_idx])
    
    # Compute start index offset and ending index.
    offset = -div(length(M.diagonals), 2) + 1
    end_idx = div(length(M.diagonals), 2) + 1
     
    # Compute product by shifting indices over explicitly stored elements.
    for idx = 1:length(v)
        res[idx] = LinearAlgebra.dot(M[idx, max(offset, 1):min(end_idx, mat_dim)], v[max(offset, 1):min(end_idx, mat_dim)]) 
        offset += 1
        end_idx += 1
    end

    # Return result.
    return res
end


function *(M::UpperBandMatrix, v::Vector)
    
    # Allocate vector for storing results.
    res = Vector{}(undef, length(v))

    # Compute start index offset and ending index.
    offset = 1
    end_idx = length(M.diagonals)

    # Compute product by shifting indices over explicitly stored elements.
    for idx = 1:length(v)
        res[idx] = LinearAlgebra.dot(M[idx, offset:min(end_idx, length(M.diagonals[1]))], v[offset:min(end_idx, length(M.diagonals[1]))]) 
        offset += 1
        end_idx += 1
    end

    # Return result.
    return res
end


function *(M::LowerBandMatrix, v::Vector)
    
    # Allocate vector for storing results.
    res = Vector{}(undef, length(v))

    # Compute start index offset.
    offset = length(M.diagonals[1]) - length(M.diagonals[end]) + 1

    # Compute product by shifting indices over explicitly stored elements.
    for idx = 1:length(v)
        res[idx] = LinearAlgebra.dot(M[idx, max(offset, 1):idx], v[max(offset, 1):idx]) 
        offset += 1
    end

    # Return result.
    return res
end


function convert(::Type{BandMatrix{T}}, M::BandMatrix) where {T}
    res = BandMatrix{T}(convert(Array{Array{T, 1},1}, M.diagonals))
    return res
end

function convert(::Type{UpperBandMatrix{T}}, M::UpperBandMatrix) where {T}
    res = UpperBandMatrix{T}(convert(Array{Array{T, 1},1}, M.diagonals))
    return res
end

function convert(::Type{LowerBandMatrix{T}}, M::LowerBandMatrix) where {T}
    res = LowerBandMatrix{T}(convert(Array{Array{T, 1},1}, M.diagonals))
    return res
end


function \(M::BandMatrix, v::Vector)
    
end


function \(M::UpperBandMatrix, v::Vector)

end


function \(M::LowerBandMatrix, v::Vector)

end

function lu(M::BandMatrix)
    
    if typeof(M).parameters[1] == Int64
        M_el = convert(BandMatrix{Float64}, M)
    else
        M_el = M  
    end
    
    # Inspect center diagonal to get matrix dimensions.
    center_idx = div(length(M_el.diagonals), 2) + 1
    mat_dim = length(M_el.diagonals[center_idx])
    col_range = div(length(M.diagonals), 2) + 1

    # Compute start index offset and ending index.
    offset = -div(length(M.diagonals), 2) + 1
    end_idx = div(length(M.diagonals), 2) + 1
    
    for idx_col = 1:mat_dim-1
        
        if sum(M[idx_col, max(offset, 1):min(end_idx, mat_dim)]) - M[idx_col, idx_col] > M[idx_col, idx_col]
            # TODO error in dominance.
        end
        offset += 1
        end_idx += 1

        piv = M_el[idx_col, idx_col]
        for idx_row = idx_col+1:min(idx_col+div(length(M.diagonals), 2), mat_dim)
            div = -M_el[idx_row, idx_col]/piv
            M_el[idx_row, idx_col+1:min(idx_col+col_range, mat_dim)] += div*M_el[idx_col, idx_col+1:min(idx_col+col_range, mat_dim)]
            M_el[idx_row, idx_col] = -div
        end
    end

    
    if sum(M[mat_dim, max(offset, 1):min(end_idx, mat_dim)]) - M[end, end] > M[end, end]
       # Error, not diagonally dominant. 
    end

    l = LowerBandMatrix{Float64}(vcat(M_el.diagonals[1:div(length(M_el.diagonals), 2)], [ones(mat_dim)]))
    u = UpperBandMatrix{Float64}(M_el.diagonals[center_idx:end])
    return l, u
end


function lu(M::UpperBandMatrix)

    # Compute start index offset and ending index.
    offset = 1
    end_idx = length(M.diagonals)

    # Compute product by shifting indices over explicitly stored elements.
    for idx = 1:length(M.diagonals[1])
        if sum(M[idx, offset:min(end_idx, length(M.diagonals[1]))]) - M[idx, idx] > M[idx, idx]
            # TODO error in dominance.
        end
        offset += 1
        end_idx += 1
    end

    l = BandMatrix{Float64}([ones(length(M.diagonals[1]))])
    u = convert(UpperBandMatrix{Float64}, M)
    return l, u
end


function lu(M::LowerBandMatrix)

    if typeof(M).parameters[1] == Int64
        M_el = convert(LowerBandMatrix{Float64}, M)
    else
        M_el = M  
    end
    
    mat_dim = length(M_el.diagonals[end])
    col_range = div(length(M.diagonals), 2) + 1

    # Compute start index offset.
    offset = length(M.diagonals[1]) - length(M.diagonals[end]) + 1
    
    for idx_col = 1:mat_dim-1

        if sum(M[idx_col, max(offset, 1):idx_col]) - M[idx_col, idx_col] > M[idx_col, idx_col]
            # TODO error in dominance.
        end
        offset += 1

        piv = M_el[idx_col, idx_col]
        for idx_row = idx_col+1:min(idx_col+length(M.diagonals)-1, mat_dim)
            div = -M_el[idx_row, idx_col]/piv
            M_el[idx_row, idx_col] = -div
        end
    end

    if sum(M[mat_dim, max(offset, 1):mat_dim]) - M[end, end] > M[end, end]
        # TODO error in dominance.
    end

    l = LowerBandMatrix{Float64}(vcat(M_el.diagonals[1:end-1], [ones(mat_dim)]))
    u = UpperBandMatrix{Float64}([M_el.diagonals[end]])
    return l, u
end


bm = BandMatrix{Int64}([[1, 2], [1, 2, 3], [3, 2]])
ubm = UpperBandMatrix{Int64}([[1, 2, 3], [3, 2]])
lbm = LowerBandMatrix{Int64}([[1, 2], [1, 2, 3]])

bm2 = BandMatrix{Int64}([[1, 2, 3, 4], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6], [1, 2, 3, 3, 4], [1, 2, 3, 4]])
lbm2 = LowerBandMatrix{Int64}([[7, 8, 9], [4, 5, 6, 7], [1, 2, 2, 4, 5]])
bm3 = BandMatrix{Int64}([[4, 5, 8], [3, 4, 6, 7], [1, 3, 5, 6, 7], [2, 5, 7, 8], [7, 8, 9]])
bm4 = BandMatrix{Int64}([[2, 3, 4], [1, 1, 1, 1], [3, 2, 3]])
lbm5 = LowerBandMatrix{Int64}([[3, 3, 3], [2, 2, 2, 2], [1, 2, 3, 4, 5]])

