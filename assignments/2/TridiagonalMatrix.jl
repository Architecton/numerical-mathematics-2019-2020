import Base:*,\,convert,copy,size,getindex,setindex


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


function convert(::Type{TridiagonalMatrix{T}}, M::TridiagonalMatrix) where {T}

end


function Base.size(M::TridiagonalMatrix)::Tuple{Int64,Int64}
    return (length(M.diagonals[2]), length(M.diagonals[2]))
end


function Base.getindex(M::TridiagonalMatrix, idx::Int)

end


function Base.getindex(M::TridiagonalMatrix, idx::Vararg{Int, 2})

end


function Base.setindex!(M::TridiagonalMatrix, val, idx::Int)

end


function Base.setindex!(M::TridiagonalMatrix, val, idx::Vararg{Int, 2})

end


function *(M::TridiagonalMatrix, v::Vector)

end


function lu(M::TridiagonalMatrix)

end
