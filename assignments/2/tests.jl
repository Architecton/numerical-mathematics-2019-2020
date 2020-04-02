include("./TridiagonalMatrices.jl")
using .TridiagonalMatrices
using Test

"""
    extract_tridiagonal_band(M)

Extract tridiagonal bands from specified matrix. 
The function returns diagonals represented as an 
array of arrays suitable for initializing tridiagonal 
matrix datatype.

# Examples
```julia-repl
TODO
```
"""
function extract_tridiagonal_band(M, kind)
    
    
    # Check validity of kind parameter.
    if kind ∉ ["center", "upper", "lower"]
        throw(DomainError(kind, "unknown type of tridiagonal matrix specified"))
    end
    
    # Initialize array for storing diagonals.
    diags = Array{Array{typeof(M).parameters[1], 1}, 1}(undef, 0)
    
    # Set offsets based on kind parameter.
    if kind == "center"
        offsets = [-1, 0, 1]
    elseif kind == "upper"
        offsets = [0, 1]
    elseif kind == "lower"
        offsets = [-1, 0]
    end
    
    # Go over diagonals' offsets.
    for offset = offsets

        # Initialize array for next diagonal.
        diag_nxt = Array{typeof(M).parameters[1]}(undef, size(M, 1) - abs(offset))
        
        # Determine side from which to take diagonal based on sign of offset.
        if sign(offset) < 0
            idx_row = abs(offset) + 1
            idx_col = 1
        else
            idx_row = 1
            idx_col = offset + 1
        end
        
        # Get next diagonal and add to list.
        for idx = 0:size(M, 1)-abs(offset)-1
            diag_nxt[idx+1] = M[idx_row+idx, idx_col+idx]
        end
        append!(diags, [diag_nxt])
    end

    # Return list of diagonals in format such that band matrices can be initialized.
    return diags
end


"""
    random_band_matrix(dim, kind, band_width, integers=false)

Generate a random tridiagonal matrix (not using the tridiagonal matrix datatype). The parameter 
integers specifies whether to fill the random matrix with integer values 
(from interval [1, 10]) or with floats from the interval [0, 1).

# Examples
```julia-repl
TODO
```
"""
function random_tridiagonal_matrix(dim, kind, integers=false)
    
    # Check validity of kind parameter.
    if kind ∉ ["center", "upper", "lower"]
        throw(DomainError(kind, "unknown type of tridiagonal matrix specified"))
    end
    
    # Fill with integers or floats?
    if !integers 
        M = rand(dim, dim)
    else
        M = rand(1:10, dim, dim)
    end
    
    # Mask-off elements not on diagonal band.
    start_row_b = 3
    col_range = 1
    for idx_row = start_row_b:dim
        for idx_col = 1:col_range
            M[idx_row, idx_col] = 0.0
            M[idx_col, idx_row] = 0.0
        end
        col_range += 1
    end
    
    # If matrix upper or lower diagonal, mask-off
    # appropriate diagonal band.
    if kind == "upper"
        for idx = 2:dim
            M[idx, idx-1] = 0.0
        end
    elseif kind == "lower"
        for idx = 2:dim
            M[idx-1, idx] = 0.0
        end
    end
    
    return M
end


### TEST SETS ###

@testset "Matrix initializations" begin
    for dim = 2:10
        mat = random_tridiagonal_matrix(dim, "center")
        diags = extract_tridiagonal_band(mat, "center")
        tdm = TridiagonalMatrix{typeof(mat).parameters[1]}(diags)
        @test Matrix(tdm) == mat
    end

    for dim = 2:10
        mat = random_tridiagonal_matrix(dim, "upper")
        diags = extract_tridiagonal_band(mat, "upper")
        utdm = UpperTridiagonalMatrix{typeof(mat).parameters[1]}(diags)
        @test Matrix(utdm) == mat
    end

    for dim = 2:10
        mat = random_tridiagonal_matrix(dim, "lower")
        diags = extract_tridiagonal_band(mat, "lower")
        ltdm = LowerTridiagonalMatrix{typeof(mat).parameters[1]}(diags)
        @test Matrix(ltdm) == mat
    end
end

@testset "Matrix size" begin
    for dim = 2:10
        mat = random_tridiagonal_matrix(dim, "center")
        diags = extract_tridiagonal_band(mat, "center")
        tdm = TridiagonalMatrix{typeof(mat).parameters[1]}(diags)
        @test size(mat) == size(tdm)
    end

    for dim = 2:10
        mat = random_tridiagonal_matrix(dim, "upper")
        diags = extract_tridiagonal_band(mat, "upper")
        utdm = UpperTridiagonalMatrix{typeof(mat).parameters[1]}(diags)
        @test size(mat) == size(utdm)
    end

    for dim = 2:10
        mat = random_tridiagonal_matrix(dim, "lower")
        diags = extract_tridiagonal_band(mat, "lower")
        ltdm = LowerTridiagonalMatrix{typeof(mat).parameters[1]}(diags)
        @test size(mat) == size(ltdm)
    end
end

@testset "Linear getindex" begin
   
    for dim = 2:10
        mat = random_tridiagonal_matrix(dim, "center")
        diags = extract_tridiagonal_band(mat, "center")
        tdm = TridiagonalMatrix{typeof(mat).parameters[1]}(diags)
        for lin_idx = 1:prod(size(tdm))
            @test tdm[lin_idx]  == mat[lin_idx]
        end
    end
    
    for dim = 2:10
        mat = random_tridiagonal_matrix(dim, "upper")
        diags = extract_tridiagonal_band(mat, "upper")
        utdm = UpperTridiagonalMatrix{typeof(mat).parameters[1]}(diags)
        for lin_idx = 1:prod(size(utdm))
            @test utdm[lin_idx]  == mat[lin_idx]
        end
    end
    
    for dim = 2:10
        mat = random_tridiagonal_matrix(dim, "lower")
        diags = extract_tridiagonal_band(mat, "lower")
        ltdm = LowerTridiagonalMatrix{typeof(mat).parameters[1]}(diags)
        for lin_idx = 1:prod(size(ltdm))
            @test ltdm[lin_idx]  == mat[lin_idx]
        end
    end
end

@testset "Cartesian getindex" begin

    for dim = 2:10
        mat = random_tridiagonal_matrix(dim, "center")
        diags = extract_tridiagonal_band(mat, "center")
        tdm = TridiagonalMatrix{typeof(mat).parameters[1]}(diags)
        for idx1 = 1:size(tdm, 1)
            for idx2 = 1:size(tdm, 2)
                @test tdm[idx1, idx2] == mat[idx1, idx2]
            end
        end
    end

    for dim = 2:10
        mat = random_tridiagonal_matrix(dim, "upper")
        diags = extract_tridiagonal_band(mat, "upper")
        utdm = UpperTridiagonalMatrix{typeof(mat).parameters[1]}(diags)
        for idx1 = 1:size(utdm, 1)
            for idx2 = 1:size(utdm, 2)
                @test utdm[idx1, idx2] == mat[idx1, idx2]
            end
        end
    end

    for dim = 2:10
        mat = random_tridiagonal_matrix(dim, "lower")
        diags = extract_tridiagonal_band(mat, "lower")
        ltdm = LowerTridiagonalMatrix{typeof(mat).parameters[1]}(diags)
        for idx1 = 1:size(ltdm, 1)
            for idx2 = 1:size(ltdm, 2)
                @test ltdm[idx1, idx2] == mat[idx1, idx2]
            end
        end
    end
end

@testset "Linear setindex" begin
    
    for dim = 2:10
        mat = random_tridiagonal_matrix(dim, "center")
        diags = extract_tridiagonal_band(mat, "center")
        tdm = TridiagonalMatrix{typeof(mat).parameters[1]}(diags)
        for lin_idx = 1:prod(size(tdm))
            cart_idx = CartesianIndices(size(tdm))[lin_idx]
            val = rand()
            if abs(cart_idx[2] - cart_idx[1]) > 1
                @test_throws DomainError tdm[lin_idx] = val
            else
                tdm[lin_idx] = val
                mat[lin_idx] = val
                @test tdm[lin_idx] == mat[lin_idx]
            end
        end
    end

    for dim = 2:10
        mat = random_tridiagonal_matrix(dim, "upper")
        diags = extract_tridiagonal_band(mat, "upper")
        utdm = UpperTridiagonalMatrix{typeof(mat).parameters[1]}(diags)
        for lin_idx = 1:prod(size(utdm))
            cart_idx = CartesianIndices(size(utdm))[lin_idx]
            val = rand()
            if abs(cart_idx[2] - cart_idx[1]) > 1 || cart_idx[2] - cart_idx[1] < 0
                @test_throws DomainError utdm[lin_idx] = val
            else
                utdm[lin_idx] = val
                mat[lin_idx] = val
                @test utdm[lin_idx] == mat[lin_idx]
            end
        end
    end

    for dim = 2:10
        mat = random_tridiagonal_matrix(dim, "lower")
        diags = extract_tridiagonal_band(mat, "lower")
        ltdm = LowerTridiagonalMatrix{typeof(mat).parameters[1]}(diags)
        for lin_idx = 1:prod(size(ltdm))
            cart_idx = CartesianIndices(size(ltdm))[lin_idx]
            val = rand()
            if abs(cart_idx[2] - cart_idx[1]) > 1 || cart_idx[2] - cart_idx[1] > 0
                @test_throws DomainError ltdm[lin_idx] = val
            else
                ltdm[lin_idx] = val
                mat[lin_idx] = val
                @test ltdm[lin_idx] == mat[lin_idx]
            end
        end
    end
end


@testset "Cartesian setindex" begin
    
    for dim = 2:10
        mat = random_tridiagonal_matrix(dim, "center")
        diags = extract_tridiagonal_band(mat, "center")
        tdm = TridiagonalMatrix{typeof(mat).parameters[1]}(diags)
        for idx1 = 1:size(tdm, 1)
            for idx2 = 1:size(tdm, 2)
                val = rand()
                if abs(idx2 - idx1) > 1
                    @test_throws DomainError tdm[idx1, idx2] = val
                else
                    tdm[idx1, idx2] = val
                    mat[idx1, idx2] = val
                    @test tdm[idx1, idx2] == mat[idx1, idx2]
                end
            end
        end
    end

    for dim = 2:10
        mat = random_tridiagonal_matrix(dim, "upper")
        diags = extract_tridiagonal_band(mat, "upper")
        utdm = UpperTridiagonalMatrix{typeof(mat).parameters[1]}(diags)
        for idx1 = 1:size(utdm, 1)
            for idx2 = 1:size(utdm, 2)
                val = rand()
                if abs(idx2 - idx1) > 1 || idx2 - idx1 < 0
                    @test_throws DomainError utdm[idx1, idx2] = val
                else
                    utdm[idx1, idx2] = val
                    mat[idx1, idx2] = val
                    @test utdm[idx1, idx2] == mat[idx1, idx2]
                end
            end
        end
    end

    for dim = 2:10
        mat = random_tridiagonal_matrix(dim, "lower")
        diags = extract_tridiagonal_band(mat, "lower")
        ltdm = LowerTridiagonalMatrix{typeof(mat).parameters[1]}(diags)
        for idx1 = 1:size(ltdm, 1)
            for idx2 = 1:size(ltdm, 2)
                val = rand()
                if abs(idx2 - idx1) > 1 || idx2 - idx1 > 0
                    @test_throws DomainError ltdm[idx1, idx2] = val
                else
                    ltdm[idx1, idx2] = val
                    mat[idx1, idx2] = val
                    @test ltdm[idx1, idx2] == mat[idx1, idx2]
                end
            end
        end
    end
end


@testset "Matrix vector multiplication" begin
    
    for dim = 2:10
        mat = random_tridiagonal_matrix(dim, "center")
        diags = extract_tridiagonal_band(mat, "center")
        tdm = TridiagonalMatrix{typeof(mat).parameters[1]}(diags)
        v = rand(dim)
        @test tdm*v ≈ mat*v atol=1.0e-4
    end

    for dim = 2:10
        mat = random_tridiagonal_matrix(dim, "upper")
        diags = extract_tridiagonal_band(mat, "upper")
        utdm = UpperTridiagonalMatrix{typeof(mat).parameters[1]}(diags)
        v = rand(dim)
        @test utdm*v ≈ mat*v atol=1.0e-4
    end

    for dim = 2:10
        mat = random_tridiagonal_matrix(dim, "lower")
        diags = extract_tridiagonal_band(mat, "lower")
        ltdm = LowerTridiagonalMatrix{typeof(mat).parameters[1]}(diags)
        v = rand(dim)
        @test ltdm*v ≈ mat*v atol=1.0e-4
    end
end



# @testset "LU decomposition" begin
# # TODO
# end
# 
# @testset "Backslash operator" begin
# # TODO
# end

