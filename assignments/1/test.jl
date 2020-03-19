include("./BandMatrices.jl")
using .BandMatrices
using Test

"""
    extract_diagonals(M, offsets)

Extract diagonals with offsets from main diagonals specified as a
list of offsets. The function returns diagonals represented as an
array of arrays suitable initializing band matrix datatypes.

# Examples
```julia-repl
julia> M = rand(4, 4)
4×4 Array{Float64,2}:
 0.664013  0.636112  0.679132  0.795837
 0.238675  0.715638  0.319589  0.568116
 0.108106  0.842201  0.621499  0.748234
 0.807353  0.52464   0.971043  0.880246

julia> diags = extract_diagonals(M, [-1, 0, 2])
3-element Array{Array{Float64,1},1}:
 [0.238675, 0.842201, 0.971043]          
 [0.664013, 0.715638, 0.621499, 0.880246]
 [0.679132, 0.568116]    
```
"""
function extract_diagonals(M, offsets)
    
    # Initialize array for storing diagonals.
    diags = Array{Array{typeof(M).parameters[1], 1}, 1}(undef, 0)
    
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

Generate a random band matrix (not using the band matrix datatype). The parameter
kind can take on the values 'center', 'upper' or 'lower' and is used to specify the
type of band matrix to generate. The parameter band_width specifies the number of diagonals
in the band matrix. The parameter integers specifies whether to fill the random matrix with
integer values (from interval [1, 10]) or with floats from the interval [0, 1).

# Examples
```julia-repl
julia> M = random_band_matrix(4, "center", 3, true)
4×4 Array{Int64,2}:
 7   2  0   0
 9  10  2   0
 0   2  8   6
 0   0  6  10
```
"""
function random_band_matrix(dim, kind, band_width, integers=false)
    
    # Fill with integers or floats?
    if !integers 
        M = rand(dim, dim)
    else
        M = rand(1:10, dim, dim)
    end

    # If creating a random center band matrix.
    if kind == "center"
        
        # Check parameters.
        if band_width != 1 && div(band_width, 2) == 0
            throw(DomainError(band_width, "band width parameter must be odd"))
        end

        if band_width > dim + dim - 1
            throw(DomainError(band_width, "band width parameter too large"))
        end

        if band_width <= 0
            throw(DomainError(band_width, "band width parameter value cannot be less than 1"))
        end
        
        # Mask off of-diagonal elements and return matrix.
        start_row_b = div(band_width, 2) + 2
        col_range = 1
        for idx_row = start_row_b:dim
            for idx_col = 1:col_range
                M[idx_row, idx_col] = 0.0
                M[idx_col, idx_row] = 0.0
            end
            col_range += 1
        end

        return M

    elseif kind == "upper"
        # If creating a random upper band matrix.
        
        # Check parameters.
        if band_width > dim
            throw(DomainError(band_width, "band width parameter too large"))
        end
        
        if band_width <= 0
            throw(DomainError(band_width, "band width parameter value cannot be less than 1"))
        end

        # Mask off of-diagonal elements and return matrix.
        start_col = band_width + 1
        row_range = 1
        for idx_col = start_col:dim
            for idx_row = 1:row_range
                M[idx_row, idx_col] = 0.0
            end
            row_range += 1
        end

        col_range = 1
        for row_idx = 2:dim
            M[row_idx, 1:col_range] .= 0
            col_range += 1
        end

        return M

    elseif kind == "lower"
        # If creating a random lower band matrix.
        
        # Check parameters.
        if band_width > dim
            throw(DomainError(band_width, "band width parameter too large"))
        end
        
        if band_width <= 0
            throw(DomainError(band_width, "band width parameter value cannot be less than 1"))
        end

        # Mask off of-diagonal elements and return matrix.
        start_row_b = div(band_width, 2) + 2
        start_row_b = band_width + 1
        col_range = 1
        for idx_row = start_row_b:dim
            for idx_col = 1:col_range
                M[idx_row, idx_col] = 0.0
            end
            col_range += 1
        end
        
        col_start = 2
        for row_idx = 1:dim-1
            M[row_idx, col_start:end] .= 0
            col_start += 1
        end

        return M

    else
        throw(DomainError(kind, "unknown type of band matrix specified"))
    end
end

@testset "Matrix initializations" begin

    # Test center band matrix initializations.
    for dim = 2:10
        for width = 1:2:dim+dim-1
            mat = random_band_matrix(dim, "center", width)
            offset_lim = div(width, 2)
            diags = extract_diagonals(mat, collect(-offset_lim:offset_lim))
            bm = BandMatrix{typeof(mat).parameters[1]}(diags)
            @test Matrix(bm) == mat
        end
    end

    # Test upper band matrix initializations.
    for dim = 2:10
        for width = 1:dim
            mat = random_band_matrix(dim, "upper", width)
            diags = extract_diagonals(mat, collect(0:width-1))
            bm = UpperBandMatrix{typeof(mat).parameters[1]}(diags)
            @test Matrix(bm) == mat
        end
    end

    # Test lower band matrix initializations.
    for dim = 2:10
        for width = 1:dim
            mat = random_band_matrix(dim, "lower", width)
            diags = extract_diagonals(mat, collect(-width+1:0))
            bm = LowerBandMatrix{typeof(mat).parameters[1]}(diags)
            @test Matrix(bm) == mat
        end
    end
end

@testset "Matrix sizes" begin

    # Test center band matrix size.
    for dim = 2:10
        for width = 1:2:dim+dim-1
            mat = random_band_matrix(dim, "center", width)
            offset_lim = div(width, 2)
            diags = extract_diagonals(mat, collect(-offset_lim:offset_lim))
            bm = BandMatrix{typeof(mat).parameters[1]}(diags)
            @test size(bm) == size(Matrix(bm))
        end
    end

    # Test upper band matrix size.
    for dim = 2:10
        for width = 1:dim
            mat = random_band_matrix(dim, "upper", width)
            diags = extract_diagonals(mat, collect(0:width-1))
            bm = UpperBandMatrix{typeof(mat).parameters[1]}(diags)
            @test size(bm) == size(Matrix(bm))
        end
    end

    # Test lower band matrix size.
    for dim = 2:10
        for width = 1:dim
            mat = random_band_matrix(dim, "lower", width)
            diags = extract_diagonals(mat, collect(-width+1:0))
            bm = LowerBandMatrix{typeof(mat).parameters[1]}(diags)
            @test size(bm) == size(Matrix(bm))
        end
    end
end


@testset "Linear matrix indexing" begin
    
    # Test center band matrix linear indexing.
    for dim = 2:10
        for width = 1:2:dim+dim-1
            mat = random_band_matrix(dim, "center", width)
            offset_lim = div(width, 2)
            diags = extract_diagonals(mat, collect(-offset_lim:offset_lim))
            bm = BandMatrix{typeof(mat).parameters[1]}(diags)
            for idx = 1:prod(size(bm))
                @test bm[idx] == Matrix(bm)[idx]
            end
        end
    end

    # Test upper band matrix linear indexing.
    for dim = 2:10
        for width = 1:dim
            mat = random_band_matrix(dim, "upper", width)
            diags = extract_diagonals(mat, collect(0:width-1))
            bm = UpperBandMatrix{typeof(mat).parameters[1]}(diags)
            for idx = 1:prod(size(bm))
                @test bm[idx] == Matrix(bm)[idx]
            end
        end
    end

    # Test lower band matrix linear indexing.
    for dim = 2:10
        for width = 1:dim
            mat = random_band_matrix(dim, "lower", width)
            diags = extract_diagonals(mat, collect(-width+1:0))
            bm = LowerBandMatrix{typeof(mat).parameters[1]}(diags)
            for idx = 1:prod(size(bm))
                @test bm[idx] == Matrix(bm)[idx]
            end
        end
    end
end


@testset "Cartesian matrix indexing" begin
 
    # Test center band matrix cartesian indexing.
    for dim = 2:10
        for width = 1:2:dim+dim-1
            mat = random_band_matrix(dim, "center", width)
            offset_lim = div(width, 2)
            diags = extract_diagonals(mat, collect(-offset_lim:offset_lim))
            bm = BandMatrix{typeof(mat).parameters[1]}(diags)
            for idx1 = 1:size(bm, 1)
                for idx2 = 1:size(bm, 2)
                    @test bm[idx1, idx2] == Matrix(bm)[idx1, idx2]
                end
            end
        end
    end

    # Test upper band matrix cartesian indexing.
    for dim = 2:10
        for width = 1:dim
            mat = random_band_matrix(dim, "upper", width)
            diags = extract_diagonals(mat, collect(0:width-1))
            bm = UpperBandMatrix{typeof(mat).parameters[1]}(diags)
            for idx1 = 1:size(bm, 1)
                for idx2 = 1:size(bm, 2)
                    @test bm[idx1, idx2] == Matrix(bm)[idx1, idx2]
                end
            end
        end
    end

    # Test lower band matrix cartesian indexing.
    for dim = 2:10
        for width = 1:dim
            mat = random_band_matrix(dim, "lower", width)
            diags = extract_diagonals(mat, collect(-width+1:0))
            bm = LowerBandMatrix{typeof(mat).parameters[1]}(diags)
            for idx1 = 1:size(bm, 1)
                for idx2 = 1:size(bm, 2)
                    @test bm[idx1, idx2] == Matrix(bm)[idx1, idx2]
                end
            end
        end
    end
end


@testset "Matrix vector multiplication" begin

    # Test center band matrix vector multiplication.
    for dim = 2:10
        for width = 1:2:dim+dim-1
            mat = random_band_matrix(dim, "center", width)
            offset_lim = div(width, 2)
            diags = extract_diagonals(mat, collect(-offset_lim:offset_lim))
            bm = BandMatrix{typeof(mat).parameters[1]}(diags)
            b = rand(dim)
            @test bm*b ≈ Matrix(bm)*b atol=1.0e-4
        end
    end

    # Test upper band matrix vector multiplication.
    for dim = 2:10
        for width = 1:dim
            mat = random_band_matrix(dim, "upper", width)
            diags = extract_diagonals(mat, collect(0:width-1))
            bm = UpperBandMatrix{typeof(mat).parameters[1]}(diags)
            b = rand(dim)
            @test bm*b ≈ Matrix(bm)*b atol=1.0e-4
        end
    end

    # Test lower band matrix vector multiplication.
    for dim = 2:10
        for width = 1:dim
            mat = random_band_matrix(dim, "lower", width)
            diags = extract_diagonals(mat, collect(-width+1:0))
            bm = LowerBandMatrix{typeof(mat).parameters[1]}(diags)
            b = rand(dim)
            @test bm*b ≈ Matrix(bm)*b atol=1.0e-4
        end
    end
end


@testset "LU decomposition" begin
    
    # Test center band matrix LU decomposition.
    for dim = 2:10
        for width = 1:2:dim+dim-1
            mat = random_band_matrix(dim, "center", width, true)
            offset_lim = div(width, 2)
            diags = extract_diagonals(mat, collect(-offset_lim:offset_lim))
            bm = BandMatrix{typeof(mat).parameters[1]}(diags)
            
            center_idx = div(length(bm.diagonals), 2) + 1
            bm.diagonals[center_idx] .+= (dim-1)*10

            l,u = lu(bm)
            l_ref, u_ref = lu(Matrix(bm))
            @test l ≈ l_ref atol=1.0e-4
            @test u ≈ u_ref atol=1.0e-4
        end
    end

    # Test upper band matrix LU decomposition.
    for dim = 2:10
        for width = 1:dim
            mat = random_band_matrix(dim, "upper", width, true)
            diags = extract_diagonals(mat, collect(0:width-1))
            bm = UpperBandMatrix{typeof(mat).parameters[1]}(diags)
            
            bm.diagonals[1] .+= (dim-1)*10

            l,u = lu(bm)
            l_ref, u_ref = lu(Matrix(bm))
            @test l ≈ l_ref atol=1.0e-4
            @test u ≈ u_ref atol=1.0e-4

        end
    end

    # Test lower band matrix LU decomposition.
    for dim = 2:10
        for width = 1:dim
            mat = random_band_matrix(dim, "lower", width, true)
            diags = extract_diagonals(mat, collect(-width+1:0))
            bm = LowerBandMatrix{typeof(mat).parameters[1]}(diags)
            
            bm.diagonals[end] .+= (dim-1)*10

            l,u = lu(bm)
            l_ref, u_ref = lu(Matrix(bm))
            @test l ≈ l_ref atol=1.0e-4
            @test u ≈ u_ref atol=1.0e-4

        end
    end

end


@testset "Backslash operator" begin
    
    # Test center band matrix backslash operator.
    for dim = 2:10
        for width = 1:2:dim+dim-1
            mat = random_band_matrix(dim, "center", width, true)
            offset_lim = div(width, 2)
            diags = extract_diagonals(mat, collect(-offset_lim:offset_lim))
            bm = BandMatrix{typeof(mat).parameters[1]}(diags)
            
            # Ensure diagonal dominance.
            center_idx = div(length(bm.diagonals), 2) + 1
            bm.diagonals[center_idx] .+= (dim-1)*10

            # Create random system of linear equations and solve.
            b = rand(1:10, dim)
            x = bm\b
            
            @test bm*x ≈ b atol=1.0e-2
        end
    end

    # Test upper band matrix backslash operator.
    for dim = 2:10
        for width = 1:dim
            mat = random_band_matrix(dim, "upper", width, true)
            diags = extract_diagonals(mat, collect(0:width-1))
            bm = UpperBandMatrix{typeof(mat).parameters[1]}(diags)
            
            # Ensure diagonal dominance.
            bm.diagonals[1] .+= (dim-1)*10

            # Create random system of linear equations and solve.
            b = rand(1:10, dim)
            x = bm\b
            @test bm*x ≈ b atol=1.0e-2
            
        end
    end

    # Test lower band matrix backslash operator.
    for dim = 2:10
        for width = 1:dim
            mat = random_band_matrix(dim, "lower", width, true)
            diags = extract_diagonals(mat, collect(-width+1:0))
            bm = LowerBandMatrix{typeof(mat).parameters[1]}(diags)
            
            # Ensure diagonal dominance.
            bm.diagonals[end] .+= (dim-1)*10
            
            # Create random system of linear equations and solve.
            b = rand(1:10, dim)
            x = bm\b

            @test bm*x ≈ b atol=1.0e-2
        end
    end

end

