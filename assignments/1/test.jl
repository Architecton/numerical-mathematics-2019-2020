include("./BandMatrices.jl")
using .BandMatrices
using Test


function extract_diagonals(M, offsets)
    
    # Get diagonals
    diags = Array{Array{typeof(M).parameters[1], 1}, 1}(undef, 0)

    for offset = offsets
        diag_nxt = Array{typeof(M).parameters[1]}(undef, size(M, 1) - abs(offset))

        if sign(offset) < 0
            idx_row = abs(offset) + 1
            idx_col = 1
        else
            idx_row = 1
            idx_col = offset + 1
        end

        for idx = 0:size(M, 1)-abs(offset)-1
            diag_nxt[idx+1] = M[idx_row+idx, idx_col+idx]
        end
        append!(diags, [diag_nxt])
    end
    return diags
end


function random_band_matrix(dim, kind, band_width)

    M = rand(dim, dim)
    if kind == "center"
        
        if band_width != 1 && div(band_width, 2) == 0
            throw(DomainError(band_width, "band width parameter must be odd"))
        end

        if band_width > dim + dim - 1
            throw(DomainError(band_width, "band width parameter too large"))
        end

        if band_width <= 0
            throw(DomainError(band_width, "band width parameter value cannot be less than 1"))
        end

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
        
        if band_width > dim
            throw(DomainError(band_width, "band width parameter too large"))
        end
        
        if band_width <= 0
            throw(DomainError(band_width, "band width parameter value cannot be less than 1"))
        end

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

        if band_width > dim
            throw(DomainError(band_width, "band width parameter too large"))
        end
        
        if band_width <= 0
            throw(DomainError(band_width, "band width parameter value cannot be less than 1"))
        end

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
        # TODO error.
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
end;

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
end;


@testset "Matrix indexing" begin

end;


@testset "Setting matrix elements" begin

end;


@testset "Matrix vector multiplication" begin

end;


@testset "LU decomposition" begin

end;


@testset "Backslash operator" begin

end;
