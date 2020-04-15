include("./TridiagonalMatrices.jl")
using .TridiagonalMatrices
using LinearAlgebra
using Test


"""
    extract_tridiagonal_band(M, kind::String)

Extract tridiagonal bands from specified matrix. 
The function returns diagonals represented as an 
array of arrays suitable for initializing tridiagonal 
matrix datatype.

# Examples
```julia-repl
julia> M = [1 2 0 0; 2 3 2 0; 0 4 7 3; 0 0 5 6]
4×4 Array{Int64,2}:
 1  2  0  0
 2  3  2  0
 0  4  7  3
 0  0  5  6

julia> extract_tridiagonal_band(M, "center")
3-element Array{Array{Int64,1},1}:
 [2, 4, 5]
 [1, 3, 7, 6]
 [2, 2, 3]

julia> extract_tridiagonal_band(M, "upper")
2-element Array{Array{Int64,1},1}:
 [1, 3, 7, 6]
 [2, 2, 3]

julia> extract_tridiagonal_band(M, "lower")
2-element Array{Array{Int64,1},1}:
 [2, 4, 5]
 [1, 3, 7, 6]
```
"""
function extract_tridiagonal_band(M, kind::String)
    
    
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
    random_tridiagonal_matrix(dim::Int, kind::String, integers::Bool=false)

Generate a random tridiagonal matrix (not using the tridiagonal matrix datatype). The parameter 
integers specifies whether to fill the random matrix with integer values 
(from interval [1, 10]) or with floats from the interval [0, 1).

# Examples
```julia-repl
julia> random_tridiagonal_matrix(5, "center")
5×5 Array{Float64,2}:
 0.417417  0.644575  0.0       0.0       0.0
 0.480585  0.824313  0.676599  0.0       0.0
 0.0       0.805721  0.585806  0.792382  0.0
 0.0       0.0       0.460766  0.197488  0.268444
 0.0       0.0       0.0       0.861061  0.827607

julia> random_tridiagonal_matrix(3, "upper", true)
3×3 Array{Int64,2}:
 7  4  0
 0  5  6
 0  0  1

julia> random_tridiagonal_matrix(4, "lower")
4×4 Array{Float64,2}:
 0.240992  0.0       0.0       0.0
 0.135737  0.88936   0.0       0.0
 0.0       0.769493  0.198417  0.0
 0.0       0.0       0.343407  0.0261404
```
"""
function random_tridiagonal_matrix(dim::Int, kind::String, integers::Bool=false)
    
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
MAX_DIM = 20

@testset "Generating random tridiagonal matrices (test for tests)" begin

    ### CENTER-BAND TRIDIAGONAL MATRICES
    mat1 = [7 10 0 0; 
            4 8 4 0; 
            0 2 8 9; 
            0 0 7 2]
    mat2 = [8 3 0 0 0 0; 
            6 10 9 0 0 0; 
            0 1 6 7 0 0; 
            0 0 5 10 6 0; 
            0 0 0 3 7 3; 
            0 0 0 0 7 4]
    mat3 = [8 7 0 0 0 0 0 0 0 0; 
            6 2 8 0 0 0 0 0 0 0; 
            0 1 2 2 0 0 0 0 0 0; 
            0 0 4 4 9 0 0 0 0 0; 
            0 0 0 8 7 8 0 0 0 0; 
            0 0 0 0 6 3 3 0 0 0; 
            0 0 0 0 0 3 8 1 0 0; 
            0 0 0 0 0 0 1 3 5 0; 
            0 0 0 0 0 0 0 8 1 1;
            0 0 0 0 0 0 0 0 5 4]

    ### UPPER-BAND TRIDIAGONAL MATRICES ###
    mat4 = [7 5 0 0; 
            0 6 4 0; 
            0 0 7 9; 
            0 0 0 4]
    mat5 = [7 4 0 0 0 0; 
            0 2 1 0 0 0; 
            0 0 1 7 0 0; 
            0 0 0 1 10 0; 
            0 0 0 0 8 3; 
            0 0 0 0 0 5]
    mat6 = [8 1 0 0 0 0 0 0 0 0; 
            0 2 4 0 0 0 0 0 0 0; 
            0 0 10 9 0 0 0 0 0 0; 
            0 0 0 1 9 0 0 0 0 0; 
            0 0 0 0 9 8 0 0 0 0; 
            0 0 0 0 0 10 6 0 0 0; 
            0 0 0 0 0 0 2 2 0 0; 
            0 0 0 0 0 0 0 4 1 0; 
            0 0 0 0 0 0 0 0 7 3; 
            0 0 0 0 0 0 0 0 0 7]

    ### LOWER-BAND TRIDIAGONAL MATRICES ###
    mat7 = [7 0 0 0; 
            2 9 0 0; 
            0 6 4 0; 
            0 0 9 3]
    mat8 = [10 0 0 0 0 0; 
            5 2 0 0 0 0; 
            0 5 6 0 0 0; 
            0 0 5 7 0 0; 
            0 0 0 7 6 0; 
            0 0 0 0 1 6]
    mat9 = [6 0 0 0 0 0 0 0 0 0; 
            5 6 0 0 0 0 0 0 0 0; 
            0 8 7 0 0 0 0 0 0 0; 
            0 0 6 2 0 0 0 0 0 0; 
            0 0 0 3 4 0 0 0 0 0; 
            0 0 0 0 8 8 0 0 0 0; 
            0 0 0 0 0 4 4 0 0 0; 
            0 0 0 0 0 0 3 6 0 0; 
            0 0 0 0 0 0 0 8 10 0; 
            0 0 0 0 0 0 0 0 8 9]
    
    # Initialize center tridiagonal matrix type instances.
    tdm1 = TridiagonalMatrix{typeof(mat1).parameters[1]}(extract_tridiagonal_band(mat1, "center"))
    tdm2 = TridiagonalMatrix{typeof(mat2).parameters[1]}(extract_tridiagonal_band(mat2, "center"))
    tdm3 = TridiagonalMatrix{typeof(mat3).parameters[1]}(extract_tridiagonal_band(mat3, "center"))
   
    # Initialize upper tridiagonal matrix type instances.
    utdm1 = UpperTridiagonalMatrix{typeof(mat4).parameters[1]}(extract_tridiagonal_band(mat4, "upper"))
    utdm2 = UpperTridiagonalMatrix{typeof(mat5).parameters[1]}(extract_tridiagonal_band(mat5, "upper"))
    utdm3 = UpperTridiagonalMatrix{typeof(mat6).parameters[1]}(extract_tridiagonal_band(mat6, "upper"))
    
    # Initialize lower tridiagonal matrix type instances.
    ltdm1 = LowerTridiagonalMatrix{typeof(mat7).parameters[1]}(extract_tridiagonal_band(mat7, "lower"))
    ltdm2 = LowerTridiagonalMatrix{typeof(mat8).parameters[1]}(extract_tridiagonal_band(mat8, "lower"))
    ltdm3 = LowerTridiagonalMatrix{typeof(mat9).parameters[1]}(extract_tridiagonal_band(mat9, "lower"))

    # Check if matrices properly initialized.
    @test Matrix(tdm1) == mat1
    @test Matrix(tdm2) == mat2
    @test Matrix(tdm3) == mat3
    @test Matrix(utdm1) == mat4
    @test Matrix(utdm2) == mat5
    @test Matrix(utdm3) == mat6
    @test Matrix(ltdm1) == mat7
    @test Matrix(ltdm2) == mat8
    @test Matrix(ltdm3) == mat9

end

@testset "Matrix initializations" begin

    @testset "Center tridiagonal matrices" begin
        for dim = 2:MAX_DIM
            mat = random_tridiagonal_matrix(dim, "center")
            diags = extract_tridiagonal_band(mat, "center")
            tdm = TridiagonalMatrix{typeof(mat).parameters[1]}(diags)
            @test Matrix(tdm) == mat
        end
    end
    
    @testset "Upper tridiagonal matrices" begin
        for dim = 2:MAX_DIM
            mat = random_tridiagonal_matrix(dim, "upper")
            diags = extract_tridiagonal_band(mat, "upper")
            utdm = UpperTridiagonalMatrix{typeof(mat).parameters[1]}(diags)
            @test Matrix(utdm) == mat
        end
    end
    
    @testset "Lower tridiagonal matrices" begin
        for dim = 2:MAX_DIM
            mat = random_tridiagonal_matrix(dim, "lower")
            diags = extract_tridiagonal_band(mat, "lower")
            ltdm = LowerTridiagonalMatrix{typeof(mat).parameters[1]}(diags)
            @test Matrix(ltdm) == mat
        end
    end
end

@testset "Matrix size" begin

    @testset "Center tridiagonal matrices" begin
        for dim = 2:MAX_DIM
            mat = random_tridiagonal_matrix(dim, "center")
            diags = extract_tridiagonal_band(mat, "center")
            tdm = TridiagonalMatrix{typeof(mat).parameters[1]}(diags)
            @test size(mat) == size(tdm)
        end
    end
    
    @testset "Upper tridiagonal matrices" begin
        for dim = 2:MAX_DIM
            mat = random_tridiagonal_matrix(dim, "upper")
            diags = extract_tridiagonal_band(mat, "upper")
            utdm = UpperTridiagonalMatrix{typeof(mat).parameters[1]}(diags)
            @test size(mat) == size(utdm)
        end
    end
    
    @testset "Lower tridiagonal matrices" begin
        for dim = 2:MAX_DIM
            mat = random_tridiagonal_matrix(dim, "lower")
            diags = extract_tridiagonal_band(mat, "lower")
            ltdm = LowerTridiagonalMatrix{typeof(mat).parameters[1]}(diags)
            @test size(mat) == size(ltdm)
        end
    end
end

@testset "Linear getindex" begin
    
    @testset "Center tridiagonal matrices" begin
        for dim = 2:MAX_DIM
            mat = random_tridiagonal_matrix(dim, "center")
            diags = extract_tridiagonal_band(mat, "center")
            tdm = TridiagonalMatrix{typeof(mat).parameters[1]}(diags)
            for lin_idx = 1:prod(size(tdm))
                @test tdm[lin_idx]  == mat[lin_idx]
            end
        end
    end
   
    @testset "Upper tridiagonal matrices" begin
        for dim = 2:MAX_DIM
            mat = random_tridiagonal_matrix(dim, "upper")
            diags = extract_tridiagonal_band(mat, "upper")
            utdm = UpperTridiagonalMatrix{typeof(mat).parameters[1]}(diags)
            for lin_idx = 1:prod(size(utdm))
                @test utdm[lin_idx]  == mat[lin_idx]
            end
        end
    end
    
    @testset "Lower tridiagonal matrices" begin
        for dim = 2:MAX_DIM
            mat = random_tridiagonal_matrix(dim, "lower")
            diags = extract_tridiagonal_band(mat, "lower")
            ltdm = LowerTridiagonalMatrix{typeof(mat).parameters[1]}(diags)
            for lin_idx = 1:prod(size(ltdm))
                @test ltdm[lin_idx]  == mat[lin_idx]
            end
        end
    end
end

@testset "Cartesian getindex" begin
    
    @testset "Center tridiagonal matrices" begin
        for dim = 2:MAX_DIM
            mat = random_tridiagonal_matrix(dim, "center")
            diags = extract_tridiagonal_band(mat, "center")
            tdm = TridiagonalMatrix{typeof(mat).parameters[1]}(diags)
            for idx1 = 1:size(tdm, 1)
                for idx2 = 1:size(tdm, 2)
                    @test tdm[idx1, idx2] == mat[idx1, idx2]
                end
            end
        end
    end
    
    @testset "Upper tridiagonal matrices" begin
        for dim = 2:MAX_DIM
            mat = random_tridiagonal_matrix(dim, "upper")
            diags = extract_tridiagonal_band(mat, "upper")
            utdm = UpperTridiagonalMatrix{typeof(mat).parameters[1]}(diags)
            for idx1 = 1:size(utdm, 1)
                for idx2 = 1:size(utdm, 2)
                    @test utdm[idx1, idx2] == mat[idx1, idx2]
                end
            end
        end
    end

    @testset "Lower tridiagonal matrices" begin
        for dim = 2:MAX_DIM
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
end

@testset "Linear setindex" begin
    
    @testset "Center tridiagonal matrices" begin
        for dim = 2:MAX_DIM
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
    end
    
    @testset "Upper tridiagonal matrices" begin
        for dim = 2:MAX_DIM
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
    end
    
    @testset "Lower tridiagonal matrices" begin
        for dim = 2:MAX_DIM
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
end

@testset "Cartesian setindex" begin
    
    @testset "Center tridiagonal matrices" begin
        for dim = 2:MAX_DIM
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
    end
    
    @testset "Upper tridiagonal matrices" begin
        for dim = 2:MAX_DIM
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
    end

    @testset "Lower tridiagonal matrices" begin
        for dim = 2:MAX_DIM
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
end

@testset "Matrix vector multiplication" begin
    
    @testset "Center tridiagonal matrices" begin
        for dim = 2:MAX_DIM
            mat = random_tridiagonal_matrix(dim, "center")
            diags = extract_tridiagonal_band(mat, "center")
            tdm = TridiagonalMatrix{typeof(mat).parameters[1]}(diags)
            v = rand(dim)
            @test tdm*v ≈ mat*v atol=1.0e-4
        end
    end
    
    @testset "Upper tridiagonal matrices" begin
        for dim = 2:MAX_DIM
            mat = random_tridiagonal_matrix(dim, "upper")
            diags = extract_tridiagonal_band(mat, "upper")
            utdm = UpperTridiagonalMatrix{typeof(mat).parameters[1]}(diags)
            v = rand(dim)
            @test utdm*v ≈ mat*v atol=1.0e-4
        end
    end
    
    @testset "Lower tridiagonal matrices" begin
        for dim = 2:MAX_DIM
            mat = random_tridiagonal_matrix(dim, "lower")
            diags = extract_tridiagonal_band(mat, "lower")
            ltdm = LowerTridiagonalMatrix{typeof(mat).parameters[1]}(diags)
            v = rand(dim)
            @test ltdm*v ≈ mat*v atol=1.0e-4
        end
    end
end


@testset "LU decomposition" begin
    
    @testset "Center tridiagonal matrices" begin
        for dim = 2:MAX_DIM
            mat = random_tridiagonal_matrix(dim, "center")
            diags = extract_tridiagonal_band(mat, "center")
            tdm = TridiagonalMatrix{typeof(mat).parameters[1]}(diags)
            l, u = lu(tdm)
            l_ref, u_ref = lu(Matrix(tdm), Val(false))
            @test l ≈ l_ref
            @test u ≈ u_ref
        end
    end
   
    @testset "Upper tridiagonal matrices" begin
        for dim = 2:MAX_DIM
            mat = random_tridiagonal_matrix(dim, "upper")
            diags = extract_tridiagonal_band(mat, "upper")
            utdm = UpperTridiagonalMatrix{typeof(mat).parameters[1]}(diags)
            l, u = lu(utdm)
            l_ref, u_ref = lu(Matrix(utdm), Val(false))
            @test l ≈ l_ref atol=1.0e-4
            @test u ≈ u_ref atol=1.0e-4
        end
    end

    @testset "Lower tridiagonal matrices" begin
        for dim = 2:MAX_DIM
            mat = random_tridiagonal_matrix(dim, "lower")
            diags = extract_tridiagonal_band(mat, "lower")
            ltdm = LowerTridiagonalMatrix{typeof(mat).parameters[1]}(diags)
            l, u = lu(ltdm)
            l_ref, u_ref = lu(Matrix(ltdm), Val(false))
            @test l ≈ l_ref atol=1.0e-4
            @test u ≈ u_ref atol=1.0e-4
        end
    end
end

@testset "Backslash operator" begin
    
    @testset "Center tridiagonal matrices" begin
        for dim = 2:MAX_DIM
            mat = random_tridiagonal_matrix(dim, "center")
            diags = extract_tridiagonal_band(mat, "center")
            tdm = TridiagonalMatrix{typeof(mat).parameters[1]}(diags)
            v = rand(dim)
            @test tdm\v ≈ Matrix(tdm)\v atol=1.0e-4
        end
    end
    
    @testset "Upper tridiagonal matrices" begin
        for dim = 2:MAX_DIM
            mat = random_tridiagonal_matrix(dim, "upper")
            diags = extract_tridiagonal_band(mat, "upper")
            utdm = UpperTridiagonalMatrix{typeof(mat).parameters[1]}(diags)
            v = rand(dim)
            @test utdm\v ≈ Matrix(utdm)\v atol=1.0e-4
        end
    end

    @testset "Lower tridiagonal matrices" begin 
        for dim = 2:MAX_DIM
            mat = random_tridiagonal_matrix(dim, "lower")
            diags = extract_tridiagonal_band(mat, "lower")
            ltdm = LowerTridiagonalMatrix{typeof(mat).parameters[1]}(diags)
            v = rand(dim)
            @test ltdm\v ≈ Matrix(ltdm)\v atol=1.0e-4
        end
    end
end

@testset "tridiagonalization" begin
    for dim = 2:MAX_DIM
        M = rand(dim, dim)
        M +=  M'
        trid, Q = tridiag(M)
        @test Q'*Matrix(trid)*Q ≈ M atol=1.0e-4
    end
end

@testset "shifted inverse power method" begin
    for dim = 2:MAX_DIM

        # Initialize random real symmetric matrix.
        M = rand(dim, dim)*5.0
        M += M'

        # Get reference eigenvalues.
        evs = Set(eigvals(M))

        # Initialize set for storing found eigenvalues.
        found_evs = Set()

        # Sweep over vicinity of known reference eigenvalues
        for e = minimum(evs)-1:0.1:maximum(evs)

            # Perform shifted inverse power method to get estimate of
            # eigenvalue closest to initial guess e.
            e_found, evec_found = inv_eigen(M, e)

            # @test M*evec_found ≈ e_found*evec_found atol=1.0e-1

            # Add found eigenvalue to set.
            if !isnothing(e_found)
                push!(found_evs, (round(e_found, digits=4), evec_found))
            end
        end

        # Check if all eigenvalues found.
        found = false
        for el1 in evs
            for el2 in found_evs
                if abs(el1 - el2[1]) < 1.0e-1
                   found = true

                   # Test if eigenvector correct.
                   @test M*el2[2] ≈ el1[1]*el2[2] atol=1.0e-1
                   break
               end  
            end
            
            # Test if eigenvalue found by shifted inverse power method.
            @test found
        end

    end
end

