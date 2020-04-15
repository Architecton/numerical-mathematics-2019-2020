function sample_laplacian(dim)
    # TODO
end

# Plot eigenvalues
evs = eigvals(L)
plot(evs[1:EIG_LIM1], seriestype = :scatter, title=@sprintf("First %d Eigenvalues\n.", EIG_LIM1))

# Find eigenvectors for smallest eigenvalues.
for ev in evs
    e, x = inv_eigen(M, ev)
    
end
