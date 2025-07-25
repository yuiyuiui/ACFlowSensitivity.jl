#!/usr/bin/env julia

using DelimitedFiles
using Printf
using ACFlow
using Plots

# Deal with self-energy function
#
# Read self-energy function
dlm = readdlm("siw.data")
#
# Get grid
grid = dlm[:, 1]
#
# Get self-energy function
Σinp = dlm[:, 2] + im * dlm[:, 3] # Value part
Σinp+=rand(20)
Σorigin=copy(Σinp)
Σerr = dlm[:, 4] + im * dlm[:, 5] # Error part
#
# Subtract hartree term
Σ∞ = 1.0
@. Σinp = Σinp - Σ∞

# For MaxEnt solver

# Setup parameters
#
# For [BASE] block
# See types.jl/_PBASE for default setup
B = Dict{String,Any}("solver" => "MaxEnt",  # Choose MaxEnt solver
                     "mtype" => "gauss",   # Default model function
                     "mesh" => "tangent", # Mesh for spectral function
                     "ngrid" => 20,        # Number of grid points for input data
                     "nmesh" => 801,       # Number of mesh points for output data
                     "wmax" => 8.0,       # Right boundary of mesh
                     "wmin" => -8.0,      # Left boundary of mesh
                     "beta" => 10.0)
#
# For [MaxEnt] block
# See types.jl/_PMaxEnt for default setup
S = Dict{String,Any}("nalph" => 15,        # Number of α parameters
                     "alpha" => 1e12,      # Starting value of α parameter
                     "blur" => -1.0)
#
setup_param(B, S)

# Call the solver
mesh, Aout, Σout = solve(grid, Σinp, Σerr)

# Calculate final self-energy function on real axis
#
# Construct final self-energy function
@. Σout = Σout + Σ∞
#

plot(grid, imag(Σorigin))
plot(mesh, real(Σout))
plot(mesh, imag(Σout))
