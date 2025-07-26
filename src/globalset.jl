tolerance(T) = eps(real(T))^(1 // 2)
strict_tol(T) = eps(real(T))^(2 // 3)
relax_tol(T) = eps(real(T))^(1 // 4)

"""
    APC

Alias of Complex type (Arbitrary Precision Complex).

See also: [`API`](@ref), [`APF`](@ref).
"""
const APC = Complex{BigFloat}
