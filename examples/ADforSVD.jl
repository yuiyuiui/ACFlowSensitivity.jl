function L(A)
    @assert size(A,1)==size(A,2)
    n=size(A,1)
    res=0.0
    for i=1:n
        for j=1:n
            res+=A[i,j]
        end
    end
    return res
end

δ=1e4*eps(Float64)
δA=δ*rand(2,2)

A=rand(2,2)
(L(A+δA)-L(A))*(δA^(-1))

