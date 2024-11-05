For 
$${\rm Function}:A(x)=\sum \gamma_i\delta(x-x_i)$$

$${\rm Input}:\{\omega_n=(n+\frac{1}{2})\frac{2\pi}{\beta}\},~n=0,..,N-1\\
\{G(i\omega_n)=\int_{R}\frac{A(x)}{i\omega_n-x}dx\}+{\rm noise},~n=0,..,N-1$$

$$~$$
$${\rm Output}: {\rm array}:mesh\\
{\rm Function}:\widetilde{A}(x)$$

In the process to reconstruct the spectral density function, can we just set the poles of of $\widetilde{A}$ as $\{x_i\},~i=0,..,N-1$ ?

In the barycentric method of ACFlow, it use aaa algorithm to get a rational approximation $\frac{N(z)}{D(z)}$ and get poles from it.

But with this method , you can only get $N/2$ poles as most and they can't be accurately $\{i\omega_n\}$.

So can we directly set poles as $\{\omega_n\}$ ?