For given poles $P  =\{p_i\}_{i=1}^M$, we use the Newton's method to minimize:
$$\|K\gamma - G\|^2$$

$$K = \left[\frac{1}{i\omega_j - p_k}\right]_{j=1,k=1}^{N,M}$$

The following is about how to find the poles.
For a given BarRatFunc
$$f(z) = \frac{\sum_{j=1}^n\frac{w_jv_j}{z-g_j}}{\sum_{j=1}^n\frac{w_j}{z-g_j}}$$

solve
$$\det\left(\left[\begin{matrix}0&w_1&..&w_n\\1&g_1\\&&...\\1&&&g_n \end{matrix}\right] - \lambda\left[\begin{matrix}0&\\&I_n \end{matrix}\right]\right)=0$$

Because it's easy to prove $\forall \lambda$ satisfies above formula, we have
$$\sum_{j=1}^n\frac{w_j}{\lambda-g_j}=0$$