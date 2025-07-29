Why I don't use `pγdiff` to calculate the derivative of `p` and `γ` with respect to `GFV` in `NAC`:

Although it does set minimize $\chi^2 = \sum_{j=1}^N\left|\sum_{k=1}^M  \frac{1/M}{i\omega_j - p_k} - G_j \right|^2$ as a prepose, it doesn't constraint $\sum_{j=1}^M\gamma_j=1$. It means if you set

$$\text{GFV} -> (1+\delta)\text{GFV}$$

Then with `NAC` what you get may be still $p$ and $(1+\delta)\gamma$, which is conflict with the assumption of $\gamma = 1/M$ in $\chi^2$.