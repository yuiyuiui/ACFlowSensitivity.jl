**Flow of SSK**

![alt text](flow.jpg)

1. Initialize the configuration
2. Find the best $\theta$ of Monte Carlo(MC)/Simulated Annealing(SA)
3. Do MC/SA and measure configuration every givn steps.
4. Average all measured configurations to get the final spectrum and calculate $Re(A(w+i\eta))$
5. Get the poles from $Re(A(w+i\eta))$

Detailed steps:
2: For a series of $\theta$, get their corresponding $\chi^2$ with small-scaled MC. Then decide a good $\theta$ by `chi2kink` or `chi2min`.
4. What we get is a distribution but not a exact configuration.
5. This means we can't directly get the poles. We get poles by extracting peaks from the chart of $w\to Re(A(w+i\eta))$.

Of course it's possible that the number of peaks of $Re(A(w+i\eta))$ is not equal to the number of poles we set. If this happens, we just choose the last configuration.

**Differentiate**
Denote:
$$G^R = [\Re(G_1), \cdots, \Re(G_N), \Im(G_1), \cdots, \Im(G_N)]$$

For
$$p(A,G) = \frac{1}{Z_G} e^{-\chi^2(A,G)/(2\Theta)}$$

Denote $p_0 = e^{-\chi^2(A,G)/(2\Theta)}$

Then
$$\mathbb{E}(\frac{\partial}{\partial G^R}\log(p)) = 0\\
=\mathbb{E}(\frac{\partial}{\partial G^R}\log(p_0)) -\frac{\partial}{\partial G^R}\log(Z_G)\\
\Longrightarrow \mathbb{E}(\frac{\partial}{\partial G^R}\log(p_0)) = \frac{\partial}{\partial G^R}\log(Z_G)
$$

So
$$\frac{\partial}{\partial G^R}\mathbb{E}(A)\\
=\mathbb{E}(A\frac{\partial}{\partial G^R}\log(p))\\
=\mathbb{E}(A\frac{\partial}{\partial G^R}\log(p_0)) - \mathbb{E}(A)\frac{\partial}{\partial G^R}\log(Z_G)\\
=\mathbb{E}(A\frac{\partial}{\partial G^R}\log(p_0)) - \mathbb{E}(A)\frac{\partial}{\partial G^R}\log(\log(p_0))\\
=\text{Cov}(A, \frac{\partial}{\partial G^R}\log(p_0))\\
=\text{Cov}\left(A, -\frac{\partial}{\partial G^R}\sum_{j=1}^N(G^R_j-K_jA)^2/\left(2\sigma_j^2\Theta\right)\right)\\
=\frac{1}{\Theta}\text{Cov}\left(A, -\left[\frac{G^R_j-K_jA}{\sigma_j^2}\right]_j\right)\\
=\frac{1}{\Theta}\text{Cov}\left(A,\Sigma^{-2}KA\right),~\left(\Sigma = \text{diag}(\sigma_1, \cdots, \sigma_N)\right)\\
=\frac{1}{\Theta}\text{Cov}\left(A,A\right)K'\Sigma^{-2}\\
$$

We use $F$ represents the map from $A$ on fine mesh to $A$ on mesh. Then:
$$A_{\text{mesh}} = FA\\
A_{\text{out}} = \Delta^{-1}\mathbb{E}(A_{\text{mesh}})\\
= \Delta^{-1}F\mathbb{E}(A)\\
\Longrightarrow \frac{\partial}{\partial G^R}\mathbb{E}(A_{\text{out}}) = \Delta^{-1}F\frac{\partial}{\partial G^R}\mathbb{E}(A)\\
=\frac{1}{\Theta}\Delta^{-1}F\text{Var}(A)K'\Sigma^{-2}\\
=\frac{1}{\Theta}\Delta^{-1}F(\mathbb{E}(AA')-\mathbb{E}(A)\mathbb{E}(A'))K'\Sigma^{-2}\\
=\frac{1}{\Theta}\left(\Delta^{-1}\mathbb{E}(FAA'K')-A_{\text{out}}\mathbb{E}(A'K')\right)\Sigma^{-2}
$$
