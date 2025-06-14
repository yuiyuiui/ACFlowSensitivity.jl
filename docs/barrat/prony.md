### 1. Prony Interpolation
https://en.wikipedia.org/wiki/Prony%27s_method

it fit a set of sampling data with a sum of complex exponential functions. That is to say, exponentially decaying triangular functions.

$$f(t) = \sum_{i=1}^{N} A_i e^{\alpha_i t} \cos(\omega_i t + \phi_i)$$

### 2. Prony Approximation
[*On approximation of functions by exponential sums*](https://www.sciencedirect.com/science/article/pii/S106352030500014X)

* Summary: low-rank decomposition of sampling Hankel matrix.

* Algorithm box:

For $(2N-1)$ uniformly (how about non-uniform?) sampling data $\{(z_k,f_k)\}_{k=1}^{2N-1}$ with $f_k$ representing $f(k)$, construct the Hankel matrix:
$$H = \left[\begin{matrix}f_1&\cdots&f_N\\f_2&\cdots&f_{N+1}\\\vdots&\ddots&\vdots\\f_N&\cdots&f_{2N-1}\end{matrix}\right]$$

Apply svd decomposition (Takagi’s factorization) to $H$, we get its singular values:
$$\sigma_1\geq..\geq\sigma_{M-1}>\epsilon\geq\sigma_M\leq\epsilon$$

Then we get $u$ s.t. $Hu = \sigma_M \overline{u}$. The we denote:

$$P_u(z) = \sum_{n=1}^{N} u_n z^n$$

Get $M$ zeros ${\gamma_i}$ of $P_u(z)$ in the  “significant” region (near the unit circle) s.t. corresponding $\{w_i\}$ that are got by:
$$\sum_{m=1}^M w_m\gamma_m^k \approx f_k, k=1,2,...,2N-1$$

(using the least square method)

satisfies: $|w_m|\geq \sigma_M$.

Then we get the Prony approximation:
$$\widetilde{f}(t) = \sum_{m=1}^M w_m \gamma_m^t \\ \|\widetilde{f}(k)-f(k)\| = O(\epsilon),~k=1,..,2N-1$$

### Green's Function's Analytic Continuation with Prony Approximation
[*Minimal Pole Representation and Controlled Analytic Continuation of Matsubara Response Functions*](https://arxiv.org/abs/2312.10576)
Represent the Green's function:
$$G(z) = \sum_{k} \frac{A_k}{z-\xi_k},~\xi_k\in\mathbb{C}, Im(z)\leq0$$

1. Fit $\{G(iw_n)\}_{n=1}^N$ with Prony Approximation and get $G^1(z)$
2. By using inverse Joukowsky transform, map the $[iw_1,iw_N]$ (and its copy) to the unit circle and map the real axis to a circle $\Gamma$ in the interior of the closed unit disk $D$. The lower complex plane is mapped to $D^o$. Then calculate:
$$h_k:=\frac{1}{2\pi i}\int_{\Gamma}G^1(z)z^kdz = \sum_{m} A_m \widetilde{\xi}_m^k$$ and form a prony problem
3. Solve above prony problem and get a compact representation: $\{\widetilde{\xi}_m\}_{m=1}^M$
4. using Joukowsky transform and recover $\{\xi_k\}_{k=1}^M$

