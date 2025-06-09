### 1. Prony approximation
https://en.wikipedia.org/wiki/Prony%27s_method

it fit a set of sampling data with a sum of complex exponential functions. That is to say, exponentially decaying triangular functions.

$$f(t) = \sum_{i=1}^{N} A_i e^{\alpha_i t} \cos(\omega_i t + \phi_i)$$

### 2. Improved Prony approximation
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