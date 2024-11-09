$\rm 1.~ How ~ACFlow ~~brc ~algorithm ~find~ poles ~with~ delts ~type ~spectral~ density$

The way barycentric algorithm get poles and corresponding amplitudes is function poles!(). 

${\rm poles!():}$
$\rm Input:~Barycentric~Function$
$~$
${\rm Output:~reconstruct~poles ~and~amplitudes}~\{\widetilde{P}_k,\widetilde{\gamma}_k\}_{k=1}^m $
$~$
${\rm (1).~ bc\_poles():}$
   Find the places of poles of a barycentric rational approximation.
   For such a function:
   $$\frac{\sum_{k=1}^m\frac{w_k}{z-z_k}\overline{G}(z_k)}{\sum_{k=1}^m\frac{w_k}{z-z_k}},~w_k\neq0,~z_k\neq z_j~when ~k\neq j$$

   It's poles are solutions of equation: (including Removable singularity)
   $$\left|\begin{pmatrix}0&w_1&..&w_m\\1&&\\...&&{\rm diag\{z_1,..,z_m\}}\\1&&& \end{pmatrix}-\lambda \begin{pmatrix}0 & \\ &I_m\end{pmatrix}\right|=0$$

   And it can also be proven that all $z_k$ are not solution, which is reasonable.
   $~$
   ${\rm Input:}~\{z_k,w_k\}_{k=1}^m$ 
   $~$
   ${\rm Output:~recontruct~poles}~\{\widetilde{P}_k\}_{k=1}^m$





   $$~$$

$\rm (2).~ gradient\_fd()$
   Get derivatives of the error function:
   $$EA:\{\gamma_k\}_{k=1}^m\to \sum_{j=1}^N\left|\overline{G}(i\omega_j)-\sum_{k=1}^m\frac{\gamma_k}{i\omega_j-pole_k}\right|$$

   by finite difference. EA means error of amplitudes.
   $~$

   ${\rm Input:}~\{\widetilde{P}_k\}_{k=1}^m$
   $~$
   ${\rm Output:~function}~\nabla EA(\{\gamma_k\}_{k=1}^m)$
   $$~$$ 

$\rm (3). ~optimize()$
   Find the minmum point of function $EA$ by Newton's iterative method:
   $$x_{k+1}=x_k-H^{-1}(x_k)\nabla EA(x_k)$$

   In which $H^{-1}$ is the Hessel matrix:
   $$\frac{\partial^2EA}{\partial x^2}$$

   ${\rm Input:~function}~EA(x),~\nabla EA(x)$
   $~$
   ${\rm Output:~minmum ~points}$, that is to say, reconstruct amplitudes $\{\widetilde{\gamma}_k\}_{k=1}^m$
   $$~$$

$\rm (4).~Improvement$

We find that this method can find the minimum pole with hight accuracy but much less accurate for farther poles. So we find poles on by one. That is to say, after find $\{\widetilde{P}_k,\widetilde{\gamma}_k\}$, we do a update
$$\bar{G}_j^{(k)}\to \bar{G}_j^{(k+1)}=\bar{G}_j^{(k)}-\frac{\gamma_k}{z_j-\widetilde{P}_k},~~j\geq k+1$$

And then input $\bar{G}_j^{(k+1)},~j=k+1,..,m$ and calculate $\{\widetilde{P}_{k+1},\widetilde{\gamma}_{k+1}\}$


--------

$\rm 2.~Introdunction ~to ~AD$
(1) ForwardDiff.jl :  The way it realizes AD is to use dual number and store the derivatives of the basic functions in advance.

If you have a basic function $f(x)$, then ForwardDiff.jl apply $(f,f'$) on dual number $(a,b)$ to get $(f(a),f'(a)b)$.

--------
$\rm 3. ~Where ~difficult ~to~ apply ~AD$
1. svd
2. eigenval in function poles!
3. E\B in function poles


