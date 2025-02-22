In the MaxEntropy method, we denote the output grid as $op$, the weight of the output grif is vector $w$, the kernel matrix $K=(\frac{1}{w-iw_j})$. So we have:
$$G={\rm diagm}(w)KAe$$

$G$ is the vector of green function values and $e$ is a column vector with all elements being $1$.
Then what you do is:

$$K\to U_0,S_0,V_0=svd(K) \to U,S,V$$

$S$ only preserves elements in $S_0$ bigger than $1e-12$. And $U,V$ only preserve related columns. Then you express
$$A={\rm diagm}(m)\exp(Vu)$$

I understand formulas you use in following Newton's method with this form. And I also see the vector $A$ can in fact be expressed as $Vu$ because $KV^{\perp}=0$. But I can't understand why
$$A_{opt} \in \mathcal{S}=\{{\rm diagm}(m)\exp(Vu):u \in \mathbb{R}^n\}$$

Or at least why $A_{opt}$ is close to $\mathcal{S}$. 

I am really interested in this numerical skill. Looking forward to you reply!


=====


In the Maximum Entropy (MaxEnt) method, we denote the output grid as \( op \), with the weight vector of the output grid being \( w \). The kernel matrix is defined as \( K = \left( \frac{1}{w - iw_j} \right)_{w \in op, j=1:N} \). This leads to the expression:
$$ G = {\rm diagm}(w)KAe $$
where \( G \) represents the vector of Green's function values, and \( e \) is a column vector of ones.

The numerical procedure proceeds as follows:
1. **Singular Value Decomposition (SVD)**:
   $$ K \to U_0, S_0, V_0 = {\rm svd}(K) \to U, S, V $$
   Here, \( S \) retains only elements of \( S_0 \) larger than \( 1 \times 10^{-12} \), with \( U \) and \( V \) preserving the corresponding columns.

2. **Parametrization**:
   The vector \( A \) is expressed as
   $$ A = {\rm diagm}(m)\exp(Vu) $$

While I understand the formulation of Newton's method in this context, and recognize that \( A \) could theoretically be simplified to \( Vu \) due to the property \( KV^{\perp} = 0 \), I find difficulty in rigorously justifying why the optimal solution satisfies:
$$ A_{\rm opt} \in \mathcal{S} = \left\{ {\rm diagm}(m)\exp(Vu) : u \in \mathbb{R}^n \right\} $$
or at minimum, why \( A_{\rm opt} \) should be closely approximated by elements in \( \mathcal{S} \).

This numerical technique intrigues me. I would greatly appreciate clarification on the theoretical underpinning of this parametrization choice.
