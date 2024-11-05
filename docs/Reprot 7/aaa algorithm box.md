$\rm Input:$
$$\{(z_k,G(z_k))\},~k=1,..,n$$

$\rm Step 1:$
chosen points set $A$
wating points set $B$
Put all $z_k$ into $B$
best_err=$\infty$
best_A=$\empty$
best_weight=$\empty$
$$R(z)=|G(z)-\bar{G}|,~z\in B$$

$$L=\left(\frac{G(z_j)-G(z_k)}{z_j-z_k}\right)_{jk}$$

$L$ is called Lowner matrix.
$$~$$
$\rm Setp 2:$
choose
$$z'={\rm argmax}_{z\in B} R$$

add $z'$ into $A$ and delete $z'$ from $B$

Denote $L_{AB}$ as the submatrix of $L$ that the rows of $L_{AB}$ correspond to the rows of $A$ where the elements are located, and the columns of $L_{AB}$ correspond to the columns of $B$ where the elements are located.

Select $w$ as the eigenvector corresponding to the smallest singular value of $L_{AB}$

$$~$$
$\rm Step~3.$
Construct Barycentric rational approximation with $A$
$$\frac{N_A}{D_A}$$

$$N_A(z)=\sum_{z_j\in A}\frac{w_jG(z_j)}{z-z_j},~D_A(z)=\sum_{z_j\in A}\frac{w_j}{z-z_j}$$

$$~$$
$\rm Step~4.$
Renew:
$$R(z)=|G(z)-\frac{N_A(z)}{D_A(z)}|,~z\in B$$

$$~$$
$\rm Step~5.$
$$err=\max_{z\in B}R(z)$$

if err<best_err
Renew:
best_err=err
best_A=A
best_weight=$w$

$$~$$
$\rm Step~5.$
If the best_err is sufficiently small, or if many iterations have been performed, or if the residual has not changed significantly over the last few iterations, then terminate the loop. 

Otherwise, return to Step 2.

$$~$$
$\rm Step~6.$
return best_A,best_weight

Assuming $L$ is a $2m\times 2m$ square matrix, then best_weight is equal to
$$L'= {\rm argmin}\{ {\rm singulars~of }~L_{nk}:L_{nk}{\rm~is~ n\times k ~submatrix ~of ~}L,~k\leq m-1\}$$

$$L'w=\left(\min \sigma(L)\right) w$$




