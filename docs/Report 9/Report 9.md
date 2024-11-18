AD for barycentric method

$${\rm Input:}~\mathcal{G_0}=\{G_{0}(iw_k)\}_{k=1}^N$$

$${\rm Output:}~\nabla L_{\mathcal{G_0}}(\mathcal{G_0}) \\ L_{\mathcal{G_0}}=\int_{\mathbb{R}}\left|\widetilde{A_{\mathcal{G_0}}}(\mathcal{G},x)-\widetilde{A}(\mathcal{G_0},x)\right|^2dx $$

-------
$\rm STEP:$

$\rm Step 1:$ 
Input $\{iw_k\}_{k=1}^N,~\mathcal{G_0}$, get corresponding Lowner Matrix 
$$L(\mathcal{G_0})=\left(\frac{G(iw_j)-G(iw_k)}{iw_j-iw_k}\right)_{1\leq j,k\leq N}$$.

Then run aaa algorithm and get the index $I(\mathcal{G_0})$ of the submatrix $L_{sub}(\mathcal{G_0})$.

Then do svd decomposition on $L_{sub}(\mathcal{G_0})$ and we get the reconstruct spectral density function $\widetilde{A}_{sub}(\mathcal{G_0},x)$.

$\rm Step 2:$
Now we construct Loss function: 
$$\mathcal{G}\to L_{\mathcal{G_0}}(\mathcal{G})$$

as a composition of four functions.

function 1: 
$$\mathcal{G}\to L(\mathcal{G})[I(\mathcal{G_0})]$$

function 2: 
$${\rm svd:}~L(\mathcal{G})[I(\mathcal{G_0})] \to (\_,\_,V_{\mathcal{G_0}}(\mathcal{G}))$$.

function 3: 
$$V_{\mathcal{G_0}}(\mathcal{G})[:,end]\to \widetilde{A_{\mathcal{G_0}}}(\mathcal{G},x)=-\frac{1}{\pi}{\rm Im}\widetilde{G_{\mathcal{G_0}}}(\mathcal{G},x)$$

function 4:
$$\widetilde{A_{\mathcal{G_0}}}(\mathcal{G},x) \to L_{\mathcal{G_0}}(\mathcal{G})=\int_{\mathbb{R}}\left|\widetilde{A_{\mathcal{G_0}}}(\mathcal{G},x)-\widetilde{A}(\mathcal{G_0},x)\right|^2dx$$

$\rm Step 3:$
Calculate 
$$\nabla L_{\mathcal{G_0}}(\mathcal{G_0})=\nabla f_4(\widetilde{A_{\mathcal{G_0}}}(\mathcal{G_0},x))*\nabla f_3(V_{\mathcal{G_0}}(\mathcal{G_0})[:,end])*\nabla f_2(L(\mathcal{G_0})[I(\mathcal{G_0})])*\nabla f_1(\mathcal{G_0})*e$$

Here $e$ is the column vector of all ones.

And in fact we combine function 2 to 4 as a whole function to apply AD on complex svd.

------
Meaning of definition about gradient of $\mathbb{C}\to \mathbb{C}$ :


