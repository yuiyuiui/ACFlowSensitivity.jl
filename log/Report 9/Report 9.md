AD for barycentric method

$${\rm Input:}~\mathcal{G_0}=\{G_{0}(iw_k)\}_{k=1}^N$$

$${\rm Output:}~\nabla L_{\mathcal{G_0}}(\mathcal{G_0}) \\ L_{\mathcal{G_0}}=\int_{\mathbb{R}}\left|\widetilde{A_{\mathcal{G_0}}}(\mathcal{G},x)-\widetilde{A}(\mathcal{G_0},x)\right|^2dx $$

-------
$\rm STEP~FOR~BACKWARD~AD:$

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

Short for $L_0$

function 2: 
$$L(\mathcal{G})[I(\mathcal{G_0})] \to {\rm svd}\left(V_{\mathcal{G_0}}(\mathcal{G})\right).V[:,end]$$.

function 3: 
$$\mathcal{G},w_{\mathcal{G_0}}(\mathcal{G})\to \widetilde{A_{\mathcal{G_0}}}(\mathcal{G},x)=-\frac{1}{\pi}{\rm Im}\widetilde{G_{\mathcal{G_0}}}(\mathcal{G},x)$$

function 4:
$$\widetilde{A_{\mathcal{G_0}}}(\mathcal{G},x) \to L_{\mathcal{G_0}}(\mathcal{G})=\int_{\mathbb{R}}\left|\widetilde{A_{\mathcal{G_0}}}(\mathcal{G},x)-\widetilde{A}(\mathcal{G_0},x)\right|^2dx$$

$~$
$\rm Step 3:$

$$L_{\mathcal{G_0}}(\mathcal{G_0})=f_4\circ f_3\left(\mathcal{G},f_2\circ f_1(\mathcal{G})\right)$$

Calculate: 
$$\nabla L_{\mathcal{G_0}}(\mathcal{G_0})=2\frac{\partial f_4\circ f_3}{\partial \mathcal{G^*}}(\mathcal{G},f_2\circ f_1(\mathcal{G_0}))+2\frac{\partial f_4\circ f_3(\mathcal{G_0},f_2)}{\partial L_0^*}\cdot\frac{\partial L_0^*}{\partial \mathcal{G^*}}$$

And in fact we combine function 2 to 4 as a whole function to apply AD on complex svd, as well as function 3 and 4.


The formular is right because for 
$$f,g:\mathbb{C}\to \mathbb{C},~g~{\rm or}~f \rm ~is ~analytic$$

$$\Longrightarrow \nabla g(f(z)) = 2\frac{\partial g}{\partial f^*}\cdot\frac{\partial f^*}{\partial z^*}$$

And for general complex function
$$g=A(x,y)+iB(x,y),~f(x,y)=u(x,y)+iv(x,y)$$

If we see matrix
$$\begin{bmatrix}
a&b\\c&d
\end{bmatrix}~{\rm as~a~complex ~number~} a-d+i(b+c)$$

Then we have 
$$\nabla g(f(z))=\begin{bmatrix}
\frac{\partial A(u,v)}{\partial u} & \frac{\partial A(u,v)}{\partial v}\\
\frac{\partial B(u,v)}{\partial u} & \frac{\partial B(u,v)}{\partial v}
\end{bmatrix}*\begin{bmatrix}
\frac{\partial u}{\partial x} & \frac{\partial u}{\partial y}\\
\frac{\partial v}{\partial x} & \frac{\partial v}{\partial y}
\end{bmatrix}=\nabla g ~{\rm matrix~product~}\nabla f$$

So if we denote such matrix product as $*_m$, then we have 
$$\nabla g(f(z))=\nabla g *_m\nabla f$$

------

$\rm STEP~FOR~FORWARD~AD:$



