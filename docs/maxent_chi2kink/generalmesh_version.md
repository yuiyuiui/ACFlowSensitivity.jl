$$A(u) = m\circ \exp(Vu)$$

$$\frac{\partial A}{\partial u} = \text{diag}(A)V$$

$$\chi^2(A) = \frac{1}{\sigma^2}(G-\widetilde{G})'(G-\widetilde{G}) = (K\text{diag}(w)A-G)'(K\text{diag}(w)A-G)$$

$$\frac{\partial \chi^2}{\partial A}=\frac{2}{\sigma^2}\text{diag}(w)K'(K\text{diag}(w)A-G)$$

$$\text{SJ} = A-m-A\circ \ln(A \circ/ m)$$

$$\frac{\partial \text{SJ}}{\partial A} = -\text{diag}(w)\ln(A\circ/m)=-\text{diag}(w)Vu$$

$$Q(A) = \alpha\text{SJ}-\frac{1}{2}\chi^2$$

$$\frac{\partial Q}{\partial A} = -\text{diag}(w)V(\alpha u+\frac{1}{\sigma^2}SU'(K\text{diag}(w)A-G))$$

$$\frac{\partial Q}{\partial A}=0 \Longleftrightarrow J(u) = \alpha u+\frac{1}{\sigma^2}SU'(K\text{diag}(w)A-G)=0$$

$$H(u) = \frac{\partial J}{\partial u} = \alpha I+\frac{1}{\sigma^2}S^2V'\text{diag}(w)\text{diag}(A)V$$

-------
![alt text](flow_chart.jpg)
$$\text{Function}:J(u,\alpha,G),\chi^2(u,G)$$

$$J(u^{\text{opt}}_i,\alpha_i,G)=0\\
\Longrightarrow \frac{\partial u^{\text{opt}}_i}{\partial G} = -(\frac{\partial J}{\partial u})^{-1} \frac{\partial J}{\partial G} = -H^{-1}\frac{\partial J}{\partial G}
$$

$$\frac{\partial \chi^{2~\text{opt}}_i}{\partial G} = \frac{\partial \chi^{2}}{\partial u}\frac{\partial u^{\text{opt}}_i}{\partial G}+\frac{\partial \chi^{2}}{\partial G}$$

For curve fitting:
$$\text{loss}(p) = \sum_{i=1}^L(p_1+\frac{p_2}{1+\exp(-p_3(x_i-p_4))}-y_i)^2$$


For $\alpha^{\text{opt}}$ to $u^{\text{opt}}$
$$J(u^{\text{opt}},\alpha^{\text{opt}},G)=0\\

\Longrightarrow \frac{\partial u^{\text{opt}}}{\partial G} = -(\frac{\partial J}{\partial u})^{-1} \left(\frac{\partial J}{\partial \alpha}\frac{\partial \alpha^{\text{opt}}}{\partial G}+\frac{\partial J}{\partial G} \right) 

= -H^{-1} \left(\frac{\partial J}{\partial \alpha}\frac{\partial \alpha^{\text{opt}}}{\partial G}+\frac{\partial J}{\partial G} \right)
$$

Supple differentiation:
$$\frac{\partial J}{\partial G} = -\frac{1}{\sigma^2}SU'$$

$$\frac{\partial J}{\partial \alpha} = u$$


