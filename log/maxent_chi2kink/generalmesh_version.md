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