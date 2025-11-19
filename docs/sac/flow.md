1. Set Default model $m(w)$. In ACFlow, it claims that: Only flat model is valid for the StochAC solver.
2. Denote $$x = \phi(w) = \int_{-\infty}^{w} m(w') dw'$$
$$n(x) = \frac{A(\phi^{-1}(x))}{m(\phi^{-1}(x))}$$

3. Denote $$H[n(x)] = \int_0^{\beta}d\tau \frac{1}{\sigma(\tau)^2} \left|\int_0^1 dx K(\tau, x) n(x) - G(x)\right|^2$$

4. Sample $n(x)$ by Monte Carlo just like `ssk`. Then do a average over $n(x)$:
$$\left<n(x)\right> = \frac{1}{Z}\int Dn ~n(x) e^{-\alpha H[n]}$$

5. Recover $A(w)$:
$$\left<A(w)\right> = \left<n(\phi(w))\right>m(w)$$

6. Improvement: do average over configurations of MC: $$\left<\left<n(x)\right>\right> = ...$$

7. Shortage: Although we introduce the smooth item, it's practical effect is bad.

Even for delta type, it's effect is far worse than `ssk`.


