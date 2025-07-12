**Flow of SSK**

1. Initialize the configuration
2. Find the best $\theta$ of Monte Carlo(MC)/Simulated Annealing(SA)
3. Do MC/SA and measure configuration every givn steps.
4. Average all measured configurations to get the final spectrum and calculate $Re(A(w+i\eta))$
5. Get the poles from $Re(A(w+i\eta))$

Detailed steps:
2: For a series of $\theta$, get their corresponding $\chi^2$ with small-scaled MC. Then decide a good $\theta$ by `chi2kink` or `chi2min`.
4. What we get is a distribution but not a exact configuration.
5. This means we can't directly get the poles. We get poles by extracting peaks from the chart of $w\to Re(A(w+i\eta))$.

Of course it's possible that the number of peaks of $Re(A(w+i\eta))$ is not equal to the number of poles we set. If this happens, we just choose the last configuration.

**Improvement**

1. Amplitudes of different poles can be different
2. When average configurations, we should give configuration with different weights according to their $\chi^2$