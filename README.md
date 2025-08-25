# ACFlowSensitivity
<!-- 
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://yuiyuiui.github.io/ACFlowSensitivity.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://yuiyuiui.github.io/ACFlowSensitivity.jl/dev/)
[![Build Status](https://github.com/yuiyuiui/ACFlowSensitivity.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/yuiyuiui/ACFlowSensitivity.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/yuiyuiui/ACFlowSensitivity.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/yuiyuiui/ACFlowSensitivity.jl)
-->

`ACFlowSensitivity` is a Julia package that reimplements and enhances analytic continuation methods from [`ACFlow`](https://github.com/huangli712/ACFlow). It provides tools for analyzing the sensitivity of these algorithms using automatic differentiation and other advanced techniques.

For input $\mathcal{G}=\{G(iw_n)\}_{n=1}^N$ and analytic continuation algorithm

![formula](https://latex.codecogs.com/svg.image?f%3A%5Cmathbb%7BC%7D%5EN%5Cto%5Cmathbb%7BR%7D%5EM%2C%5Cmathcal%7BG%7D%5Cmapsto%5Cwidetilde%7BA%7D%3D%5C%7B%5Cwidetilde%7BA%7D_j%5C%7D_%7Bj%3D1%7D%5EM)

Here $\text{reA}$ means the reconstructed spenctral density function and $w_j$ is the integral weight of the output mesh.

As a result we calculate:

![Gradient formula](https://latex.codecogs.com/svg.image?\nabla%20f(\mathcal{G})=\left(\frac{\partial\widetilde{A}_j}{\partial\mathcal{G}_k}\right)_{M\times%20N})

Our purpose is to implement following methods (not all) and their sensitivity analysis (`RI` means reimplement. `SA` means Sensitivity Analysis for both fermionic and bosonic systems (Only fermionic now). `cont,delta,mixed` are spectrum types):

|Method|RI cont|RI delta|RI mixed|SA cont|SA delta|SA mixed|
|:---|:---|:---|:---|:---|:---|:---|
|BarRat|✅|✅||✅|✅|
|MaxEnt Chi2kink|✅|✅||✅|✅||
|MaxEnt Bryan|✅|✅||✅|✅||
|MaxEnt Classic|✅|✅||✅|✅||
|MaxEnt Historic|✅|✅|||||
|SSK|✅|✅||✅|✅||
|SAC|✅|✅||✅|✅||
|SPX|✅|✅||✅|✅||
|SOM|✅|✅||✅|✅||
|NAC|✅|✅||✅|✅||

Example of using `ACFlowSensitivity` to plot the error bound of a specific method is shown in the file folder `plot`.

By using `ACFlowSensitivity`, you can get a error bound like this:

![Error Bound Example 1](./plot/maxent/chi2kink/maxent_cont_eb_2.svg)
![Error Bound Example 2](./plot/maxent/chi2kink/maxent_cont_bsc_eb.svg)

