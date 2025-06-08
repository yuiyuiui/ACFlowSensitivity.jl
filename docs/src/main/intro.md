# Introduction

```@contents
Pages = ["main/intro.md", "main/solve.md", "main/barrat.md", "main/maxent.md", "main/math.md","main/generatedata.md","main/mesh.md","main/model.md"]
Depth = 2
```

## Installing

Since ACFlowSensitivity.jl is not yet registered in the Julia package registry, you need to clone the repository locally and install it using `Pkg.develop`:

```julia
using Pkg
Pkg.develop(path="path/to/ACFlowSensitivity.jl")
```

Alternatively, you can install directly from the GitHub repository:

```julia
using Pkg
Pkg.develop(url="https://github.com/yuiyuiui/ACFlowSensitivity.jl")
```

ACFlowSensitivity.jl is a pure Julia package; no dependencies (aside from the Julia standard library) are required.

## Getting started

After installation, start by loading `ACFlowSensitivity`

```julia
using ACFlowSensitivity
```
