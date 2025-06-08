# ACFlowSensitivity.jl

Welcome to the documentation for [ACFlowSensitivity](https://github.com/yuiyuiui/ACFlowSensitivity.jl)! It's a Julia package implementing and improving Green's function analytic continuation aalgorithms maly based on [ACFlow](https://github.com/huangli712/ACFlow), and calculate numerical differentiation of them.

## Overview

ACFlowSensitivity.jl accept values of Green' function on the imaginary axis and general a vector as reconstructed spectral density.

The high level interface of ACFlowSensitivity is provided by the following functions:
*   [`solve`](@ref): analyticly continue Green's function to the real axis and get the spectrum
*   [`solvediff`](@ref): calculate numerical differentiation of [`solve`](@ref)

## Current functionality

*   Reimplementation:

|Method|RI [`Cont`](@ref)|RI [`Delta`](@ref)|RI [`Mixed`](@ref)|
|:---|:---|:---|:---|
|[`BarRat`](@ref)|✅|||
|[`MaxEntChi2kink`](@ref)|✅|❌|❌|


*   Improvement: [`MaxEntChi2kink`](@ref) with iterative model (under tests).
*   Newimplementation:
*   Differentiation:

|Method|SA [`Cont`](@ref)|SA [`Delta`](@ref)|SA [`Mixed`](@ref)|
|:---|:---|:---|:---|
|[`BarRat`](@ref)|✅|||
|[`MaxEntChi2kink`](@ref)|✅|❌|❌|
