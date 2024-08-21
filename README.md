# Rhythm

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://elgazzarr.github.io/Rhythm.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://elgazzarr.github.io/Rhythm.jl/dev/)
[![Build Status](https://github.com/elgazzarr/Rhythm.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/elgazzarr/Rhythm.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/elgazzarr/Rhythm.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/elgazzarr/Rhythm.jl)


Rhythm is library for scalable inference, simulation, and control of complex dynamical systems. 

Given observations $y_{1:T}$ and optional control inputs $u_{1:T}$, Rhythm can be used to: 

1. **Filtering**: Infer the latent states $p(x_{1:T}|y_{1:T},u_{1:T})$.
2. **Prediction**: Predict future states $p(x_{T+1:T+H}|y_{1:T},u_{1:T})$ or observations $p(y_{T+1:T+H}|y_{1:T},u_{1:T})$.
3. **Learning** Estimate the parameters of the model $p(\theta|y_{1:T},u_{1:T})$.
4. **Control** Optimize the control inputs $u_{1:T}$ to optimize a given objective function. 
5. **Generation** Generate new samples from the model $ \sim p(x_{1:T},y_{1:T}|u_{1:T})$.


This range of capabilities is achieved through one class of models: Latent FLows.



