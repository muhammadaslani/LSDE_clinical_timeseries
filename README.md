# Generative Modeling of Clinical Time Series via Latent Stochastic Differential Equations

This repository contains the code for the paper:

> **Generative Modeling of Clinical Time Series via Latent Stochastic Differential Equations**
> Ahmed ElGazzar

## Overview

We present a framework for generative modeling of irregularly-sampled clinical time series using latent stochastic differential equations (SDEs). The model learns continuous-time latent dynamics via a variational autoencoder with neural SDE priors, trained with an evidence lower bound (ELBO) objective using path-wise KL divergence computed via Girsanov's theorem.

The framework is implemented in Julia using [Lux.jl](https://github.com/LuxDL/Lux.jl), [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl), and [SciMLSensitivity.jl](https://github.com/SciML/SciMLSensitivity.jl).

## Repository Structure

```
.
├── src/                    # Core library (Rhythm.jl)
│   ├── core/               # Model definitions (LatentSDE, LatentODE, LatentCDE, LatentLSTM)
│   └── utils/              # Training, losses, visualization utilities
├── experiments/
│   ├── pkpd/               # Tumor growth (PK/PD) experiments
│   ├── glucose/            # Bergman glucose-insulin experiments
│   └── icu/                # PhysioNet ICU 2012 experiments
├── test/                   # Unit tests
└── docs/                   # Documentation
```

Each experiment directory contains:
- `configs/` — YAML configuration files for each model variant
- `data/` — Data generation (synthetic) or loading (ICU) scripts
- `training/` — Training, evaluation, and visualization modules
- `run_benchmarks.jl` — Main entry point for running all model comparisons
- `run_ablation.jl` — Ablation study entry point (glucose only)

## Installation

**Requirements:** Julia 1.6.7 or later.

```bash
# Clone the repository
git clone https://github.com/elgazzarr/Rhythm.jl.git
cd Rhythm.jl

# Install dependencies
julia --project -e 'using Pkg; Pkg.instantiate()'
```

## Running Experiments

Each experiment can be run as a standalone Julia script:

```bash
# PK/PD tumor growth — all models (Latent SDE, ODE, CDE, LSTM)
julia --project experiments/pkpd/run_benchmarks.jl

# Glucose-insulin — all models
julia --project experiments/glucose/run_benchmarks.jl

# PhysioNet ICU 2012 — all models
julia --project experiments/icu/run_benchmarks.jl
```

**Ablation study** (no-context and no-control variants, glucose dataset):

```bash
julia --project experiments/glucose/run_ablation.jl
```

Alternatively, use the provided `Makefile`:

```bash
make pkpd          # Run PK/PD benchmarks
make glucose        # Run glucose benchmarks
make icu            # Run ICU benchmarks
make ablation       # Run glucose ablation study
make all            # Run everything
```

## Models

| Model | Description |
|-------|-------------|
| **Latent SDE** | Neural SDE with learned drift, diffusion, and augmented posterior drift |
| **Latent ODE** | Deterministic ODE baseline (neural ODE in latent space) |
| **Latent CDE** | Controlled differential equation baseline |
| **Latent LSTM** | Recurrent baseline with LSTM dynamics |

## Datasets

| Dataset | Type | Description |
|---------|------|-------------|
| **PK/PD** | Synthetic | Tumor growth under chemotherapy and radiotherapy (6-month simulations) |
| **Glucose** | Synthetic | Bergman minimal glucose-insulin model with meal/insulin inputs (12-hour simulations) |
| **ICU** | Real | PhysioNet Challenge 2012 — irregularly-sampled ICU vital signs |

## Citation

If you find this code useful, please cite:

```bibtex
@article{elgazzar2025latentsde,
  title={Generative Modeling of Clinical Time Series via Latent Stochastic Differential Equations},
  author={ElGazzar, Ahmed},
  year={2025}
}
```

## License

This project is licensed under the MIT License — see [LICENSE.md](LICENSE.md) for details.
