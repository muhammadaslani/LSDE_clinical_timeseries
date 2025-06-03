# PKPD Dataset - Structured Implementation

This directory contains a structured implementation of PKPD (Pharmacokinetic-Pharmacodynamic) modeling and forecasting, organized similarly to the ICU dataset approach.

## Directory Structure

```
pkpd/
├── data/
│   └── data_prep.jl                    # Data generation and preprocessing functions
├── configs/
│   ├── PkPD_config_LSDE.yml           # Configuration for Latent SDE models
│   └── PkPD_config_LODE.yml           # Configuration for Latent ODE models
├── experiments/
│   ├── forecasting/
│   │   ├── 1fold_forecast.jl          # Single fold forecasting experiment
│   │   ├── kfold_forecasting.jl       # K-fold cross-validation experiment
│   │   ├── training/
│   │   │   ├── loss_fn.jl             # Loss functions for different models
│   │   │   ├── eval_fn.jl             # Evaluation functions
│   │   │   ├── forecasting_fn.jl      # Forecasting functions
│   │   │   ├── viz_fn.jl              # Visualization functions
│   │   │   └── kfold_trainer.jl       # K-fold training utilities
│   │   ├── models/
│   │   │   └── model_creator.jl       # Model creation functions
│   │   └── results/                   # Generated results and figures
│   └── sys_id/
│       ├── training/                  # System identification training
│       ├── models/                    # System identification models
│       └── results/                   # System identification results
├── PkPd_structured.jl                 # Main structured experiment file
├── PkPd_latent_SDE.jl                # Original experiment file (updated)
└── README.md                          # This file
```

## Key Features

### 1. Data Generation (`data/data_prep.jl`)
- Synthetic PKPD data generation with realistic patient covariates
- Tumor growth modeling with treatment effects
- Health score evolution
- Data preprocessing and batching utilities

### 2. Model Types
- **Latent SDE**: Stochastic differential equation models for capturing uncertainty
- **Latent ODE**: Deterministic differential equation models
- **RNN**: Recurrent neural network models (framework ready)

### 3. Training Infrastructure
- Modular loss functions for different model types
- Comprehensive evaluation metrics (RMSE, CRPS)
- K-fold cross-validation support
- Professional visualization tools

### 4. Experiment Types
- **Single Fold**: Quick model training and evaluation
- **K-Fold**: Robust performance assessment with cross-validation
- **System Identification**: Parameter estimation and model discovery

## Usage Examples

### Quick Start - Single Fold Experiment
```julia
include("experiments/forecasting/1fold_forecast.jl")
```

### Comprehensive Evaluation - K-Fold Cross-Validation
```julia
include("experiments/forecasting/kfold_forecasting.jl")
```

### Custom Experiment - Using Structured Components
```julia
include("PkPd_structured.jl")
```

## PKPD Model Details

### Output Variables
1. **Tumor Volume**: Continuous variable representing tumor size (cm³)
2. **Health Score**: Continuous variable representing patient health status

### Input Variables
- **Chemotherapy**: Binary treatment indicator
- **Radiotherapy**: Binary treatment indicator
- **Patient Covariates**: Age, gender, weight, height, tumor type

### Model Features
- Realistic tumor growth dynamics with carrying capacity
- Treatment effects with diminishing returns
- Patient-specific parameter variations based on covariates
- Irregular observation patterns
- Measurement noise and missing data

## Performance Metrics

### For Neural DE Models (LSDE, LODE)
- **RMSE**: Root Mean Square Error for point predictions
- **CRPS**: Continuous Ranked Probability Score for probabilistic predictions

### For RNN Models
- **RMSE**: Root Mean Square Error for point predictions

## Configuration

Model configurations are stored in YAML files in the `configs/` directory:
- Hyperparameters (learning rates, batch sizes, etc.)
- Architecture specifications
- Training schedules
- Solver settings

## Visualization

The visualization system provides:
- Time series plots with observation and forecast periods
- Confidence intervals for predictions
- Treatment intervention indicators
- Professional color schemes for publications
- Performance metric displays

## Extensions

The modular structure makes it easy to:
- Add new model types
- Implement custom loss functions
- Create new visualization styles
- Add additional performance metrics
- Extend to new experimental designs

## Dependencies

- Rhythm.jl (main neural differential equations framework)
- Lux.jl (neural network framework)
- DifferentialEquations.jl (ODE/SDE solvers)
- CairoMakie.jl (plotting)
- MLUtils.jl (machine learning utilities)
- YAML.jl (configuration files)

## Notes

This structured implementation maintains compatibility with the original PKPD experiments while providing a more organized and extensible framework for research and development.
