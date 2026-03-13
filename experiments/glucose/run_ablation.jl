# Ablation study for Glucose Latent SDE model
# Tests: (1) No context (context_dim=0), (2) No control inputs (u zeroed out)
using Rhythm, Lux, DifferentialEquations, Random, SciMLSensitivity, ComponentArrays,
    Optimisers, OptimizationOptimisers, Statistics, MLUtils, Printf, CairoMakie,
    Distributions, YAML, NNlib

rng = Random.MersenneTwister(124);

# Include data and training files
include("data/data_generation.jl");
include("data/data_utils.jl");
include("training/losses.jl");
include("training/evaluation.jl");
include("training/forecast.jl");
include("training/visualization.jl");
include("training/trainer.jl");

# Load data
data, dims, timepoints, normalization_stats =
    load_dataset(; n_samples=512, obs_fraction=0.5, normalization=true, seed=123);
variables_of_interest = ["Glucose"];
k_folds = 5

# ─────────────────────────────────────────────────────────────────────────────
# Baseline: Full Latent SDE (for comparison)
# ─────────────────────────────────────────────────────────────────────────────
lsde_models, lsde_params, lsde_states, lsde_performances, _ =
    kfold_train(data, dims, k_folds, rng, joinpath(@__DIR__, "configs/glucose_config_lsde.yml"), "lsde", timepoints,
        loss_fn, eval_fn, forecast, viz_fn);

lsde_stats = assess_model_performance(lsde_performances, variables_of_interest;
    model_name="Latent SDE (Full)", forecast_fn=forecast,
    plot_sample=true, sample_n=4, viz_fn=viz_fn,
    models=lsde_models, params=lsde_params, states=lsde_states,
    data=data, normalization_stats=normalization_stats, timepoints=timepoints,
    config_path=joinpath(@__DIR__, "configs/glucose_config_lsde.yml"));

# ─────────────────────────────────────────────────────────────────────────────
# Ablation 1: No context (context_dim=0)
# Removes the encoder's temporal context from the SDE drift_aug.
# ─────────────────────────────────────────────────────────────────────────────
nc_models, nc_params, nc_states, nc_performances, _ =
    kfold_train(data, dims, k_folds, rng, joinpath(@__DIR__, "configs/glucose_config_lsde_no_context.yml"), "lsde", timepoints,
        loss_fn, eval_fn, forecast, viz_fn);

nc_stats = assess_model_performance(nc_performances, variables_of_interest;
    model_name="SDE No Context", forecast_fn=forecast,
    plot_sample=true, sample_n=4, viz_fn=viz_fn,
    models=nc_models, params=nc_params, states=nc_states,
    data=data, normalization_stats=normalization_stats, timepoints=timepoints,
    config_path=joinpath(@__DIR__, "configs/glucose_config_lsde_no_context.yml"));

# ─────────────────────────────────────────────────────────────────────────────
# Ablation 2: No control inputs (zero out meal and insulin signals)
# The dynamics cannot see meals/insulin — tests whether model uses inputs.
# ─────────────────────────────────────────────────────────────────────────────
data_no_ctrl = Tuple(i in (1, 6) ? zero(d) : d for (i, d) in enumerate(data));

nctrl_models, nctrl_params, nctrl_states, nctrl_performances, _ =
    kfold_train(data_no_ctrl, dims, k_folds, rng, joinpath(@__DIR__, "configs/glucose_config_lsde.yml"), "lsde", timepoints,
        loss_fn, eval_fn, forecast, viz_fn);

nctrl_stats = assess_model_performance(nctrl_performances, variables_of_interest;
    model_name="SDE No Control", forecast_fn=forecast,
    plot_sample=true, sample_n=4, viz_fn=viz_fn,
    models=nctrl_models, params=nctrl_params, states=nctrl_states,
    data=data_no_ctrl, normalization_stats=normalization_stats, timepoints=timepoints,
    config_path=joinpath(@__DIR__, "configs/glucose_config_lsde.yml"));

# ─────────────────────────────────────────────────────────────────────────────
# Compare ablations
# ─────────────────────────────────────────────────────────────────────────────
model_comparison = compare_glucose_models(
    Dict("SDE (Full)" => lsde_stats, "SDE No Context" => nc_stats,
        "SDE No Control" => nctrl_stats),
    sort_by="overall");
