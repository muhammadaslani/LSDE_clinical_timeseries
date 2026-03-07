# Ablation study for PKPD Latent SDE model
# Tests: (1) No context (context_dim=0), (2) No control inputs (u zeroed out)
using Revise
using Rhythm
using Lux, DifferentialEquations, Random, SciMLSensitivity, ComponentArrays, Optimisers, OptimizationOptimisers, Statistics
using MLUtils, Printf, SciMLSensitivity, OneHotArrays, CairoMakie, Distributions
using YAML

rng = Random.MersenneTwister(124);

# Include data and training files
include("../../data/data_prep.jl");
include("../../data/data_utils.jl");
include("training/loss_fn.jl");
include("training/eval_fn.jl");
include("training/forecasting_fn.jl");
include("training/viz_fn.jl");
include("training/kfold_trainer.jl");

# Load data
data, train_loader, val_loader, test_loader, dims, ts_obs, ts_for, normalization_stats =
    generate_dataloader(; n_samples=512, batchsize=32, split=(0.6, 0.1), obs_fraction=0.4, normalization=false, seed=123);
variables_of_interest = ["Health Score", "Tumor Volume", "Cancer cell count"];

k_folds = 2
timepoints = (ts_obs, ts_for);

# ─────────────────────────────────────────────────────────────────────────────
# Baseline: Full Latent SDE (for comparison)
# ─────────────────────────────────────────────────────────────────────────────
config_lsde_path = joinpath(@__DIR__, "../../configs/PkPD_config_lsde.yml");
lsde_models, lsde_params, lsde_states, lsde_performances =
    kfold_train(data, dims, k_folds, rng, config_lsde_path, "lsde", timepoints,
        loss_fn, eval_fn, forecast, viz_fn);

lsde_cfg = load_config(config_lsde_path);
lsde_stats = assess_model_performance(lsde_performances, variables_of_interest;
    model_name="Latent SDE (Full)", forecast_fn=forecast,
    plot_sample=true, sample_n=5, viz_fn=vis_fn,
    models=lsde_models, params=lsde_params, states=lsde_states,
    data=data, normalization_stats=normalization_stats, timepoints=timepoints,
    config=merge(lsde_cfg["model"]["validation"], lsde_cfg["training"]["validation"]));

# ─────────────────────────────────────────────────────────────────────────────
# Ablation 1: No context (context_dim=0)
# Removes the encoder's temporal context from the SDE drift_aug.
# ─────────────────────────────────────────────────────────────────────────────
config_nc_path = joinpath(@__DIR__, "../../configs/PkPD_config_lsde_no_context.yml");
nc_models, nc_params, nc_states, nc_performances =
    kfold_train(data, dims, k_folds, rng, config_nc_path, "lsde", timepoints,
        loss_fn, eval_fn, forecast, viz_fn);

nc_cfg = load_config(config_nc_path);
nc_stats = assess_model_performance(nc_performances, variables_of_interest;
    model_name="SDE No Context", forecast_fn=forecast,
    plot_sample=true, sample_n=5, viz_fn=vis_fn,
    models=nc_models, params=nc_params, states=nc_states,
    data=data, normalization_stats=normalization_stats, timepoints=timepoints,
    config=merge(nc_cfg["model"]["validation"], nc_cfg["training"]["validation"]));

# ─────────────────────────────────────────────────────────────────────────────
# Ablation 2: No control inputs (zero out treatment signals)
# The dynamics cannot see drug treatment — tests whether model uses treatment.
# ─────────────────────────────────────────────────────────────────────────────
data_no_ctrl = Tuple(i in (1, 8) ? zero(d) : d for (i, d) in enumerate(data));

nctrl_models, nctrl_params, nctrl_states, nctrl_performances =
    kfold_train(data_no_ctrl, dims, k_folds, rng, config_lsde_path, "lsde", timepoints,
        loss_fn, eval_fn, forecast, viz_fn);

nctrl_cfg = load_config(config_lsde_path);
nctrl_stats = assess_model_performance(nctrl_performances, variables_of_interest;
    model_name="SDE No Control", forecast_fn=forecast,
    plot_sample=true, sample_n=5, viz_fn=vis_fn,
    models=nctrl_models, params=nctrl_params, states=nctrl_states,
    data=data_no_ctrl, normalization_stats=normalization_stats, timepoints=timepoints,
    config=merge(nctrl_cfg["model"]["validation"], nctrl_cfg["training"]["validation"]));

# ─────────────────────────────────────────────────────────────────────────────
# Compare ablations
# ─────────────────────────────────────────────────────────────────────────────
model_comparison = compare_pkpd_models(
    Dict("SDE (Full)" => lsde_stats, "SDE No Context" => nc_stats, "SDE No Control" => nctrl_stats),
    sort_by="overall");
