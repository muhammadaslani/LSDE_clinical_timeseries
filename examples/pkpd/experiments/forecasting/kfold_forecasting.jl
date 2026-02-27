# K-fold cross-validation forecasting experiment for PKPD models
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
    generate_dataloader(; n_samples=512, batchsize=32, split=(0.6, 0.1), obs_fraction=0.5, normalization=false, seed=123);
plot_tumor_and_treatment(data, ts_obs, patients=1:9);
variables_of_interest = ["Health Score", "Tumor Volume", "Cancer cell count"];

k_folds = 2
timepoints = (ts_obs, ts_for);

# Latent SDE K-Fold Training
config_lsde_path = joinpath(@__DIR__, "../../configs/PkPD_config_lsde.yml");
lsde_models, lsde_params, lsde_states, lsde_performances =
    kfold_train(data, dims, k_folds, rng, config_lsde_path, "lsde", timepoints,
        loss_fn, eval_fn, forecast, viz_fn);

lsde_cfg = load_config(config_lsde_path);
lsde_stats = assess_model_performance(lsde_performances, variables_of_interest;
    model_name="Latent SDE", forecast_fn=forecast,
    plot_sample=true, sample_n=1, viz_fn=vis_fn,
    models=lsde_models, params=lsde_params, states=lsde_states,
    data=data, normalization_stats=normalization_stats, timepoints=timepoints,
    config=merge(lsde_cfg["model"]["validation"], lsde_cfg["training"]["validation"]));

# Latent ODE K-Fold Training
config_lode_path = joinpath(@__DIR__, "../../configs/PkPD_config_lode.yml");
lode_models, lode_params, lode_states, lode_performances =
    kfold_train(data, dims, k_folds, rng, config_lode_path, "lode", timepoints,
        loss_fn, eval_fn, forecast, viz_fn);

lode_cfg = load_config(config_lode_path);
lode_stats = assess_model_performance(lode_performances, variables_of_interest;
    model_name="Latent ODE", forecast_fn=forecast,
    plot_sample=true, sample_n=3, viz_fn=vis_fn,
    models=lode_models, params=lode_params, states=lode_states,
    data=data, normalization_stats=normalization_stats, timepoints=timepoints,
    config=merge(lode_cfg["model"]["validation"], lode_cfg["training"]["validation"]));

# Latent LSTM K-Fold Training
config_lstm_path = joinpath(@__DIR__, "../../configs/PkPD_config_latent_lstm.yml");
lstm_models, lstm_params, lstm_states, lstm_performances =
    kfold_train(data, dims, k_folds, rng, config_lstm_path, "latent_lstm", timepoints,
        loss_fn, eval_fn, forecast, viz_fn);

lstm_cfg = load_config(config_lstm_path);
lstm_stats = assess_model_performance(lstm_performances, variables_of_interest;
    model_name="Latent LSTM", forecast_fn=forecast,
    plot_sample=true, sample_n=4, viz_fn=viz_fn,
    models=lstm_models, params=lstm_params, states=lstm_states,
    data=data, normalization_stats=normalization_stats, timepoints=timepoints,
    config=merge(lstm_cfg["model"]["validation"], lstm_cfg["training"]["validation"]));

# Latent CDE K-Fold Training
config_lcde_path = joinpath(@__DIR__, "../../configs/PkPD_config_lcde.yml");
lcde_models, lcde_params, lcde_states, lcde_performances =
    kfold_train(data, dims, k_folds, rng, config_lcde_path, "latent_cde", timepoints,
        loss_fn, eval_fn, forecast, viz_fn);

lcde_cfg = load_config(config_lcde_path);
lcde_stats = assess_model_performance(lcde_performances, variables_of_interest;
    model_name="Latent CDE", forecast_fn=forecast,
    plot_sample=true, sample_n=4, viz_fn=viz_fn,
    models=lcde_models, params=lcde_params, states=lcde_states,
    data=data, normalization_stats=normalization_stats, timepoints=timepoints,
    config=merge(lcde_cfg["model"]["validation"], lcde_cfg["training"]["validation"]));

# Compare all models
model_comparison = compare_pkpd_models(
    Dict("Latent SDE" => lsde_stats, "Latent ODE" => lode_stats),
    sort_by="overall");
