# K-fold cross-validation forecasting experiment for PKPD models
using Rhythm, Lux, DifferentialEquations, Random, SciMLSensitivity, ComponentArrays,
    Optimisers, OptimizationOptimisers, Statistics, MLUtils, Printf, OneHotArrays,
    CairoMakie, Distributions, YAML

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
data, train_loader, val_loader, test_loader, dims, ts_obs, ts_for, normalization_stats =
    load_dataset(; n_samples=512, batchsize=32, split=(0.6, 0.1), obs_fraction=0.5, normalization=false, seed=123);
variables_of_interest = ["Health Score", "Tumor Volume", "Cancer cell count"];

k_folds = 2
timepoints = (ts_obs, ts_for);
sample_n = 5

# Latent SDE K-Fold Training
lsde_models, lsde_params, lsde_states, lsde_performances, _ =
    kfold_train(data, dims, k_folds, rng, joinpath(@__DIR__, "configs/PkPD_config_lsde.yml"), "lsde", timepoints,
        loss_fn, eval_fn, forecast, viz_fn);

lsde_stats = assess_model_performance(lsde_performances, variables_of_interest;
    model_name="Latent SDE", forecast_fn=forecast,
    plot_sample=true, sample_n=sample_n, viz_fn=viz_fn,
    models=lsde_models, params=lsde_params, states=lsde_states,
    data=data, normalization_stats=normalization_stats, timepoints=timepoints,
    config_path=joinpath(@__DIR__, "configs/PkPD_config_lsde.yml"));

# Latent ODE K-Fold Training
lode_models, lode_params, lode_states, lode_performances, _ =
    kfold_train(data, dims, k_folds, rng, joinpath(@__DIR__, "configs/PkPD_config_lode.yml"), "lode", timepoints,
        loss_fn, eval_fn, forecast, viz_fn);

lode_stats = assess_model_performance(lode_performances, variables_of_interest;
    model_name="Latent ODE", forecast_fn=forecast,
    plot_sample=true, sample_n=sample_n, viz_fn=viz_fn,
    models=lode_models, params=lode_params, states=lode_states,
    data=data, normalization_stats=normalization_stats, timepoints=timepoints,
    config_path=joinpath(@__DIR__, "configs/PkPD_config_lode.yml"));

# Latent LSTM K-Fold Training
lstm_models, lstm_params, lstm_states, lstm_performances, _ =
    kfold_train(data, dims, k_folds, rng, joinpath(@__DIR__, "configs/PkPD_config_latent_lstm.yml"), "latent_lstm", timepoints,
        loss_fn, eval_fn, forecast, viz_fn);

lstm_stats = assess_model_performance(lstm_performances, variables_of_interest;
    model_name="Latent LSTM", forecast_fn=forecast,
    plot_sample=true, sample_n=sample_n, viz_fn=viz_fn,
    models=lstm_models, params=lstm_params, states=lstm_states,
    data=data, normalization_stats=normalization_stats, timepoints=timepoints,
    config_path=joinpath(@__DIR__, "configs/PkPD_config_latent_lstm.yml"));

# Latent CDE K-Fold Training
lcde_models, lcde_params, lcde_states, lcde_performances, _ =
    kfold_train(data, dims, k_folds, rng, joinpath(@__DIR__, "configs/PkPD_config_lcde.yml"), "latent_cde", timepoints,
        loss_fn, eval_fn, forecast, viz_fn);

lcde_stats = assess_model_performance(lcde_performances, variables_of_interest;
    model_name="Latent CDE", forecast_fn=forecast,
    plot_sample=true, sample_n=sample_n, viz_fn=viz_fn,
    models=lcde_models, params=lcde_params, states=lcde_states,
    data=data, normalization_stats=normalization_stats, timepoints=timepoints,
    config_path=joinpath(@__DIR__, "configs/PkPD_config_lcde.yml"));

# Compare all models
model_comparison = compare_pkpd_models(
    Dict("Latent SDE" => lsde_stats, "Latent ODE" => lode_stats, "Latent LSTM" => lstm_stats, "Latent CDE" => lcde_stats),
    sort_by="overall");
