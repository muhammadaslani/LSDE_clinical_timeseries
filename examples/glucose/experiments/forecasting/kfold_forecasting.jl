# K-fold cross-validation forecasting experiment for Glucose models
using Revise
using Rhythm
using Lux, DifferentialEquations, Random, SciMLSensitivity, ComponentArrays, Optimisers, OptimizationOptimisers, Statistics
using MLUtils, Printf, CairoMakie, Distributions
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
    generate_dataloader(; n_samples=128, batchsize=16, split=(0.6, 0.2), obs_fraction=0.5, normalization=true, seed=123);
variables_of_interest = ["Glucose"];
k_folds = 2

# Latent SDE K-Fold Training
config_lsde_path = joinpath(@__DIR__, "../../configs/glucose_config_lsde.yml");
lsde_models, lsde_params, lsde_states, lsde_performances =
    kfold_train(data, dims, k_folds, rng, config_lsde_path, "lsde", ts_for,
        loss_fn, eval_fn, forecast_nde, viz_fn);

lsde_stats = assess_model_performance(lsde_performances, variables_of_interest;
    model_name="Latent SDE", forecast_fn=forecast_nde,
    plot_sample=true, sample_n=3, viz_fn=viz_fn,
    models=lsde_models, params=lsde_params, states=lsde_states,
    data=data, normalization_stats=normalization_stats, timepoints=(ts_obs, ts_for),
    config=YAML.load_file(config_lsde_path)["training"]["validation"]);

# Latent ODE K-Fold Training
config_lode_path = joinpath(@__DIR__, "../../configs/glucose_config_lode.yml");
lode_models, lode_params, lode_states, lode_performances =
    kfold_train(data, dims, k_folds, rng, config_lode_path, "lode", ts_for,
        loss_fn, eval_fn, forecast_nde, viz_fn);

lode_stats = assess_model_performance(lode_performances, variables_of_interest;
    model_name="Latent ODE", forecast_fn=forecast_nde,
    plot_sample=true, sample_n=1, viz_fn=viz_fn,
    models=lode_models, params=lode_params, states=lode_states,
    data=data, normalization_stats=normalization_stats, timepoints=(ts_obs, ts_for),
    config=YAML.load_file(config_lode_path)["training"]["validation"]);

# Latent LSTM K-Fold Training
config_lstm_path = joinpath(@__DIR__, "../../configs/glucose_config_latent_lstm.yml");
lstm_models, lstm_params, lstm_states, lstm_performances =
    kfold_train(data, dims, k_folds, rng, config_lstm_path, "latent_lstm", ts_for,
        loss_fn, eval_fn, forecast_lstm, viz_fn);

lstm_stats = assess_model_performance(lstm_performances, variables_of_interest;
    model_name="Latent LSTM", forecast_fn=forecast_lstm,
    plot_sample=true, sample_n=1, viz_fn=viz_fn,
    models=lstm_models, params=lstm_params, states=lstm_states,
    data=data, normalization_stats=normalization_stats, timepoints=(ts_obs, ts_for),
    config=YAML.load_file(config_lstm_path)["training"]["validation"]);

# Latent CDE K-Fold Training
config_lcde_path = joinpath(@__DIR__, "../../configs/glucose_config_latent_cde.yml");
lcde_models, lcde_params, lcde_states, lcde_performances =
    kfold_train(data, dims, k_folds, rng, config_lcde_path, "latent_cde", ts_for,
        loss_fn, eval_fn, forecast_cde, viz_fn; timepoints_obs=ts_obs);

lcde_stats = assess_model_performance(lcde_performances, variables_of_interest;
    model_name="Latent CDE", forecast_fn=forecast_cde,
    plot_sample=true, sample_n=1, viz_fn=viz_fn,
    models=lcde_models, params=lcde_params, states=lcde_states,
    data=data, normalization_stats=normalization_stats, timepoints=(ts_obs, ts_for),
    config=YAML.load_file(config_lcde_path)["training"]["validation"]);

# Compare all models
model_comparison = compare_glucose_models(
    Dict("Latent SDE" => lsde_stats, "Latent ODE" => lode_stats, "Latent LSTM" => lstm_stats, "Latent CDE" => lcde_stats),
    sort_by="overall");
