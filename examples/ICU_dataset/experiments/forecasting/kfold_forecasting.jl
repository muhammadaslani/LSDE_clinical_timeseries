# K-fold cross-validation forecasting experiment for ICU models
using Revise
using Rhythm
using Lux, DifferentialEquations, Random, SciMLSensitivity, ComponentArrays, Optimisers, OptimizationOptimisers, Statistics
using MLUtils, Printf, SciMLSensitivity, OneHotArrays, CairoMakie, Distributions
using YAML
using DataFrames, CSV

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
split_at = 24
variables_of_interest = ["MAP", "HR", "Temp"];
data, _, _, _, _, normalization_stats =
    load_data(; split_at=split_at, n_samples=512, batch_size=32,
               variables_of_interest=variables_of_interest, normalization=true);

# Setup timepoints as (obs, forecast) tuple — normalised to (0,1] so dt=0.01 ≈ 100 solver steps
n_obs   = split_at
n_for   = size(data[7], 2)
n_total = n_obs + n_for
ts_obs  = Float32.(1:n_obs)          ./ n_total   # obs window
ts_for  = Float32.(n_obs+1:n_total)  ./ n_total   # forecast window
timepoints = (ts_obs, ts_for);

# Compute dims from data
dims = Dict(
    "input_dim"  => size(data[1], 1),
    "obs_dim"    => size(data[2], 1),
    "output_dim" => ones(Int, size(data[7], 1)),
)


k_folds = 2

# Latent SDE K-Fold Training
config_lsde_path = joinpath(@__DIR__, "../../configs/ICU_config_lsde.yml");
lsde_models, lsde_params, lsde_states, lsde_performances =
    kfold_train(data, dims, k_folds, rng, config_lsde_path, "lsde", timepoints,
        loss_fn, eval_fn, forecast, viz_fn);

lsde_cfg = load_config(config_lsde_path);
lsde_stats = assess_model_performance(lsde_performances, variables_of_interest;
    model_name="Latent SDE", forecast_fn=forecast,
    plot_sample=true, sample_n=1, viz_fn=viz_fn,
    models=lsde_models, params=lsde_params, states=lsde_states,
    data=data, normalization_stats=normalization_stats, timepoints=timepoints,
    config=merge(get(lsde_cfg["model"], "validation", Dict()), lsde_cfg["training"]["validation"]));

# Latent ODE K-Fold Training
config_lode_path = joinpath(@__DIR__, "../../configs/ICU_config_lode.yml");
lode_models, lode_params, lode_states, lode_performances =
    kfold_train(data, dims, k_folds, rng, config_lode_path, "lode", timepoints,
        loss_fn, eval_fn, forecast, viz_fn);

lode_cfg = load_config(config_lode_path);
lode_stats = assess_model_performance(lode_performances, variables_of_interest;
    model_name="Latent ODE", forecast_fn=forecast,
    plot_sample=true, sample_n=1, viz_fn=viz_fn,
    models=lode_models, params=lode_params, states=lode_states,
    data=data, normalization_stats=normalization_stats, timepoints=timepoints,
    config=merge(get(lode_cfg["model"], "validation", Dict()), lode_cfg["training"]["validation"]));

# Latent LSTM K-Fold Training
config_lstm_path = joinpath(@__DIR__, "../../configs/ICU_config_latent_lstm.yml");
lstm_models, lstm_params, lstm_states, lstm_performances =
    kfold_train(data, dims, k_folds, rng, config_lstm_path, "latent_lstm", timepoints,
        loss_fn, eval_fn, forecast, viz_fn);

lstm_cfg = load_config(config_lstm_path);
lstm_stats = assess_model_performance(lstm_performances, variables_of_interest;
    model_name="Latent LSTM", forecast_fn=forecast,
    plot_sample=true, sample_n=1, viz_fn=viz_fn,
    models=lstm_models, params=lstm_params, states=lstm_states,
    data=data, normalization_stats=normalization_stats, timepoints=timepoints,
    config=merge(get(lstm_cfg["model"], "validation", Dict()), lstm_cfg["training"]["validation"]));

# Latent CDE K-Fold Training
config_lcde_path = joinpath(@__DIR__, "../../configs/ICU_config_lcde.yml");
lcde_models, lcde_params, lcde_states, lcde_performances =
    kfold_train(data, dims, k_folds, rng, config_lcde_path, "latent_cde", timepoints,
        loss_fn, eval_fn, forecast, viz_fn);

lcde_cfg = load_config(config_lcde_path);
lcde_stats = assess_model_performance(lcde_performances, variables_of_interest;
    model_name="Latent CDE", forecast_fn=forecast,
    plot_sample=true, sample_n=9, viz_fn=viz_fn,
    models=lcde_models, params=lcde_params, states=lcde_states,
    data=data, normalization_stats=normalization_stats, timepoints=timepoints,
    config=merge(get(lcde_cfg["model"], "validation", Dict()), lcde_cfg["training"]["validation"]));

# Compare all models
model_comparison = compare_models(
    Dict("Latent SDE"  => lsde_stats, "Latent ODE"  => lode_stats,
         "Latent LSTM" => lstm_stats,  "Latent CDE"  => lcde_stats),
    sort_by="rmse");
