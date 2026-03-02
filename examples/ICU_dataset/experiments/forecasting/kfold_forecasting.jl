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
target_variables = ["MAP", "HR", "Temp"];
data, _, _, _, _, normalization_stats =
    load_data(; split_at=split_at, n_samples=512, batch_size=32,
        target_variables=target_variables, normalization=true);

# Setup timepoints as (obs, forecast) tuple — normalised to (0,1] so dt=0.01 ≈ 100 solver steps
n_obs = split_at
n_for = size(data[8], 2)   # y_fut is index 8
n_total = n_obs + n_for
ts_obs = Float32.(1:n_obs) ./ n_total   # obs window
ts_for = Float32.(n_obs+1:n_total) ./ n_total   # forecast window
timepoints = (ts_obs, ts_for);

# Compute dims from data
# Tuple: (x_hist[1], u_hist[2], y_hist[3], y_masks_hist[4], x_hist_masks[5],
#          x_fut[6],  u_fut[7],  y_fut[8],  y_masks_fut[9],  x_fut_masks[10])
# obs_dim is size(x_hist)+size(x_hist) due to mask concatenation inside loss/eval fns
dims = Dict(
    "input_dim" => size(data[2], 1),                     # u_hist
    "obs_dim" => size(data[1], 1) + size(data[1], 1),  # x_hist + x_masks (same rows)
    "output_dim" => ones(Int, size(data[3], 1)),          # y_hist
)


k_folds = 2

# Latent SDE K-Fold Training
config_lsde_path = joinpath(@__DIR__, "../../configs/ICU_config_lsde.yml");
lsde_models, lsde_params, lsde_states, lsde_performances =
    kfold_train(data, dims, k_folds, rng, config_lsde_path, "lsde", timepoints,
        loss_fn, eval_fn, forecast, viz_fn);

lsde_cfg = load_config(config_lsde_path);
lsde_stats = assess_model_performance(lsde_performances, target_variables;
    model_name="Latent SDE", forecast_fn=forecast,
    plot_sample=true, sample_n=13, viz_fn=viz_fn,
    models=lsde_models, params=lsde_params, states=lsde_states,
    data=data, normalization_stats=normalization_stats, timepoints=timepoints,
    config=merge(get(lsde_cfg["model"], "validation", Dict()), lsde_cfg["training"]["validation"]));

# Latent ODE K-Fold Training
config_lode_path = joinpath(@__DIR__, "../../configs/ICU_config_lode.yml");
lode_models, lode_params, lode_states, lode_performances =
    kfold_train(data, dims, k_folds, rng, config_lode_path, "lode", timepoints,
        loss_fn, eval_fn, forecast, viz_fn);

lode_cfg = load_config(config_lode_path);
lode_stats = assess_model_performance(lode_performances, target_variables;
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
lstm_stats = assess_model_performance(lstm_performances, target_variables;
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
lcde_stats = assess_model_performance(lcde_performances, target_variables;
    model_name="Latent CDE", forecast_fn=forecast,
    plot_sample=true, sample_n=9, viz_fn=viz_fn,
    models=lcde_models, params=lcde_params, states=lcde_states,
    data=data, normalization_stats=normalization_stats, timepoints=timepoints,
    config=merge(get(lcde_cfg["model"], "validation", Dict()), lcde_cfg["training"]["validation"]));

# Compare all models
model_comparison = compare_models(
    Dict("Latent SDE" => lsde_stats, "Latent ODE" => lode_stats),
    sort_by="rmse");
