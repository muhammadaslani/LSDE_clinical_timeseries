##dependencies
using Revise, Rhythm, Lux, DifferentialEquations, Random, SciMLSensitivity, ComponentArrays, Optimisers, OptimizationOptimisers, Statistics
using MLUtils, Printf, SciMLSensitivity, OneHotArrays, CairoMakie, Distributions
using YAML
using DataFrames, CSV

# Include files using relative paths from current directory
include("../../data/data_prep.jl");
include("training/kfold_trainer.jl");
include("training/forecasting_fn.jl");
include("training/loss_fn.jl");
include("training/eval_fn.jl");
include("training/viz_fn.jl");
include("models/latent_lstm.jl");

# Set random seed for reproducibility
rng = Random.MersenneTwister(123);

# Load data
variables_of_interest = ["MAP", "HR", "Temp"];
n_features = length(variables_of_interest);
data, train_loader, val_loader, test_loader, time_series_dataset = load_data(
    split_at=24, 
    n_samples=512, 
    batch_size=32, 
    variables_of_interest=variables_of_interest
);

# Setup timepoints
n_timepoints = size(hcat(data[2], data[6]))[2];
tspan = (1.0, n_timepoints);
timepoints = (range(tspan[1], tspan[2], length=n_timepoints)) / 20 |> Array{Float32};

# Perform k-fold training with Latent SDE model
n_folds = 2
config_path_lsde = joinpath(@__DIR__, "..", "..", "configs", "ICU_config_lsde.yml");
model_type_lsde = "lsde"

# Perform k-fold cross-validation for LSDE
@info "Starting $n_folds-fold cross-validation for LSDE model"
lsde_models, lsde_params, lsde_states, lsde_performances = kfold_train(
    data, 
    n_folds, 
    rng, 
    config_path_lsde, 
    model_type_lsde, 
    timepoints, 
    loss_fn_nde, 
    eval_fn_nde, 
    forecast_nde,
    viz_fn_forecast_nde
);

# Present LSDE model performance with sample plot
lsde_stats = assess_model_performance(lsde_performances, variables_of_interest, model_name="Latent SDE", forecast_fn=forecast_nde,
                           plot_sample=true, sample_n=3, viz_fn=viz_fn_forecast_nde, models=lsde_models, params=lsde_params, 
                           states=lsde_states, data=test_loader.data, timepoints=timepoints, 
                           config=YAML.load_file(config_path_lsde));
# Perform k-fold training with Latent ODE model
config_path_lode = joinpath(@__DIR__, "..", "..", "configs", "ICU_config_lode.yml");
model_type_lode = "lode"
@info "Starting $n_folds-fold cross-validation for LODE model"
lode_models, lode_params, lode_states, lode_performances = kfold_train(
    data, 
    n_folds, 
    rng, 
    config_path_lode,
    model_type_lode, 
    timepoints, 
    loss_fn_nde, 
    eval_fn_nde, 
    forecast_nde,
    viz_fn_forecast_nde
);

# Present LODE model performance with sample plot
lode_stats = assess_model_performance(lode_performances, variables_of_interest, model_name="Latent ODE", forecast_fn=forecast_nde,
                           plot_sample=true, sample_n=3,viz_fn=viz_fn_forecast_nde, models=lode_models, params=lode_params, 
                           states=lode_states, data=test_loader.data, timepoints=timepoints, 
                           config=YAML.load_file(config_path_lode));


# RNN model training and evaluation
rnn_config_path = joinpath(@__DIR__, "..", "..", "configs", "ICU_config_latent_lstm.yml");

# Perform k-fold cross-validation for RNN
@info "Starting $n_folds-fold cross-validation for latent lstm model"
rnn_models, rnn_params, rnn_states, rnn_performances = kfold_train(
    data, 
    n_folds, 
    rng, 
    rnn_config_path, 
    model_type_rnn,
    timepoints, 
    loss_fn_rnn, 
    eval_fn_rnn, 
    forecast_rnn,
    viz_fn_forecast_nde
);

# Present RNN model performance with sample plot
latent_lstm_stats = assess_model_performance(rnn_performances, variables_of_interest, model_name="Latent LSTM", forecast_fn=forecast_rnn,
                           plot_sample=true, sample_n=3, viz_fn=viz_fn_forecast_nde, models=rnn_models, params=rnn_params, 
                           states=rnn_states, data=test_loader.data, timepoints=timepoints, 
                           config=YAML.load_file(rnn_config_path));


# Compare RNN model with others
model_comparison_rnn = compare_models(
    Dict("Latent SDE" => lsde_stats, "Latent ODE" => lode_stats, "Latent LSTM" => latent_lstm_stats),
    sort_by="rmse",  # Sort by RMSE (can also use "crps")
    ascending=true   # Best models first (lowest values)
);

