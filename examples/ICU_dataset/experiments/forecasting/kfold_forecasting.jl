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

# Set random seed for reproducibility
rng = Random.MersenneTwister(123);

# Load data
variables_of_interest = ["MAP", "HR"];
n_features = length(variables_of_interest);
data, train_loader, val_loader, test_loader, time_series_dataset, normalization_stats = load_data(
    split_at=50, 
    n_samples=512, 
    batch_size=64, 
    variables_of_interest=variables_of_interest, 
    normalization=true
);

# Setup timepoints
n_timepoints = size(hcat(data[2], data[6]))[2];
tspan = (1.0, n_timepoints);
timepoints = Array(tspan[1]:tspan[2]) / (n_timepoints+1) |> Array{Float32};
n_folds = 2

# Perform k-fold training with Latent SDE model
config_path_lsde = joinpath(@__DIR__, "..", "..", "configs", "ICU_config_lsde.yml");
lsde_models, lsde_params, lsde_states, lsde_performances = kfold_train(
    data, 
    n_folds, 
    rng, 
    config_path_lsde, 
    "lsde", 
    timepoints, 
    loss_fn_nde, 
    eval_fn_nde, 
    forecast_nde,
    viz_fn
);

# Present LSDE model performance with sample plot
lsde_stats = assess_model_performance(lsde_performances, variables_of_interest, model_name="Latent SDE", forecast_fn=forecast_nde,
                           plot_sample=true, sample_n=2, viz_fn=viz_fn, models=lsde_models, params=lsde_params, 
                           states=lsde_states, data=data, normalization_stats=normalization_stats, timepoints=timepoints, 
                           config=YAML.load_file(config_path_lsde));

# Perform k-fold training with Latent ODE model
config_path_lode = joinpath(@__DIR__, "..", "..", "configs", "ICU_config_lode.yml");
lode_models, lode_params, lode_states, lode_performances = kfold_train(
    data, 
    n_folds, 
    rng, 
    config_path_lode,
    "lode", 
    timepoints, 
    loss_fn_nde, 
    eval_fn_nde, 
    forecast_nde,
    viz_fn
);

# Present LODE model performance with sample plot
lode_stats = assess_model_performance(lode_performances, variables_of_interest, model_name="Latent ODE", forecast_fn=forecast_nde,
                                        plot_sample=true, sample_n=1,viz_fn=viz_fn, models=lode_models, params=lode_params, 
                                        states=lode_states, data=data,normalization_stats=normalization_stats, timepoints=timepoints, 
                                        config=YAML.load_file(config_path_lode));

# Latent LSTM model training and evaluation
lstm_config_path = joinpath(@__DIR__, "..", "..", "configs", "ICU_config_latent_lstm.yml");
lstm_models, lstm_params, lstm_states, lstm_performances = kfold_train(
    data, 
    n_folds, 
    rng, 
    lstm_config_path, 
    "latent_lstm",
    timepoints, 
    loss_fn_lstm, 
    eval_fn_lstm, 
    forecast_lstm,
    viz_fn
);

# Present LSTM model performance with sample plot
latent_lstm_stats = assess_model_performance(lstm_performances, variables_of_interest, model_name="Latent LSTM", forecast_fn=forecast_lstm,
                           plot_sample=true, sample_n=1, viz_fn=viz_fn, models=lstm_models, params=lstm_params, 
                           states=lstm_states, data=data, normalization_stats=normalization_stats, timepoints=timepoints, 
                           config=YAML.load_file(lstm_config_path));


# Compare models
model_comparison_lstm = compare_models(
    Dict("Latent SDE" => lsde_stats, "Latent ODE" => lode_stats, "Latent LSTM" => latent_lstm_stats),
    sort_by="rmse", 
    ascending=true  
);

