# K-fold cross-validation forecasting experiment for PKPD models
using Revise
using Rhythm
using Lux, DifferentialEquations, Random, SciMLSensitivity, ComponentArrays, Optimisers, OptimizationOptimisers, Statistics
using MLUtils, Printf, SciMLSensitivity, OneHotArrays, CairoMakie, Distributions
using YAML
# Set random seed for reproducibility

rng = Random.MersenneTwister(123);

# Include necessary files
include("../../data/data_prep.jl");
include("../../data/data_utils.jl");
include("training/loss_fn.jl");
include("training/eval_fn.jl");
include("training/forecasting_fn.jl");
include("training/viz_fn.jl");
include("training/kfold_trainer.jl");

# loading data
data, train_loader, val_loader, test_loader, dims, ts_obs, ts_for, normalization_stats = generate_dataloader(; n_samples=1024, split=(0.6, 0.2), obs_fraction=0.5, normalization=false);

variables_of_interest = ["Health Score", "Tumor Volume", "Cancer cell count"];
k_folds = 5 # Number of folds for cross-validation

# LSDE K-Fold Training
config_lsde_path = "/Volumes/Mine/Academic/PhD/Codes/Packages/Rhythm.jl/examples/pkpd/configs/PkPD_config_lsde.yml";
lsde_models, lsde_params, lsde_states, lsde_performances = kfold_train(data, dims, k_folds, rng, config_lsde_path, "lsde", ts_for,
                                                                                loss_fn_nde, eval_fn_nde, forecast_nde, viz_fn);

lsde_stats = assess_model_performance(lsde_performances, variables_of_interest; model_name="Latent SDE", forecast_fn=forecast_nde,
                                         plot_sample=true, sample_n=11, viz_fn=viz_fn, models=lsde_models, params=lsde_params, states=lsde_states,
                                         data=data, normalization_stats, timepoints=(ts_obs, ts_for),
                                         config=YAML.load_file(config_lsde_path)["training"]["validation"]);

# LODE K-Fold Training
config_lode_path = "/Volumes/Mine/Academic/PhD/Codes/Packages/Rhythm.jl/examples/pkpd/configs/PkPD_config_lode.yml";
lode_models, lode_params, lode_states, lode_performances = kfold_train(data, dims, k_folds, rng, config_lode_path, "lode", ts_for,
                                                                        loss_fn_nde, eval_fn_nde, forecast_nde, viz_fn);

lode_stats = assess_model_performance(lode_performances, variables_of_interest; model_name="Latent ODE", forecast_fn=forecast_nde,
                                        plot_sample=true, sample_n=11, viz_fn=viz_fn, models=lode_models, params=lode_params, states=lode_states, data=data, normalization_stats,
                                        timepoints=(ts_obs, ts_for), config=YAML.load_file(config_lode_path)["training"]["validation"]);

# Latent LSTM K-Fold Training
config_latent_lstm_path =  "/Volumes/Mine/Academic/PhD/Codes/Packages/Rhythm.jl/examples/pkpd/configs/PkPD_config_latent_lstm.yml";
latent_lstm_models, latent_lstm_params, latent_lstm_states, latent_lstm_performances = kfold_train(data, dims, k_folds, rng, config_latent_lstm_path, "latent_lstm", ts_for,
                                                                                                    loss_fn_lstm, eval_fn_lstm, forecast_lstm, viz_fn);

latent_lstm_stats = assess_model_performance(latent_lstm_performances, variables_of_interest; model_name="Latent LSTM", forecast_fn=forecast_lstm,
                                             plot_sample=true, sample_n=6, viz_fn=viz_fn, models=latent_lstm_models, params=latent_lstm_params,
                                             states=latent_lstm_states, data=data, normalization_stats, timepoints=(ts_obs, ts_for),
                                             config=YAML.load_file(config_latent_lstm_path)["training"]["validation"]);

# Compare multiple models
model_comparison = compare_pkpd_models(Dict("Latent SDE" => lsde_stats,"Latent ODE" => lode_stats, "Latent LSTM" => latent_lstm_stats), sort_by="overall");