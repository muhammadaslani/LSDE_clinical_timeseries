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
include("models/model_creator.jl");


## Configuration
variables_of_interest = [ "Health Score", "Tumor Volume", "Cancer cell count" ];
k_folds = 2 # Number of folds for cross-validation

# loading data
data, train_loader, val_loader, test_loader, dims, timepoints_obs, timepoints_forecast = generate_dataloader(; n_samples=512, batchsize=32, split=(0.6,0.2), obs_fraction=0.4);

# LSDE K-Fold Training
model_type_lsde, config_lsde_path = "lsde", "/Volumes/Mine/Academic/PhD/Codes/Packages/Rhythm.jl/examples/pkpd/configs/PkPD_config_lsde.yml";
config = YAML.load_file(config_lsde_path)
lsde_model = create_latentsde(config["model"], dims, rng);
lsde_models, lsde_params, lsde_states, lsde_performances = kfold_train_pkpd(data, dims, k_folds, rng, config_lsde_path, model_type_lsde, timepoints_forecast, 
                                                                            loss_fn_nde, eval_fn_nde, forecast_nde, viz_fn_nde);

lsde_stats = assess_model_performance(lsde_performances, variables_of_interest; model_name="Latent SDE", model_type="lsde", forecast_fn=forecast_nde, plot_sample=true, 
                                        sample_n=1, viz_fn=viz_fn_nde, models=lsde_models, params=lsde_params, states=lsde_states, data=data, 
                                        timepoints=(timepoints_obs, timepoints_forecast), config=YAML.load_file(config_lsde_path)["training"]["validation"]);

# LODE K-Fold Training
model_type_lode, config_lode_path = "lode", "/Volumes/Mine/Academic/PhD/Codes/Packages/Rhythm.jl/examples/pkpd/configs/PkPD_config_lode.yml";
config = YAML.load_file(config_lode_path);
lode_model, θ, st = create_latentode(config["model"], dims, rng);



lode_models, lode_params, lode_states, lode_performances = kfold_train_pkpd(data, dims, k_folds, rng, config_lode_path, model_type_lode, timepoints_forecast, 
                                                                            loss_fn_nde, eval_fn_nde, forecast_nde, viz_fn_nde);

lode_stats = assess_model_performance(lode_performances, variables_of_interest; model_name="Latent ODE", model_type="lode", forecast_fn=forecast_nde, plot_sample=true,
                                         sample_n=1, viz_fn=viz_fn_nde, models=lode_models, params=lode_params, states=lode_states, data=data, 
                                         timepoints=(timepoints_obs, timepoints_forecast), config=YAML.load_file(config_lode_path)["training"]["validation"]);

# RNN K-Fold Training
model_type_rnn, config_rnn_path = "rnn", "/Volumes/Mine/Academic/PhD/Codes/Packages/Rhythm.jl/examples/pkpd/configs/PkPD_config_rnn.yml";
rnn_models, rnn_params, rnn_states, rnn_performances = kfold_train_pkpd(data, dims, k_folds, rng, config_rnn_path, model_type_rnn, timepoints_forecast, 
                                                                        loss_fn_rnn, eval_fn_rnn, forecast_rnn, viz_fn_nde);

rnn_stats = assess_model_performance(rnn_performances, variables_of_interest; model_name="RNN", model_type="rnn", forecast_fn=forecast_rnn, plot_sample=true,
                                     sample_n=4, viz_fn=viz_fn_nde, models=rnn_models, params=rnn_params, states=rnn_states, data=data,
                                     timepoints=(timepoints_obs, timepoints_forecast), config=YAML.load_file(config_rnn_path)["training"]["validation"]);


# Compare multiple models
model_comparison = compare_pkpd_models(Dict("LSDE" => lsde_stats, "RNN" => rnn_stats), sort_by="overall");