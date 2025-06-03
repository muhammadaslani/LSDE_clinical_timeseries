# K-fold cross-validation forecasting experiment for PKPD models
using Revise, Rhythm, Lux, DifferentialEquations, Random, SciMLSensitivity, ComponentArrays, Optimisers, OptimizationOptimisers, Statistics
using MLUtils, Printf, SciMLSensitivity, OneHotArrays, CairoMakie, Distributions
using YAML

# Include necessary files
include("../../data/data_prep.jl");
include("../../data/data_utils.jl");
include("training/loss_fn.jl");
include("training/eval_fn.jl");
include("training/forecasting_fn.jl");
include("training/viz_fn.jl");
include("training/kfold_trainer.jl");
include("models/model_creator.jl");

# Set random seed for reproducibility
rng = Random.MersenneTwister(123);
## Configuration
variables_of_interest = [ "Health Score", "Tumor Volume", "Cancer cell count" ];
k_folds = 2

# Load model configurations
config_lsde_path = "/Volumes/Mine/Academic/PhD/Codes/Packages/Rhythm.jl/examples/pkpd/configs/PkPD_config_lsde.yml";
config_lode_path = "/Volumes/Mine/Academic/PhD/Codes/Packages/Rhythm.jl/examples/pkpd/configs/PkPD_config_lode.yml";

# loading data
data, train_loader, val_loader, test_loader, dims, timepoints_obs, timepoints_forecast = generate_dataloader(; n_samples=256, batchsize=32, split=(0.6,0.2), obs_fraction=0.5);

# LSDE K-Fold Training
model_type= "lsde"

lsde_models, lsde_params, lsde_states, lsde_performances = kfold_train_pkpd(
    data,
    dims,
    k_folds,
    rng,
    config_lsde_path,
    model_type, 
    timepoints_forecast,
    loss_fn_nde,
    eval_fn_nde,
    forecast_nde,
    viz_fn_nde,
);

# Find best performing models (based on average performance across variables)
# Extract the three specific metrics
lsde_crossentropy_health = [perf[1] for perf in lsde_performances]
lsde_rmse_tumor = [perf[2] for perf in lsde_performances] 
lsde_nll_count = [perf[3] for perf in lsde_performances]

# Calculate averages
lsde_avg_crossentropy_health = mean(lsde_crossentropy_health)
lsde_avg_rmse_tumor = mean(lsde_rmse_tumor);
lsde_avg_nll_count = mean(lsde_nll_count);

lsde_fig = assess_model_performance(
    lsde_performances, variables_of_interest;
    model_name="Latent SDE",
    model_type="lsde",
    forecast_fn=forecast_nde,
    plot_sample=true,
    sample_n=3,
    viz_fn=viz_fn_nde,
    models=lsde_models,
    params=lsde_params, 
    states=lsde_states,
    data=data,
    timepoints=(timepoints_obs, timepoints_forecast),
    config=YAML.load_file(config_lsde_path)["training"]["validation"],
    best_fold_idx=1
)
# LODE K-Fold Training


lode_performances, lode_models, lode_params, lode_states = kfold_train_pkpd(
    lode_model_creator,
    loss_fn_nde,
    eval_fn_nde, 
    viz_fn_forecast_pkpd,
    create_pkpd_dataloader,
    config_lode,
    exp_path_lode;
    k_folds=k_folds,
    variables_of_interest=variables_of_interest
)

# Generate test data for visualization
test_data = test_loader.data;

lode_avg_rmse = [mean([perf[1][i] for i in 1:length(variables_of_interest)]) for perf in lode_performances]

best_lsde_idx = argmin(lsde_avg_rmse)
best_lode_idx = argmin(lode_avg_rmse)

println("Best performing folds:")
println("- Latent SDE: Fold $best_lsde_idx (Average RMSE: $(round(lsde_avg_rmse[best_lsde_idx], digits=4)))")
println("- Latent ODE: Fold $best_lode_idx (Average RMSE: $(round(lode_avg_rmse[best_lode_idx], digits=4)))")

## Detailed Performance Assessment
println("\n" * "="^40)
println("LATENT SDE PERFORMANCE")
println("="^40)



if !isnothing(lsde_fig)
    save(joinpath(exp_path_lsde, "lsde_kfold_best_forecast.png"), lsde_fig)
    println("Best Latent SDE forecast saved!")
end

println("\n" * "="^40)
println("LATENT ODE PERFORMANCE") 
println("="^40)

lode_fig = assess_model_performance(
    lode_performances, variables_of_interest;
    model_name="Latent ODE", 
    model_type="lode",
    forecast_fn=forecast_nde,
    plot_sample=true,
    sample_n=3,
    viz_fn=viz_fn_forecast_pkpd,
    models=lode_models,
    params=lode_params,
    states=lode_states, 
    data=test_data,
    timepoints=(timepoints_obs, timepoints_forecast),
    config=config_lode["training"]["validation"],
    best_fold_idx=best_lode_idx
)

if !isnothing(lode_fig)
    save(joinpath(exp_path_lode, "lode_kfold_best_forecast.png"), lode_fig)
    println("Best Latent ODE forecast saved!")
end

## Model Comparison
println("\n" * "="^40)
println("MODEL COMPARISON")
println("="^40)

# Calculate overall statistics
lsde_rmse_all = [perf[1] for perf in lsde_performances]
lsde_crps_all = [perf[2] for perf in lsde_performances]
lode_rmse_all = [perf[1] for perf in lode_performances]  
lode_crps_all = [perf[2] for perf in lode_performances]

println("Model Comparison Summary:")
println("-"^30)
for (i, var_name) in enumerate(variables_of_interest)
    lsde_rmse_vals = [rmse[i] for rmse in lsde_rmse_all]
    lsde_crps_vals = [crps[i] for crps in lsde_crps_all]
    lode_rmse_vals = [rmse[i] for rmse in lode_rmse_all]
    lode_crps_vals = [crps[i] for crps in lode_crps_all]
    
    println("$var_name:")
    println("  Latent SDE - RMSE: $(round(mean(lsde_rmse_vals), digits=4)) ± $(round(std(lsde_rmse_vals), digits=4))")
    println("              CRPS: $(round(mean(lsde_crps_vals), digits=4)) ± $(round(std(lsde_crps_vals), digits=4))")
    println("  Latent ODE - RMSE: $(round(mean(lode_rmse_vals), digits=4)) ± $(round(std(lode_rmse_vals), digits=4))")
    println("              CRPS: $(round(mean(lode_crps_vals), digits=4)) ± $(round(std(lode_crps_vals), digits=4))")
    println()
end

## Save Results
println("Saving results...")

# Save performance data
using JLD2
@save joinpath(exp_path_lsde, "lsde_kfold_results.jld2") lsde_performances lsde_models lsde_params lsde_states
@save joinpath(exp_path_lode, "lode_kfold_results.jld2") lode_performances lode_models lode_params lode_states

println("="^60)
println("K-FOLD CROSS-VALIDATION EXPERIMENT COMPLETED!")
println("Results saved to:")
println("- Latent SDE: $exp_path_lsde") 
println("- Latent ODE: $exp_path_lode")
println("="^60)
