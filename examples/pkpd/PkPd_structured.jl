# Structured PKPD experiment using the new organized approach
using Revise, Rhythm, Lux, DifferentialEquations, Random, SciMLSensitivity, ComponentArrays, Optimisers, OptimizationOptimisers, Statistics
using MLUtils, Printf, SciMLSensitivity, OneHotArrays, CairoMakie, Distributions
using YAML

# Include organized modules
include("data/data_prep.jl")
include("experiments/forecasting/training/loss_fn.jl")
include("experiments/forecasting/training/eval_fn.jl")
include("experiments/forecasting/training/forecasting_fn.jl")
include("experiments/forecasting/training/viz_fn.jl")
include("experiments/forecasting/models/model_creator.jl")

# Set random seed for reproducibility
rng = Random.MersenneTwister(123)

println("="^60)
println("PKPD Structured Forecasting Experiment")
println("="^60)

## Configuration and Data Loading
println("Loading PKPD dataset...")

# Generate dataset using the structured approach
train_loader, val_loader, test_loader, dims, timepoints_obs, timepoints_forecast = generate_dataloader(
    n_samples=256, 
    batchsize=32, 
    split=(0.6, 0.2), 
    obs_fraction=0.5
);

println("Dataset loaded successfully!")
println("- Training batches: $(length(train_loader))")
println("- Validation batches: $(length(val_loader))")
println("- Test batches: $(length(test_loader))")
println("- Dimensions: $dims")

## Model Configuration
println("\nConfiguring models...")

# Load configurations
config_lsde = YAML.load_file("configs/PkPD_config_LSDE.yml")
config_lode = YAML.load_file("configs/PkPD_config_LODE.yml")

# Set up experiment path
exp_path = joinpath(config_lsde["experiment"]["path"], config_lsde["experiment"]["name"] * "_structured")
isdir(exp_path) || mkpath(exp_path)

## Latent SDE Model Training
println("\n" * "="^40)
println("Training Latent SDE Model")
println("="^40)

# Create model using structured approach
lsde_model, lsde_θ, lsde_st = create_pkpd_latentsde(config_lsde["model"], dims, rng)

# Train model using new loss and eval functions
lsde_θ_trained = train(
    lsde_model, lsde_θ, lsde_st, 
    vcat(timepoints_obs, timepoints_forecast),
    loss_fn_nde, eval_fn_nde, viz_fn_forecast_pkpd,
    train_loader, val_loader, 
    config_lsde["training"], exp_path
)

println("Latent SDE training completed!")

## Latent ODE Model Training
println("\n" * "="^40) 
println("Training Latent ODE Model")
println("="^40)

# Create model
lode_model, lode_θ, lode_st = create_pkpd_latentode(config_lode["model"], dims, rng)

# Train model
lode_θ_trained = train(
    lode_model, lode_θ, lode_st,
    vcat(timepoints_obs, timepoints_forecast),
    loss_fn_nde, eval_fn_nde, viz_fn_forecast_pkpd,
    train_loader, val_loader,
    config_lode["training"], exp_path
)

println("Latent ODE training completed!")

## Evaluation and Visualization
println("\n" * "="^60)
println("EVALUATION AND VISUALIZATION")
println("="^60)

# Get test data
data_obs, data_forecast = test_loader.data

## Latent SDE Evaluation
println("\nEvaluating Latent SDE...")
lsde_forecasted_μ, lsde_forecasted_σ = forecast_nde(
    lsde_model, lsde_θ_trained, lsde_st, 
    data_obs, data_forecast[1], timepoints_forecast,
    config_lsde["training"]["validation"]
)
lsde_forecasted_data = (lsde_forecasted_μ, lsde_forecasted_σ)

# Generate visualization and get metrics
lsde_fig = viz_fn_forecast_pkpd(
    timepoints_obs, timepoints_forecast, 
    data_obs, data_forecast, lsde_forecasted_data; 
    sample_n=4
)

# Save figure
save(joinpath(exp_path, "lsde_forecast_structured.png"), lsde_fig)
println("Latent SDE visualization saved!")

## Latent ODE Evaluation
println("\nEvaluating Latent ODE...")
lode_forecasted_μ, lode_forecasted_σ = forecast_nde(
    lode_model, lode_θ_trained, lode_st,
    data_obs, data_forecast[1], timepoints_forecast,
    config_lode["training"]["validation"]
)
lode_forecasted_data = (lode_forecasted_μ, lode_forecasted_σ)

# Generate visualization
lode_fig = viz_fn_forecast_pkpd(
    timepoints_obs, timepoints_forecast,
    data_obs, data_forecast, lode_forecasted_data;
    sample_n=4
)

# Save figure
save(joinpath(exp_path, "lode_forecast_structured.png"), lode_fig)
println("Latent ODE visualization saved!")

## Performance Comparison
println("\n" * "="^40)
println("PERFORMANCE COMPARISON")
println("="^40)

# Calculate detailed metrics for both models
lsde_rmse, lsde_crps = viz_fn_forecast_pkpd(
    timepoints_obs, timepoints_forecast,
    data_obs, data_forecast, lsde_forecasted_data; 
    sample_n=1, plot=false
)

lode_rmse, lode_crps = viz_fn_forecast_pkpd(
    timepoints_obs, timepoints_forecast,
    data_obs, data_forecast, lode_forecasted_data;
    sample_n=1, plot=false  
)

# Display results
variables_of_interest = ["Tumor Volume", "Health Score"]
println("Performance Summary:")
println("-"^30)
for (i, var_name) in enumerate(variables_of_interest)
    println("$var_name:")
    println("  Latent SDE - RMSE: $(round(lsde_rmse[i], digits=4)), CRPS: $(round(lsde_crps[i], digits=4))")
    println("  Latent ODE - RMSE: $(round(lode_rmse[i], digits=4)), CRPS: $(round(lode_crps[i], digits=4))")
    println()
end

println("="^60)
println("STRUCTURED EXPERIMENT COMPLETED SUCCESSFULLY!")
println("Results saved to: $exp_path")
println("="^60)
