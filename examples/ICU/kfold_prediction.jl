##dependencies
using Revise, Rhythm, Lux, DifferentialEquations, Random, SciMLSensitivity, ComponentArrays, Optimisers, OptimizationOptimisers, Statistics
using MLUtils, Printf, SciMLSensitivity, OneHotArrays, CairoMakie, Distributions
using YAML
using DataFrames, CSV
include("data_prep.jl")
include("kfold_trainer.jl")

# Set random seed for reproducibility
rng = Random.MersenneTwister(123);

# Load data
variables_of_interest = ["MAP", "HR", "Temp"]
n_features = length(variables_of_interest)
data, train_loader, val_loader, test_loader, time_series_dataset = load_data(
    split_at=24, 
    n_samples=256, 
    batch_size=32, 
    variables_of_interest=variables_of_interest
);

# Setup timepoints
n_timepoints = size(hcat(data[2], data[6]))[2]
tspan = (1.0, n_timepoints)
timepoints = (range(tspan[1], tspan[2], length=n_timepoints)) / 10 |> Array{Float32};

# Define loss function (same as in Prediction.jl)
function loss_fn(model, θ, st, data)
    (u_obs, x_obs, y_obs, masks_obs, u_for, x_for, y_for, masks_for), ts, λ = data
    batch_size= size(y_for)[end]
    ŷ, px₀, kl_pq = model(x_obs, hcat(u_obs, u_for), ts, θ, st)
    recon_loss = 0.0f0
    for i in eachindex(ŷ)
        μ, log_σ² = ŷ[i][1], ŷ[i][2]
        valid_indx= findall(masks_for[i, :, :] .== 1)
        recon_loss += normal_loglikelihood(μ[1,valid_indx], log_σ²[1,valid_indx], y_for[i, valid_indx])/batch_size
    end 
    kl_loss = kl_normal(px₀...) / batch_size + mean(kl_pq[end, :]) 
    loss = recon_loss + λ * kl_loss
    return loss, st, (kl_loss, recon_loss, 0.0f0, 0.0f0)
end

# Define evaluation function (same as in Prediction.jl)
function eval_fn(model, θ, st, ts, data, config)
    u_obs, x_obs, y_obs, masks_obs, u_for, x_for, y_for, masks_for = data
    batch_size= size(y_for)[end]
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    _, Ey = predict(model, solver, x_obs, hcat(u_obs,u_for), ts, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    loss=0.0f0
    for i in eachindex(Ey)
        μ, log_σ² = dropmean(Ey[i][1], dims=4), dropmean(Ey[i][2], dims=4)
        valid_indx= findall(masks_for[i, :, :] .== 1)
        loss += normal_loglikelihood(μ[1,valid_indx], log_σ²[1,valid_indx],y_for[i, valid_indx])/batch_size
    end
    return (loss, 0.0f0, 0.0f0) 
end

# Perform k-fold training with Latent SDE model
n_folds = 5
config_path = "./configs/ICU_config_lsde.yml"
model_type = "lsde"

# Perform k-fold cross-validation for LSDE
@info "Starting $n_folds-fold cross-validation for $model_type model"
lsde_models, lsde_params, lsde_states, performances = kfold_train(
    data, 
    n_folds, 
    rng, 
    config_path, 
    model_type, 
    timepoints, 
    loss_fn, 
    eval_fn, 
    viz_fn_forecast
);

# Calculate and display average performance metrics for LSDE
lsde_avg_rmse_by_feature = [mean([perf[1][i] for perf in performances]) for i in 1:n_features];
lsde_avg_crps_by_feature = [mean([perf[2][i] for perf in performances]) for i in 1:n_features];

@info "Average RMSE by feature for LSDE:"
for (i, feature) in enumerate(variables_of_interest)
    @info "  $feature: $(lsde_avg_rmse_by_feature[i])"
end

@info "Average CRPS by feature for LSDE:"
for (i, feature) in enumerate(variables_of_interest)
    @info "  $feature: $(lsde_avg_crps_by_feature[i])"
end

# Optionally, visualize the best LSDE model's forecast
lsde_best_fold_idx = argmin([mean(perf[1]) for perf in performances])
@info "Best LSDE model is from fold $lsde_best_fold_idx"

lsde_best_model = lsde_models[lsde_best_fold_idx];
lsde_best_params = lsde_params[lsde_best_fold_idx];
lsde_best_state = lsde_states[lsde_best_fold_idx];

# Test on the full test dataset
test_data = test_loader.data;
lsde_fig, lsde_rmse, lsde_crps = kfold_forecast(
    lsde_best_model, 
    lsde_best_params, 
    lsde_best_state, 
    test_data, 
    timepoints, 
    YAML.load_file(config_path), 
    viz_fn_forecast, 
    fold_idx=lsde_best_fold_idx, 
    sample_n=1,
    plot=true
);

# Optionally save the visualization
# save("examples/ICU/ICU_lsde_kfold_best_model.eps", lsde_fig)

# Perform k-fold training with Latent ODE model

@info "Starting $n_folds-fold cross-validation for LODE model"
lode_models, lode_params, lode_states, lode_performances = kfold_train(
    data, 
    n_folds, 
    rng, 
    "./configs/ICU_config_lode.yml", 
    "lode", 
    timepoints, 
    loss_fn, 
    eval_fn, 
    viz_fn_forecast
);

# Calculate and display average performance metrics for LODE
lode_avg_rmse_by_feature = [mean([perf[1][i] for perf in lode_performances]) for i in 1:n_features];
lode_avg_crps_by_feature = [mean([perf[2][i] for perf in lode_performances]) for i in 1:n_features];

@info "Average RMSE by feature for LODE:"
for (i, feature) in enumerate(variables_of_interest)
    @info "  $feature: $(lode_avg_rmse_by_feature[i])"
end

@info "Average CRPS by feature for LODE:"
for (i, feature) in enumerate(variables_of_interest)
    @info "  $feature: $(lode_avg_crps_by_feature[i])"
end

# Optionally, visualize the best LODE model's forecast
lode_best_fold_idx = argmin([mean(perf[1]) for perf in lode_performances])
@info "Best LODE model is from fold $lode_best_fold_idx"

lode_best_model = lode_models[lode_best_fold_idx];
lode_best_params = lode_params[lode_best_fold_idx];
lode_best_state = lode_states[lode_best_fold_idx];

# Test the LODE model on the full test dataset
lode_fig, lode_rmse, lode_crps = kfold_forecast(
    lode_best_model, 
    lode_best_params, 
    lode_best_state, 
    test_data, 
    timepoints, 
    YAML.load_file("./configs/ICU_config_lode.yml"), 
    viz_fn_forecast, 
    fold_idx=lode_best_fold_idx, 
    sample_n=1,
    plot=true
);

# Optionally save the visualization
# save("examples/ICU/ICU_lode_kfold_best_model.eps", lode_fig)
