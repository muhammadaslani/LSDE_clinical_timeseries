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
variables_of_interest = ["MAP", "HR", "Temp"];
n_features = length(variables_of_interest);
data, train_loader, val_loader, test_loader, time_series_dataset = load_data(
    split_at=24, 
    n_samples=256, 
    batch_size=32, 
    variables_of_interest=variables_of_interest
);

# Setup timepoints
n_timepoints = size(hcat(data[2], data[6]))[2];
tspan = (1.0, n_timepoints);
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

# Define the forecast function (from Prediction.jl)
function forecast(model, θ, st, obs_data, u_forecast, time_forecast, config)
    u_obs, x_obs, y_obs, masks_obs = obs_data    
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    _, Ey = predict(model, solver, x_obs, hcat(u_obs,u_forecast), time_forecast, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    μ = [Ey[i][1] for i in eachindex(Ey)]
    σ = [sqrt.(exp.(Ey[i][2])) for i in eachindex(Ey)]
    return μ, σ
end 

# Define the visualization function (from Prediction.jl)
function viz_fn_forecast(t_obs, t_for, obs_data, future_true_data, forecasted_data; sample_n=1, plot=true)
    u_obs, x_obs, y_obs, masks_obs = obs_data
    u_for, x_for, y_for, masks_for = future_true_data
    μ, σ = forecasted_data
    t_obs = t_obs * 10 
    t_for = t_for * 10 

    y_labels = ["MAP", "HR", "BT"]
    fig = Figure(size=(1200, 600), fontsize=15)
    axes = CairoMakie.Axis[]
    rmse = []
    crps = []

    n_features = length(y_labels)  # Assuming one label per feature

    for i in 1:n_features
        y_label = y_labels[i]
        valid_indx_obs = findall(masks_obs[i, :, :] .== 1)
        valid_indx_for = findall(masks_for[i, :, :] .== 1)

        t_obs_val = t_obs[masks_obs[i, :, sample_n] .== 1]
        t_for_val = t_for[masks_for[i, :, sample_n] .== 1]

        y_obs_val = y_obs[i, valid_indx_obs]
        y_for_val = y_for[i, valid_indx_for]

        dists = Normal.(μ[i], σ[i])
        ŷ = rand.(dists)

        ŷ_mean = dropdims(mean(ŷ, dims=4), dims=4)
        ŷ_std = dropdims(std(ŷ, dims=4), dims=4)

        ŷ_mean_val = ŷ_mean[1, valid_indx_for]
        ŷ_std_val = ŷ_std[1, valid_indx_for]

        crps_ = empirical_crps(y_for[i:i, :, :], ŷ, masks_for[i:i, :, :])
        rmse_ = sqrt(MSELoss()(ŷ_mean_val, y_for_val))

        push!(crps, crps_)
        push!(rmse, rmse_)

        ŷ_ci_lower = ŷ_mean .- ŷ_std
        ŷ_ci_upper = ŷ_mean .+ ŷ_std

        if plot
            if isempty(t_obs_val)
                println("No observational data available for $y_label in sample $sample_n")
                ax = CairoMakie.Axis(fig[i, 1], xlabel="Time (hours)", ylabel=y_labels[i], xgridvisible=false, ygridvisible=false)
                continue
            elseif isempty(t_for_val)
                println("No future data available for $y_label in sample $sample_n")
                ax = CairoMakie.Axis(fig[i, 1], xlabel="Time (hours)", ylabel=y_labels[i], xgridvisible=false, ygridvisible=false)
                continue
            else
                ax = CairoMakie.Axis(fig[i, 1], xlabel="Time (hours)", ylabel=y_labels[i], xgridvisible=false, ygridvisible=false)
                push!(axes, ax)
                scatter!(ax, t_obs_val, y_obs[i, masks_obs[i, :, sample_n] .== 1, sample_n], color=:blue, label="Past Observations", markersize=10)
                lines!(ax, t_obs_val, y_obs[i, masks_obs[i, :, sample_n] .== 1, sample_n], color=(:blue, 0.4), linewidth=2, linestyle=:dot)

                scatter!(ax, t_for_val, y_for[i, masks_for[i, :, sample_n] .== 1, sample_n], color=:green, label="Future Ground Truth", markersize=10)
                lines!(ax, t_for_val, y_for[i, masks_for[i, :, sample_n] .== 1, sample_n], color=(:green, 0.4), linestyle=:dot)

                scatter!(ax, t_for_val, ŷ_mean[1, masks_for[i, :, sample_n] .== 1, sample_n], color=:red, label="Model Predictions", markersize=10)
                lines!(ax, t_for_val, ŷ_mean[1, masks_for[i, :, sample_n] .== 1, sample_n], color=(:red, 0.4), linestyle=:dot)

                band!(ax, t_for_val, ŷ_ci_lower[1, masks_for[i, :, sample_n] .== 1, sample_n], ŷ_ci_upper[1, masks_for[i, :, sample_n] .== 1, sample_n], color=:red, alpha=0.2)

                if i == 1
                    poly!(ax, [0, t_obs[end], t_obs[end], 0], [-10, -10, 500, 500], color=(:blue, 0.05), label="Observation Period (Past)")
                    poly!(ax, [t_obs[end], t_for_val[end], t_for_val[end], t_obs[end]], [-10, -10, 500, 500], color=(:red, 0.05), label="Forecasting Period (Future)")
                else
                    poly!(ax, [0, t_obs[end], t_obs[end], 0], [-10, -10, 500, 500], color=(:blue, 0.05))
                    poly!(ax, [t_obs[end], t_for_val[end], t_for_val[end], t_obs[end]], [-10, -10, 500, 500], color=(:red, 0.05))
                end

                all_y_values = vcat(
                    y_obs[i, masks_obs[i, :, sample_n] .== 1, sample_n],
                    y_for[i, masks_for[i, :, sample_n] .== 1, sample_n],
                    ŷ_mean[1, masks_for[i, :, sample_n] .== 1, sample_n],
                    ŷ_ci_lower[1, masks_for[i, :, sample_n] .== 1, sample_n],
                    ŷ_ci_upper[1, masks_for[i, :, sample_n] .== 1, sample_n]
                )

                y_min = minimum(all_y_values) - 0.1 * (maximum(all_y_values) - minimum(all_y_values))
                y_max = maximum(all_y_values) + 0.1 * (maximum(all_y_values) - minimum(all_y_values))
                ylims!(ax, y_min, y_max)

                if i == 1
                    fig[i, 2] = Legend(fig, ax, framevisible=false, halign=:left)
                end
            end
        end
    end

    if plot
        linkxaxes!(axes...)
        colgap!(fig.layout, 10)
        display(fig)
        return fig, rmse, crps
    else
        return rmse, crps
    end
end

# Define the kfold_forecast function for performance analysis
function assess_model_performance(performances, variables_of_interest; model_name="Model", model_type="lsde", forecast_fn=forecast,
                       plot_sample=false, sample_n=3, viz_fn=viz_fn_forecast, models=nothing, params=nothing, states=nothing, 
                       data=nothing, timepoints=nothing, config=nothing, best_fold_idx=nothing)
    """
    Presents model performance across k-folds by calculating and printing
    mean and standard deviation for RMSE and CRPS for each feature.
    Optionally plots a sample forecast from the best performing model.
    
    Arguments:
    - performances: Array of (rmse, crps) tuples from k-fold training
    - variables_of_interest: Array of feature names
    - model_name: Name of the model for display purposes
    - plot_sample: Boolean flag to enable sample plotting
    - models: Array of trained models (required if plot_sample=true)
    - params: Array of trained parameters (required if plot_sample=true)
    - states: Array of model states (required if plot_sample=true)
    - data: Test data for plotting (required if plot_sample=true)
    - timepoints: Time points array (required if plot_sample=true)
    - config: Model configuration (required if plot_sample=true)
    - best_fold_idx: Index of best fold (if not provided, will be determined from performances)
    """
    n_folds = length(performances)
    n_features = length(variables_of_interest)
    
    # Extract RMSE and CRPS values for each fold and feature
    rmse_values = zeros(n_folds, n_features)
    crps_values = zeros(n_folds, n_features)
    if model_type == "lsde" || model_type== "lode"
        for (fold_idx, (rmse, crps)) in enumerate(performances)
        rmse_values[fold_idx, :] = rmse
        crps_values[fold_idx, :] = crps
        end
    elseif model_type == "rnn" 
        for (fold_idx, (rmse, crps)) in enumerate(performances)
            rmse_values[fold_idx, :] = rmse[1:n_features]
            crps_values[fold_idx, :] .= 0.0f0
        end 
    end 
    
    # Calculate statistics
    rmse_means = mean(rmse_values, dims=1)[1, :]
    rmse_stds = std(rmse_values, dims=1)[1, :]
    crps_means = mean(crps_values, dims=1)[1, :]
    crps_stds = std(crps_values, dims=1)[1, :]
    
    # Print performance summary
    println("\n" * "="^60)
    println("$model_name Performance Summary ($n_folds-fold Cross-Validation)")
    println("="^60)
    
    println("\nRMSE (Root Mean Square Error):")
    println("-"^40)
    for (i, feature) in enumerate(variables_of_interest)
        @printf("%-10s: %.4f ± %.4f\n", feature, rmse_means[i], rmse_stds[i])
    end
    
    println("\nCRPS (Continuous Ranked Probability Score):")
    println("-"^40)
    for (i, feature) in enumerate(variables_of_interest)
        @printf("%-10s: %.4f ± %.4f\n", feature, crps_means[i], crps_stds[i])
    end
    
    # Overall performance (mean across features)
    overall_rmse_mean = mean(rmse_means)
    overall_rmse_std = sqrt(mean(rmse_stds.^2))  # Combined std
    overall_crps_mean = mean(crps_means)
    overall_crps_std = sqrt(mean(crps_stds.^2))  # Combined std
    
    println("\nOverall Performance (Average across features):")
    println("-"^40)
    @printf("RMSE: %.4f ± %.4f\n", overall_rmse_mean, overall_rmse_std)
    @printf("CRPS: %.4f ± %.4f\n", overall_crps_mean, overall_crps_std)
    println("="^60)
    
    # Optional sample plotting
    fig = nothing
    if plot_sample
        if any(isnothing.([models, params, states, data, timepoints, config]))
            @warn "Cannot plot sample: missing required arguments (models, params, states, data, timepoints, config)"
        else
            # Determine best fold if not provided
            if isnothing(best_fold_idx)
                best_fold_idx = argmin([mean(perf[1]) for perf in performances])  # Best based on average RMSE
            end
                        
            # Get best model components
            best_model = models[best_fold_idx]
            best_params = params[best_fold_idx]
            best_state = states[best_fold_idx]
            
            # Prepare data for forecasting
            inputs_data_obs, obs_data_obs, output_data_obs, masks_obs, 
            inputs_data_for, obs_data_for, output_data_for, masks_for = data
            
            # Split timepoints
            timepoints_obs = timepoints[1:size(obs_data_obs, 2)]
            timepoints_for = timepoints[size(obs_data_obs, 2)+1:end]
            
            # Prepare data for forecast function
            data_obs = (inputs_data_obs, obs_data_obs, output_data_obs, masks_obs)
            future_true_data = (inputs_data_for, obs_data_for, output_data_for, masks_for)
            
            # Generate forecast

            μ, σ = forecast_fn(best_model, best_params, best_state, data_obs, inputs_data_for, timepoints_for, config["training"]["validation"])
            forecasted_data = (μ, σ)
            
            # Create visualization
            fig, sample_rmse, sample_crps = viz_fn(timepoints_obs, timepoints_for, data_obs, future_true_data, forecasted_data, sample_n=sample_n, plot=true)
            
            println("\nSample number $sample_n forecast plotted for best model (fold $best_fold_idx)")
            println("Sample number $sample_n RMSE: [$(round.(sample_rmse, digits=4))]")
            println("Sample number $sample_n CRPS: [$(round.(sample_crps, digits=4))]")
        end
    end
    
    # Return the calculated statistics for further use if needed
    return (rmse_means=rmse_means, rmse_stds=rmse_stds, 
            crps_means=crps_means, crps_stds=crps_stds,
            overall_rmse_mean=overall_rmse_mean, overall_rmse_std=overall_rmse_std,
            overall_crps_mean=overall_crps_mean, overall_crps_std=overall_crps_std,
            figure=fig)
end

# Function for comparing multiple model performances
function compare_models(model_stats_dict; sort_by="rmse", ascending=true)
    """
    Compare performance of multiple models and display results in a formatted table.
    
    Arguments:
    - model_stats_dict: Dictionary with model names as keys and their stats as values
    - sort_by: Metric to sort by ("rmse" or "crps")
    - ascending: Sort in ascending order (true) or descending order (false)
    
    Returns:
    - Sorted dictionary of model statistics
    """
    
    # Validate inputs
    if isempty(model_stats_dict)
        @warn "No models provided for comparison"
        return model_stats_dict
    end
    
    # Extract model names and stats
    model_names = collect(keys(model_stats_dict))
    model_stats = collect(values(model_stats_dict))
    
    # Sort models by specified metric
    if sort_by == "rmse"
        sort_values = [stats.overall_rmse_mean for stats in model_stats]
    elseif sort_by == "crps"
        sort_values = [stats.overall_crps_mean for stats in model_stats]
    else
        @warn "Invalid sort_by parameter. Using 'rmse' as default."
        sort_values = [stats.overall_rmse_mean for stats in model_stats]
    end
    
    # Get sorted indices
    sorted_indices = sortperm(sort_values, rev=!ascending)
    sorted_names = model_names[sorted_indices]
    sorted_stats = model_stats[sorted_indices]
    
    # Print comparison table
    println("\n" * "="^70)
    println("Model Comparison Summary (sorted by $(uppercase(sort_by)))")
    println("="^70)
    @printf("%-15s | %-15s | %-15s | %-10s\n", "Model", "Avg RMSE", "Avg CRPS", "Rank")
    println("-"^70)
    
    best_rmse = minimum([stats.overall_rmse_mean for stats in model_stats])
    best_crps = minimum([stats.overall_crps_mean for stats in model_stats])
    
    for (rank, (name, stats)) in enumerate(zip(sorted_names, sorted_stats))
        # Add indicators for best performance
        rmse_indicator = stats.overall_rmse_mean ≈ best_rmse ? " ★" : ""
        crps_indicator = stats.overall_crps_mean ≈ best_crps ? " ★" : ""
        
        @printf("%-15s | %.4f±%.4f%s | %.4f±%.4f%s | %-10d\n", 
                name, 
                stats.overall_rmse_mean, stats.overall_rmse_std, rmse_indicator,
                stats.overall_crps_mean, stats.overall_crps_std, crps_indicator,
                rank)
    end
    
    println("-"^70)
    println("★ = Best performance for that metric")
    println("="^70)
    
    # Print detailed comparison
    println("\nDetailed Performance Comparison:")
    println("-"^40)
    
    # Find best model overall (could use weighted average or other criteria)
    best_overall_idx = argmin([stats.overall_rmse_mean + stats.overall_crps_mean for stats in sorted_stats])
    best_model_name = sorted_names[best_overall_idx]
    
    @info "Best overall model (lowest RMSE + CRPS): $best_model_name"
    
    # Calculate performance differences
    if length(model_stats) > 1
        println("\nPerformance differences (compared to best):")
        for (name, stats) in zip(sorted_names[2:end], sorted_stats[2:end])
            rmse_diff = stats.overall_rmse_mean - sorted_stats[1].overall_rmse_mean
            crps_diff = stats.overall_crps_mean - sorted_stats[1].overall_crps_mean
            @printf("  %s vs %s: +%.4f RMSE, +%.4f CRPS\n", 
                   name, sorted_names[1], rmse_diff, crps_diff)
        end
    end
    
    # Return sorted results
    return Dict(zip(sorted_names, sorted_stats))
end

# RNN-specific functions
function loss_fn_rnn(model, θ, st, data)
    (u_obs, x_obs, y_obs, masks_obs, u_for, x_for, y_for, masks_for), ts, λ = data
    batch_size = size(y_for)[end]
    ŷ, st = model(vcat(x_obs,u_obs), θ, st)
    recon_loss = 0.0f0
    for i in eachindex(ŷ)
        μ, log_σ² = ŷ[i][1], ŷ[i][2]
        valid_indx = findall(masks_for[i, :, :] .== 1)
        recon_loss += normal_loglikelihood(μ[valid_indx], log_σ²[valid_indx], y_for[i, valid_indx]) / batch_size
    end
    kl = 0.0f0
    return recon_loss, st, (kl, recon_loss, 0.0f0, 0.0f0)
end

function eval_fn_rnn(model, θ, st, ts, data, config)
    # For RNN, evaluation is similar to loss calculation
    u_obs, x_obs, y_obs, masks_obs, u_for, x_for, y_for, masks_for = data
    batch_size = size(y_for)[end]
    ŷ, st = model(vcat(x_obs, u_obs), θ, st)
    loss = 0.0f0
    for i in eachindex(ŷ)
        μ, log_σ² = ŷ[i][1], ŷ[i][2]
        valid_indx = findall(masks_for[i, :, :] .== 1)
        loss += normal_loglikelihood(μ[valid_indx], log_σ²[valid_indx], y_for[i, valid_indx]) / batch_size
    end
    return (loss, 0.0f0, 0.0f0)
end

function forecast_rnn(model, θ, st, obs_data, u_forecast, time_forecast, config)
    u_obs, x_obs, y_obs, masks_obs = obs_data
    ŷ, st = model(vcat(x_obs,u_obs), θ, st)
    μ = [ŷ[i][1] for i in eachindex(ŷ)]
    σ = [sqrt.(exp.(ŷ[i][2])) for i in eachindex(ŷ)]
    return μ, σ
end

# Define RNN-specific visualization function
function viz_fn_forecast_rnn(t_obs, t_for, obs_data, future_true_data, forecasted_data; sample_n=1, plot=true)
    u_obs, x_obs, y_obs, masks_obs = obs_data
    u_for, x_for, y_for, masks_for = future_true_data
    μ, σ = forecasted_data
    t_obs = t_obs * 10 
    t_for = t_for * 10 

    y_labels = ["MAP", "HR", "BT"]
    fig = Figure(size=(1200, 600), fontsize=15)
    axes = CairoMakie.Axis[]
    rmse = []
    crps = []

    n_features = length(y_labels)  # Assuming one label per feature

    for i in 1:n_features
        y_label = y_labels[i]
        valid_indx_obs = findall(masks_obs[i, :, :] .== 1)
        valid_indx_for = findall(masks_for[i, :, :] .== 1)

        t_obs_val = t_obs[masks_obs[i, :, sample_n] .== 1]
        t_for_val = t_for[masks_for[i, :, sample_n] .== 1]

        y_obs_val = y_obs[i, valid_indx_obs]
        y_for_val = y_for[i, valid_indx_for]

        # Generate predicted distributions (RNN format)
        dists = Normal.(μ[i], σ[i])
        ŷ = rand.(dists)

        ŷ_val = ŷ[valid_indx_for]
        rmse_ = sqrt(MSELoss()(ŷ_val, y_for_val))

        println("RMSE for $y_label: ", rmse_)
        push!(rmse, rmse_)

        if plot
            if isempty(t_obs_val)
                println("No observational data available for $y_label in sample $sample_n")
                ax = CairoMakie.Axis(fig[i, 1], xlabel="Time (hours)", ylabel=y_labels[i], xgridvisible=false, ygridvisible=false)
                continue
            elseif isempty(t_for_val)
                println("No future data available for $y_label in sample $sample_n")
                ax = CairoMakie.Axis(fig[i, 1], xlabel="Time (hours)", ylabel=y_labels[i], xgridvisible=false, ygridvisible=false)
                continue
            else
                ax = CairoMakie.Axis(fig[i, 1], xlabel="Time (hours)", ylabel=y_labels[i], xgridvisible=false, ygridvisible=false)
                push!(axes, ax)
                scatter!(ax, t_obs_val, y_obs[i, masks_obs[i, :, sample_n] .== 1, sample_n], color=:blue, label="Past Observations", markersize=10)
                lines!(ax, t_obs_val, y_obs[i, masks_obs[i, :, sample_n] .== 1, sample_n], color=(:blue, 0.4), linewidth=2, linestyle=:dot)

                scatter!(ax, t_for_val, y_for[i, masks_for[i, :, sample_n] .== 1, sample_n], color=:green, label="Future Ground Truth", markersize=10)
                lines!(ax, t_for_val, y_for[i, masks_for[i, :, sample_n] .== 1, sample_n], color=(:green, 0.4), linestyle=:dot)

                scatter!(ax, t_for_val, ŷ[ masks_for[i, :, sample_n] .== 1, sample_n], color=:red, label="Model Predictions", markersize=10)
                lines!(ax, t_for_val, ŷ[ masks_for[i, :, sample_n] .== 1, sample_n], color=(:red, 0.4), linestyle=:dot)
                if i == 1
                    poly!(ax, [0, t_obs[end], t_obs[end], 0], [-10, -10, 500, 500], color=(:blue, 0.05), label="Observation Period (Past)")
                    poly!(ax, [t_obs[end], t_for_val[end], t_for_val[end], t_obs[end]], [-10, -10, 500, 500], color=(:red, 0.05), label="Forecasting Period (Future)")
                else
                    poly!(ax, [0, t_obs[end], t_obs[end], 0], [-10, -10, 500, 500], color=(:blue, 0.05))
                    poly!(ax, [t_obs[end], t_for_val[end], t_for_val[end], t_obs[end]], [-10, -10, 500, 500], color=(:red, 0.05))
                end

                all_y_values = vcat(
                    y_obs[i, masks_obs[i, :, sample_n] .== 1, sample_n],
                    y_for[i, masks_for[i, :, sample_n] .== 1, sample_n],
                    ŷ[ masks_for[i, :, sample_n] .== 1, sample_n],
                )

                y_min = minimum(all_y_values) - 0.1 * (maximum(all_y_values) - minimum(all_y_values))
                y_max = maximum(all_y_values) + 0.1 * (maximum(all_y_values) - minimum(all_y_values))
                ylims!(ax, y_min, y_max)

                if i == 1
                    fig[i, 2] = Legend(fig, ax, framevisible=false, halign=:left)
                end
            end
        end
    end

    if plot
        linkxaxes!(axes...)
        colgap!(fig.layout, 10)
        display(fig)
        return fig, rmse, crps
    else
        return rmse, crps
    end
end

# Function to create RNN model

function create_rnn_model(config, dims, rng)
    hidden_dim = config["obs_encoder"]["hidden_size"]
    latent_dim = config["latent_dim"]
    n_features = dims["obs_dim"]+ dims["input_dim"]
    n_timepoints_for = 25  # Adjust based on your forecasting horizon
    
    model = Chain(
        encoder=Chain(
            Recurrence(LSTMCell(n_features => hidden_dim); return_sequence=true),
            Recurrence(LSTMCell(hidden_dim => latent_dim); return_sequence=false)
        ),
        decoder=Chain(
            Dense(latent_dim, hidden_dim),
            BranchLayer(
                BranchLayer(Dense(hidden_dim, n_timepoints_for), Dense(hidden_dim, n_timepoints_for, softplus)),
                BranchLayer(Dense(hidden_dim, n_timepoints_for), Dense(hidden_dim, n_timepoints_for, softplus)),
                BranchLayer(Dense(hidden_dim, n_timepoints_for), Dense(hidden_dim, n_timepoints_for, softplus))
            )
        )
    )
    
    θ, st = Lux.setup(rng, model)
    return model, θ, st
end


# Perform k-fold training with Latent SDE model
n_folds = 5
config_path = "./configs/ICU_config_lsde.yml"
model_type = "lsde"

# Perform k-fold cross-validation for LSDE
@info "Starting $n_folds-fold cross-validation for $model_type model"
lsde_models, lsde_params, lsde_states, lsde_performances = kfold_train(
    data, 
    n_folds, 
    rng, 
    config_path, 
    model_type, 
    timepoints, 
    loss_fn, 
    eval_fn, 
    forecast,
    viz_fn_forecast
);

# Present LSDE model performance with sample plot
# For performance summary only (no plot), use: plot_sample=false or omit plotting arguments
lsde_stats = assess_model_performance(lsde_performances, variables_of_interest, model_name="Latent SDE",
                           plot_sample=true, sample_n=4, models=lsde_models, params=lsde_params, 
                           states=lsde_states, data=test_loader.data, timepoints=timepoints, 
                           config=YAML.load_file(config_path));

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
    forecast,
    viz_fn_forecast
);

# Present LODE model performance with sample plot
lode_stats = assess_model_performance(lode_performances, variables_of_interest, model_name="Latent ODE",
                           plot_sample=true, sample_n=4, models=lode_models, params=lode_params, 
                           states=lode_states, data=test_loader.data, timepoints=timepoints, 
                           config=YAML.load_file("./configs/ICU_config_lode.yml"));


# RNN model training and evaluation
rnn_config_path = "./configs/ICU_RNN_config.yml"
model_type = "rnn"

# Perform k-fold cross-validation for RNN
@info "Starting $n_folds-fold cross-validation for $model_type model"
rnn_models, rnn_params, rnn_states, rnn_performances = kfold_train(
    data, 
    n_folds, 
    rng, 
    rnn_config_path, 
    model_type, 
    timepoints, 
    loss_fn_rnn, 
    eval_fn_rnn, 
    forecast_rnn,
    viz_fn_forecast_rnn
);

# Present RNN model performance with sample plot
rnn_stats = assess_model_performance(rnn_performances, variables_of_interest, model_name="RNN", model_type="rnn", forecast_fn=forecast_rnn,
                           plot_sample=true, sample_n=4, viz_fn=viz_fn_forecast_rnn, models=rnn_models, params=rnn_params, 
                           states=rnn_states, data=test_loader.data, timepoints=timepoints, 
                           config=YAML.load_file(rnn_config_path));

# Compare RNN model with others
model_comparison_rnn = compare_models(
    Dict("Latent SDE" => lsde_stats, "Latent ODE" => lode_stats, "RNN" => rnn_stats),
    sort_by="rmse",  # Sort by RMSE (can also use "crps")
    ascending=true   # Best models first (lowest values)
);

