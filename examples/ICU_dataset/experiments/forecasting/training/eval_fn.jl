function eval_fn_nde(model, θ, st, ts, data, config)
    _, x_obs, _, _, u_for, _, y_for, masks_for = data
    batch_size= size(y_for)[end]
    ŷ, _, _ = model(x_obs,  u_for, ts, θ, st)
    loss=0.0f0
    for i in eachindex(ŷ)
        μ, log_σ² =ŷ[i][1], ŷ[i][2]
        valid_indx= findall(masks_for[i, :, :] .== 1)
        loss += normal_loglikelihood(μ[1,valid_indx], log_σ²[1,valid_indx],y_for[i, valid_indx])/batch_size
    end
    return (loss, 0.0f0, 0.0f0) 
end


function eval_fn_lstm(model, θ, st, ts, data, config)
    _, x_obs, _, _, u_for, _, y_for, masks_for = data
    batch_size = size(y_for)[end]
    ŷ, _, _ = model(x_obs, u_for, ts, θ, st)
    eval_loss = 0.0f0
    for i in eachindex(ŷ)
        μ, log_σ² = ŷ[i][1], ŷ[i][2]
        valid_indx = findall(masks_for[i, :, :] .== 1)
        eval_loss += normal_loglikelihood(μ[1,valid_indx], log_σ²[1,valid_indx], y_for[i, valid_indx]) / batch_size
    end
    return (eval_loss, 0.0f0, 0.0f0)
end




function eval_forecast(true_data, forecasted_data)
    _, _, y_for, masks_for = true_data
    μ, σ² = forecasted_data
    crps = []
    rmse = []
    # Calculate RMSE and CRPS for each feature
    n_features = size(μ)[1]
    for i in 1:n_features

        dists = Normal.(μ[i], sqrt.(σ²[i]))
        ŷ = rand.(dists)

        ŷ_mean = dropdims(mean(ŷ, dims=4), dims=4)

        crps_ = empirical_crps(y_for[i:i, :, :], ŷ, masks_for[i:i, :, :])
        rmse_ = sqrt(mse(ŷ_mean[1,:,:], y_for[i,:,:], masks_for[i,:,:]))

        push!(crps, crps_)
        push!(rmse, rmse_)

    end 

    return rmse, crps

end



function assess_model_performance(performances, variables_of_interest; model_name="Model", forecast_fn=forecast,
                       plot_sample=false, sample_n=3, viz_fn=viz_fn_forecast, models=nothing, params=nothing, states=nothing, 
                       data=nothing, normalization_stats=nothing, timepoints=nothing, config=nothing, best_fold_idx=nothing)
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
    
    # All models now return both RMSE and CRPS
    for (fold_idx, (rmse, crps)) in enumerate(performances)
        rmse_values[fold_idx, :] = rmse[1:n_features]
        crps_values[fold_idx, :] = crps[1:n_features]
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
            
            # Create visualization - all models now return both rmse and crps
            fig, rmse, crps = viz_fn(timepoints_obs, timepoints_for, data_obs, future_true_data, forecasted_data, sample_n=sample_n, plot=true, normalization_stats =normalization_stats)
            
            println("\nSample number $sample_n forecast plotted for best model (fold $best_fold_idx)")
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
        sort_by = "rmse"
    end
    
    # Get sorted indices
    sorted_indices = sortperm(sort_values, rev=!ascending)
    sorted_names = model_names[sorted_indices]
    sorted_stats = model_stats[sorted_indices]
    
    # Print comparison table
    println("\n" * "="^80)
    println("Model Comparison Summary (sorted by $(uppercase(sort_by)))")
    println("="^80)
    @printf("%-15s | %-20s | %-20s | %-10s\n", "Model", "Avg RMSE", "Avg CRPS", "Rank")
    println("-"^80)
    
    # Find best performance for each metric
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
    
    println("-"^80)
    println("★ = Best performance for that metric")
    println("="^80)
    
    # Print detailed comparison
    println("\nDetailed Performance Comparison:")
    println("-"^40)
    
    # Find best model overall (based on sorting criterion)
    best_overall_idx = 1  # First in sorted list is best
    best_model_name = sorted_names[best_overall_idx]
    
    @info "Best overall model (lowest $sort_by): $best_model_name"
    
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