
function eval_fn_nde(model, θ, st, ts, data, config)
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
    
    # Only display CRPS for models that support it
    if model_type == "lsde" || model_type == "lode"
        println("\nCRPS (Continuous Ranked Probability Score):")
        println("-"^40)
        for (i, feature) in enumerate(variables_of_interest)
            @printf("%-10s: %.4f ± %.4f\n", feature, crps_means[i], crps_stds[i])
        end
    end
    
    # Overall performance (mean across features)
    overall_rmse_mean = mean(rmse_means)
    overall_rmse_std = sqrt(mean(rmse_stds.^2))  # Combined std
    overall_crps_mean = mean(crps_means)
    overall_crps_std = sqrt(mean(crps_stds.^2))  # Combined std
    
    println("\nOverall Performance (Average across features):")
    println("-"^40)
    @printf("RMSE: %.4f ± %.4f\n", overall_rmse_mean, overall_rmse_std)
    if model_type == "lsde" || model_type == "lode"
        @printf("CRPS: %.4f ± %.4f\n", overall_crps_mean, overall_crps_std)
    else
        println("CRPS: Not available for this model type")
    end
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
            if model_type == "rnn"
                # For RNN models, viz function only returns rmse (no crps)
                fig, rmse = viz_fn(timepoints_obs, timepoints_for, data_obs, future_true_data, forecasted_data, sample_n=sample_n, plot=true)
                crps = [0.0] * length(rmse)  # Set CRPS to zero for RNN
            else
                # For LSDE/LODE models, viz function returns both rmse and crps
                fig, rmse, crps = viz_fn(timepoints_obs, timepoints_for, data_obs, future_true_data, forecasted_data, sample_n=sample_n, plot=true)
            end
            
            println("\nSample number $sample_n forecast plotted for best model (fold $best_fold_idx)")
            #println("Sample number $sample_n RMSE: [$(round.(rmse, digits=4))]")
            # if model_type == "lsde" || model_type == "lode"
            #     println("Sample number $sample_n CRPS: [$(round.(crps, digits=4))]")
            # end
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
        # Only consider models that have valid CRPS values (non-zero)
        sort_values = [stats.overall_crps_mean > 0 ? stats.overall_crps_mean : Inf for stats in model_stats]
        if all(isinf.(sort_values))
            @warn "No models have CRPS values. Sorting by RMSE instead."
            sort_values = [stats.overall_rmse_mean for stats in model_stats]
            sort_by = "rmse"
        end
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
    println("\n" * "="^70)
    println("Model Comparison Summary (sorted by $(uppercase(sort_by)))")
    println("="^70)
    @printf("%-15s | %-15s | %-15s | %-10s\n", "Model", "Avg RMSE", "Avg CRPS", "Rank")
    println("-"^70)
    
    best_rmse = minimum([stats.overall_rmse_mean for stats in model_stats])
    # Only calculate best CRPS among models that have valid CRPS values
    crps_values = [stats.overall_crps_mean for stats in model_stats if stats.overall_crps_mean > 0]
    best_crps = isempty(crps_values) ? 0.0 : minimum(crps_values)
    
    for (rank, (name, stats)) in enumerate(zip(sorted_names, sorted_stats))
        # Add indicators for best performance
        rmse_indicator = stats.overall_rmse_mean ≈ best_rmse ? " ★" : ""
        crps_indicator = (stats.overall_crps_mean > 0 && stats.overall_crps_mean ≈ best_crps) ? " ★" : ""
        
        # Format CRPS display
        crps_display = stats.overall_crps_mean > 0 ? 
                      @sprintf("%.4f±%.4f%s", stats.overall_crps_mean, stats.overall_crps_std, crps_indicator) :
                      "N/A          "
        
        @printf("%-15s | %.4f±%.4f%s | %s | %-10d\n", 
                name, 
                stats.overall_rmse_mean, stats.overall_rmse_std, rmse_indicator,
                crps_display,
                rank)
    end
    
    println("-"^70)
    println("★ = Best performance for that metric")
    println("="^70)
    
    # Print detailed comparison
    println("\nDetailed Performance Comparison:")
    println("-"^40)
    
    # Find best model overall (using only RMSE for mixed model types)
    best_overall_idx = argmin([stats.overall_rmse_mean for stats in sorted_stats])
    best_model_name = sorted_names[best_overall_idx]
    
    @info "Best overall model (lowest RMSE): $best_model_name"
    
    # Calculate performance differences
    if length(model_stats) > 1
        println("\nPerformance differences (compared to best):")
        for (name, stats) in zip(sorted_names[2:end], sorted_stats[2:end])
            rmse_diff = stats.overall_rmse_mean - sorted_stats[1].overall_rmse_mean
            if stats.overall_crps_mean > 0 && sorted_stats[1].overall_crps_mean > 0
                crps_diff = stats.overall_crps_mean - sorted_stats[1].overall_crps_mean
                @printf("  %s vs %s: +%.4f RMSE, +%.4f CRPS\n", 
                       name, sorted_names[1], rmse_diff, crps_diff)
            else
                @printf("  %s vs %s: +%.4f RMSE\n", 
                       name, sorted_names[1], rmse_diff)
            end
        end
    end
    
    # Return sorted results
    return Dict(zip(sorted_names, sorted_stats))
end
