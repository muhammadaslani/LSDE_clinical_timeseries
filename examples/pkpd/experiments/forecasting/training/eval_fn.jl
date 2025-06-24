# Evaluation functions for PKPD forecasting models
function eval_fn_nde(model, θ, st, ts, data, config)
    u_obs, covars_obs, x_obs, y₁_obs, y₂_obs, mask₁_obs, mask₂_obs, 
        u_forecast, covars_forecast, x_forecast, y₁_forecast, y₂_forecast, mask₁_forecast, mask₂_forecast = data
    batch_size= size(u_forecast)[end]
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    Ex, Ey = predict(model, solver, vcat(covars_obs, reverse(y₁_obs, dims=2), reverse(y₂_obs, dims=2)), u_forecast, ts, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    ŷ₁_m, ŷ₂_m = dropmean(Ey[1], dims=4), dropmean(Ey[2], dims=4)
    eval_loss1 = CrossEntropy_Loss(ŷ₁_m, y₁_forecast, mask₁_forecast; agg=sum)/batch_size
    eval_loss2 = -poisson_loglikelihood(ŷ₂_m, y₂_forecast, mask₂_forecast)/batch_size
    eval_loss = eval_loss1 + eval_loss2
    return (eval_loss, eval_loss1, eval_loss2)
end

function eval_fn_rnn(model, θ, st, ts, data, config)
    u_obs, covars_obs, x_obs, y₁_obs, y₂_obs, mask₁_obs, mask₂_obs, 
        u_forecast, covars_forecast, x_forecast, y₁_forecast, y₂_forecast, mask₁_forecast, mask₂_forecast = data

    forecast_length = size(u_forecast, 2)
    batch_size = size(y₁_forecast)[end]
    # Combine inputs for RNN
    history = vcat(covars_obs, y₁_obs, y₂_obs, u_obs)
    # Forward pass
    ŷ, st, vae_params = model(history, u_forecast, forecast_length, θ, st)
    μ, logσ² = vae_params.μ, vae_params.logσ²
    
    # Calculate evaluation losses
    eval_loss1 = CrossEntropy_Loss(ŷ[1], y₁_forecast, mask₁_forecast; agg=sum) / batch_size
    eval_loss2 = -poisson_loglikelihood(ŷ[2], y₂_forecast, mask₂_forecast) / batch_size
    total_eval_loss = eval_loss1 + eval_loss2
    
    return (total_eval_loss, eval_loss1, eval_loss2)
end

# Performance assessment function for k-fold validation
function assess_model_performance(performances, variables_of_interest; model_name="Model", model_type="lsde", 
                                forecast_fn=forecast_nde, plot_sample=false, sample_n=3, viz_fn=viz_fn_nde, 
                                models=nothing, params=nothing, states=nothing, data=nothing, timepoints=nothing, 
                                config=nothing, best_fold_idx=nothing)
    """
    Presents PKPD model performance across k-folds by calculating and printing
    mean and standard deviation for the three evaluation metrics:
    - Health status: cross-entropy loss
    - Tumor volume: RMSE  
    - Cell count: negative log-likelihood
    Optionally plots a sample forecast from the best performing model.
    
    Arguments:
    - performances: Array of (crossentropy_health, rmse_tumor, nll_count) tuples from k-fold training
    - variables_of_interest: Array of output variable names ["Health", "Tumor", "CellCount"] 
    - model_name: Name of the model for display purposes
    - model_type: Type of model ("lsde", "lode", "rnn")
    - forecast_fn: Forecast function to use for generating predictions
    - plot_sample: Boolean flag to enable sample plotting
    - sample_n: Sample number to plot (default=3)
    - viz_fn: Visualization function
    - models: Array of trained models (required if plot_sample=True)
    - params: Array of trained parameters (required if plot_sample=True)
    - states: Array of model states (required if plot_sample=True)
    - data: Test data for plotting (required if plot_sample=True)  
    - timepoints: Time points for plotting (required if plot_sample=True)
    - config: Model configuration (required if plot_sample=True)
    - best_fold_idx: Index of best performing fold (if None, will be determined automatically)
    
    Returns:
    - Dictionary with performance statistics or plotting results
    """
    
    n_folds = length(performances)
    
    # Print header
    println("\n" * "="^70)
    println("$model_name Performance Summary ($n_folds-fold Cross-Validation)")
    println("="^70)
    
    # Extract and process metrics based on model type
    if model_type == "rnn"
        # RNN models may have different performance format
        # Assuming RNN returns (total_loss, health_loss, tumor_loss) similar to other models
        if length(performances) > 0 && length(performances[1]) == 3
            health_losses = [perf[1] for perf in performances]  # or appropriate metric
            tumor_losses = [perf[2] for perf in performances]
            count_losses = [perf[3] for perf in performances]
            
            # Calculate statistics
            health_mean, health_std = mean(health_losses), std(health_losses)
            tumor_mean, tumor_std = mean(tumor_losses), std(tumor_losses)
            count_mean, count_std = mean(count_losses), std(count_losses)
            
            println("\nPerformance Metrics:")
            println("-"^40)
            @printf("Health Status Loss: %.4f ± %.4f\n", health_mean, health_std)
            @printf("Tumor Volume Loss:  %.4f ± %.4f\n", tumor_mean, tumor_std)
            @printf("Cell Count Loss:    %.4f ± %.4f\n", count_mean, count_std)
            
            # Overall performance
            overall_mean = mean([health_mean, tumor_mean, count_mean])
            overall_std = sqrt(mean([health_std^2, tumor_std^2, count_std^2]))
            
        else
            @warn "Unexpected RNN performance format. Using fallback processing."
            return nothing
        end
    else
        # Neural DE models: (crossentropy_health, rmse_tumor, nll_count)
        crossentropy_health_values = [perf[1] for perf in performances]
        rmse_tumor_values = [perf[2] for perf in performances]
        nll_count_values = [perf[3] for perf in performances]
        
        # Calculate statistics
        crossentropy_mean, crossentropy_std = mean(crossentropy_health_values), std(crossentropy_health_values)
        rmse_mean, rmse_std = mean(rmse_tumor_values), std(rmse_tumor_values)
        nll_mean, nll_std = mean(nll_count_values), std(nll_count_values)
        
        println("\nPerformance Metrics:")
        println("-"^40)
        @printf("Health Status (Cross-entropy): %.4f ± %.4f\n", crossentropy_mean, crossentropy_std)
        @printf("Tumor Volume (RMSE):           %.4f ± %.4f\n", rmse_mean, rmse_std)
        @printf("Cell Count (Neg. Log-lik):     %.4f ± %.4f\n", nll_mean, nll_std)
        
        # Overall performance (weighted average - lower is better for all metrics)
        overall_mean = mean([crossentropy_mean, rmse_mean, nll_mean])
        overall_std = sqrt(mean([crossentropy_std^2, rmse_std^2, nll_std^2]))
    end
    
    println("\nOverall Performance:")
    println("-"^40)
    @printf("Average across metrics: %.4f ± %.4f\n", overall_mean, overall_std)
    
    # Find best performing fold (lowest overall performance)
    if isnothing(best_fold_idx)
        if model_type == "rnn"
            best_fold_idx = argmin([mean([perf[1], perf[2], perf[3]]) for perf in performances])
        else
            best_fold_idx = argmin([mean([perf[1], perf[2], perf[3]]) for perf in performances])
        end
    end
    
    println("Best performing fold: $best_fold_idx")
    println("="^70)
    
    # Optional sample plotting
    fig = nothing
    sample_metrics = nothing
    
    if plot_sample
        if any(isnothing.([models, params, states, data, timepoints, config]))
            @warn "Cannot plot sample: missing required arguments (models, params, states, data, timepoints, config)"
        else
            println("\nGenerating sample forecast from best performing model (Fold $best_fold_idx)...")
            
            # Get best model components
            best_model = models[best_fold_idx];
            best_params = params[best_fold_idx];
            best_state = states[best_fold_idx];
            
            # Extract data components
            u_obs, covars_obs, x_obs, y₁_obs, y₂_obs, mask₁_obs, mask₂_obs,
            u_forecast, covars_forecast, x_forecast, y₁_forecast, y₂_forecast, mask₁_forecast, mask₂_forecast = data;
            
            # Prepare data for plotting
            data_obs = (u_obs, covars_obs, x_obs, y₁_obs, y₂_obs, mask₁_obs, mask₂_obs);
            future_true_data = (u_forecast, covars_forecast, x_forecast, y₁_forecast, y₂_forecast, mask₁_forecast, mask₂_forecast);
            timepoints_obs, timepoints_forecast = timepoints;

            # Generate forecast
            try
                forecasted_data = forecast_fn(best_model, best_params, best_state, data_obs, 
                                            u_forecast, timepoints_forecast, config)
                
                # Create visualization
                fig, sample_crossentropy, sample_rmse, sample_nll = viz_fn(timepoints_obs, timepoints_forecast, 
                                                                          data_obs, future_true_data, forecasted_data, 
                                                                          sample_n=sample_n, plot=true);


                display(fig)
                
                sample_metrics = (sample_crossentropy, sample_rmse, sample_nll)
                
                println("\nSample #$sample_n Forecast Metrics:")
                println("-"^40)
                @printf("Health Cross-entropy: %.4f\n", sample_crossentropy)
                @printf("Tumor RMSE:          %.4f\n", sample_rmse)
                @printf("Cell Count NLL:      %.4f\n", sample_nll)
                
            catch e
                @warn "Error generating forecast visualization: $e"
            end
        end
    end
    
    # Prepare return statistics
    if model_type == "rnn"
        return_stats = (
            health_mean=health_mean, health_std=health_std,
            tumor_mean=tumor_mean, tumor_std=tumor_std,
            count_mean=count_mean, count_std=count_std,
            overall_mean=overall_mean, overall_std=overall_std,
            best_fold_idx=best_fold_idx,
            figure=fig, sample_metrics=sample_metrics
        )
    else
        return_stats = (
            crossentropy_mean=crossentropy_mean, crossentropy_std=crossentropy_std,
            rmse_mean=rmse_mean, rmse_std=rmse_std,
            nll_mean=nll_mean, nll_std=nll_std,
            overall_mean=overall_mean, overall_std=overall_std,
            best_fold_idx=best_fold_idx,
            figure=fig, sample_metrics=sample_metrics
        )
    end
    
    return return_stats
end

# Function for comparing multiple PKPD model performances
function compare_pkpd_models(model_stats_dict; sort_by="overall", ascending=true)
    """
    Compare performance of multiple PKPD models and display results in a formatted table.
    
    Arguments:
    - model_stats_dict: Dictionary with model names as keys and their assess_model_performance results as values
    - sort_by: Metric to sort by ("overall", "health", "tumor", "count")
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
    if sort_by == "overall"
        sort_values = [stats.overall_mean for stats in model_stats]
    elseif sort_by == "health"
        sort_values = [haskey(stats, :crossentropy_mean) ? stats.crossentropy_mean : stats.health_mean for stats in model_stats]
    elseif sort_by == "tumor"
        sort_values = [haskey(stats, :rmse_mean) ? stats.rmse_mean : stats.tumor_mean for stats in model_stats]
    elseif sort_by == "count"
        sort_values = [haskey(stats, :nll_mean) ? stats.nll_mean : stats.count_mean for stats in model_stats]
    else
        @warn "Invalid sort_by parameter. Using 'overall' as default."
        sort_values = [stats.overall_mean for stats in model_stats]
        sort_by = "overall"
    end
    
    # Get sorted indices
    sorted_indices = sortperm(sort_values, rev=!ascending)
    sorted_names = model_names[sorted_indices]
    sorted_stats = model_stats[sorted_indices]
    
    # Print comparison table
    println("\n" * "="^90)
    println("PKPD Model Comparison Summary (sorted by $(uppercase(sort_by)))")
    println("="^90)
    @printf("%-12s | %-12s | %-12s | %-12s | %-12s | %-6s\n", 
            "Model", "Health", "Tumor", "Count", "Overall", "Rank")
    println("-"^90)
    
    # Find best values for each metric
    best_overall = minimum([stats.overall_mean for stats in model_stats])
    best_health = minimum([haskey(stats, :crossentropy_mean) ? stats.crossentropy_mean : stats.health_mean for stats in model_stats])
    best_tumor = minimum([haskey(stats, :rmse_mean) ? stats.rmse_mean : stats.tumor_mean for stats in model_stats])
    best_count = minimum([haskey(stats, :nll_mean) ? stats.nll_mean : stats.count_mean for stats in model_stats])
    
    for (rank, (name, stats)) in enumerate(zip(sorted_names, sorted_stats))
        # Get metric values and add indicators for best performance
        health_val = haskey(stats, :crossentropy_mean) ? stats.crossentropy_mean : stats.health_mean
        health_std = haskey(stats, :crossentropy_std) ? stats.crossentropy_std : stats.health_std
        tumor_val = haskey(stats, :rmse_mean) ? stats.rmse_mean : stats.tumor_mean
        tumor_std = haskey(stats, :rmse_std) ? stats.rmse_std : stats.tumor_std
        count_val = haskey(stats, :nll_mean) ? stats.nll_mean : stats.count_mean
        count_std = haskey(stats, :nll_std) ? stats.nll_std : stats.count_std
        
        # Add indicators for best performance
        health_indicator = health_val ≈ best_health ? " ★" : ""
        tumor_indicator = tumor_val ≈ best_tumor ? " ★" : ""
        count_indicator = count_val ≈ best_count ? " ★" : ""
        overall_indicator = stats.overall_mean ≈ best_overall ? " ★" : ""
        
        @printf("%-12s | %.3f±%.3f%s | %.3f±%.3f%s | %.3f±%.3f%s | %.3f±%.3f%s | %-6d\n",
                name[1:min(12, length(name))],  # Truncate long names
                health_val, health_std, health_indicator,
                tumor_val, tumor_std, tumor_indicator,
                count_val, count_std, count_indicator,
                stats.overall_mean, stats.overall_std, overall_indicator,
                rank)
    end
    
    println("-"^90)
    println("★ = Best performance for that metric")
    println("Health: Cross-entropy (lower is better)")
    println("Tumor: RMSE (lower is better)")  
    println("Count: Negative Log-likelihood (lower is better)")
    println("="^90)
    
    # Print best model info
    best_model_name = sorted_names[1]
    @info "Best overall model (lowest average metric): $best_model_name"
    
    # Calculate performance differences
    if length(model_stats) > 1
        println("\nPerformance differences (compared to best):")
        for (name, stats) in zip(sorted_names[2:end], sorted_stats[2:end])
            overall_diff = stats.overall_mean - sorted_stats[1].overall_mean
            @printf("  %s vs %s: +%.4f overall performance\n", 
                   name, sorted_names[1], overall_diff)
        end
    end
    
    # Return sorted results
    return Dict(zip(sorted_names, sorted_stats))
end
