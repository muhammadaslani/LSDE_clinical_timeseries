# Evaluation functions for PKPD forecasting models
function eval_fn_nde(model, θ, st, ts, data, config)
    u_obs, covars_obs, _, y₁_obs, y₂_obs, _, _, u_forecast, _, x_forecast, y₁_forecast, y₂_forecast, mask₁_forecast, mask₂_forecast = data
    batch_size = size(x_forecast)[end]
    (ŷ₁, ŷ₂), _, _ = model(vcat(covars_obs, y₁_obs, y₂_obs), u_forecast, ts, θ, st)
    eval_loss_1 = CrossEntropy_Loss(ŷ₁, y₁_forecast, mask₁_forecast; agg=sum) / batch_size
    eval_loss_2 = -100*poisson_loglikelihood(ŷ₂, y₂_forecast, mask₂_forecast) / batch_size
    eval_loss = eval_loss_1 + eval_loss_2
    return (eval_loss, eval_loss_1, eval_loss_2)
end

function eval_fn_lstm(model, θ, st, ts, data, config)
    _, covars_obs, _, y₁_obs, y₂_obs, _, _, u_forecast, _, _, y₁_forecast, y₂_forecast, mask₁_forecast, mask₂_forecast = data
    batch_size = size(y₁_forecast)[end]
    (ŷ₁, ŷ₂), _, _ = model(vcat(covars_obs, y₁_obs, y₂_obs), u_forecast, ts, θ, st)
    eval_loss1 = CrossEntropy_Loss(ŷ₁, y₁_forecast, mask₁_forecast; agg=sum) / batch_size
    eval_loss2 = -100*poisson_loglikelihood(ŷ₂, y₂_forecast, mask₂_forecast) / batch_size
    total_eval_loss = eval_loss1 + eval_loss2

    return (total_eval_loss, eval_loss1, eval_loss2)
end


function eval_forecast(true_data, forecasted_data)
    _, _, _, y₁_f, y₂_f, mask₁_f, mask₂_f = true_data
    x̂_mc, ŷ_mc = forecasted_data
    
    y₁_f_class = onecold(y₁_f, Array(0:5))
    y₁_f_class = reshape(y₁_f_class, 1, size(y₁_f_class)...)
    ŷ₁_mc, ŷ₂_mc = ŷ_mc[1], ŷ_mc[2]
    ŷ₁_mc_m, ŷ₂_mc_m = dropmean(ŷ₁_mc, dims=4), dropmean(ŷ₂_mc, dims=4)

    ŷ₁_mc_m_class = onecold(softmax(ŷ₁_mc_m, dims=1), Array(0:5))
    ŷ₁_mc_m_class = reshape(ŷ₁_mc_m_class, 1, size(ŷ₁_mc_m_class)...)
    y₁_acc = acc(y₁_f_class, ŷ₁_mc_m_class, mask₁_f)
    y₁_npe = -npe(ŷ₁_mc, mask₁_f)

    ŷ₂_rmse = sqrt.(mse(ŷ₂_mc_m, y₂_f, mask₂_f))
    ŷ₂_nll = -poisson_loglikelihood_multiple_samples(ŷ₂_mc, y₂_f, mask₂_f; agg=sum)
    return (y₁_acc, y₁_npe), (ŷ₂_rmse, ŷ₂_nll)
end



# Performance assessment function for k-fold validation
function assess_model_performance(performances, variables_of_interest; model_name="Model",
    forecast_fn=forecast_nde, plot_sample=false, sample_n=3, viz_fn=viz_fn_nde,
    models=nothing, params=nothing, states=nothing, data=nothing, normalization_stats, timepoints=nothing,
    config=nothing, best_fold_idx=nothing)
    """
    Presents PKPD model performance across k-folds by calculating and printing
    mean and standard deviation for the evaluation metrics:
    - Health status: accuracy and negative predictive entropy
    - Cell count: RMSE and negative log-likelihood
    Optionally plots a sample forecast from the best performing model.

    Arguments:
    - performances: Array of ((y₁_acc, y₁_npe), (ŷ₂_rmse, ŷ₂_nll)) tuples from k-fold training
    - variables_of_interest: Array of output variable names ["Health", "CellCount"] 
    - model_name: Name of the model for display purposes
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
    println("\n" * "="^80)
    println("$model_name Performance Summary ($n_folds-fold Cross-Validation)")
    println("="^80)

    # Extract metrics from new format: ((y₁_acc, y₁_npe), (ŷ₂_rmse, ŷ₂_nll))
    y1_acc_values = [perf[1][1] for perf in performances]
    y1_npe_values = [perf[1][2] for perf in performances]
    y2_rmse_values = [perf[2][1] for perf in performances]
    y2_nll_values = [perf[2][2] for perf in performances]

    # Calculate statistics
    acc_mean, acc_std = mean(y1_acc_values), std(y1_acc_values)
    npe_mean, npe_std = mean(y1_npe_values), std(y1_npe_values)
    rmse_mean, rmse_std = mean(y2_rmse_values), std(y2_rmse_values)
    nll_mean, nll_std = mean(y2_nll_values), std(y2_nll_values)

    println("\nPerformance Metrics:")
    println("-"^50)
    @printf("Health Status Accuracy:        %.4f ± %.4f\n", acc_mean, acc_std)
    @printf("Health Status NPE:             %.4f ± %.4f\n", npe_mean, npe_std)
    @printf("Cell Count RMSE:               %.4f ± %.4f\n", rmse_mean, rmse_std)
    @printf("Cell Count Neg. Log-lik:       %.4f ± %.4f\n", nll_mean, nll_std)

    # Overall performance (simple average - no normalization or negation)
    overall_mean = mean([acc_mean, npe_mean, rmse_mean, nll_mean])
    overall_std = sqrt(mean([acc_std^2, npe_std^2, rmse_std^2, nll_std^2]))

    println("\nOverall Performance:")
    println("-"^50)
    @printf("Average across metrics: %.4f ± %.4f\n", overall_mean, overall_std)

    # Find best performing fold (using simple average)
    if isnothing(best_fold_idx)
        fold_scores = []
        for perf in performances
            acc, npe = perf[1]
            rmse, nll = perf[2]
            score = mean([acc, npe, rmse, nll])  # Simple average
            push!(fold_scores, score)
        end
        best_fold_idx = argmax(fold_scores)  # Or argmin depending on your preference
    end

    println("Best performing fold: $best_fold_idx")
    println("="^80)

    # Optional sample plotting
    fig = nothing
    sample_metrics = nothing

    if plot_sample
        if any(isnothing.([models, params, states, data, timepoints, config]))
            @warn "Cannot plot sample: missing required arguments (models, params, states, data, timepoints, config)"
        else
            println("\nGenerating sample forecast from best performing model (Fold $best_fold_idx)...")

            # Get best model components
            best_model = models[best_fold_idx]
            best_params = params[best_fold_idx]
            best_state = states[best_fold_idx]

            # Extract data components
            u_obs, covars_obs, x_obs, y₁_obs, y₂_obs, mask₁_obs, mask₂_obs,
            u_forecast, covars_forecast, x_forecast, y₁_forecast, y₂_forecast, mask₁_forecast, mask₂_forecast = data

            # Prepare data for plotting
            data_obs = (u_obs, covars_obs, x_obs, y₁_obs, y₂_obs, mask₁_obs, mask₂_obs)
            future_true_data = (u_forecast, covars_forecast, x_forecast, y₁_forecast, y₂_forecast, mask₁_forecast, mask₂_forecast)
            timepoints_obs, timepoints_forecast = timepoints

            # Generate forecast
            forecasted_data = forecast_fn(best_model, best_params, best_state, data_obs,
                u_forecast, timepoints_forecast, config)

            # Create visualization and get sample metrics
            fig = viz_fn(timepoints_obs, timepoints_forecast,
                data_obs, future_true_data, forecasted_data, normalization_stats,
                sample_n=sample_n)

            # Evaluate sample performance
            sample_metrics = eval_forecast(future_true_data, forecasted_data)
            sample_acc, sample_npe = sample_metrics[1]
            sample_rmse, sample_nll = sample_metrics[2]

            display(fig)

            println("\nSample #$sample_n Forecast Metrics:")
            println("-"^50)
            @printf("Health Status Accuracy: %.4f\n", sample_acc)
            @printf("Health Status NPE:      %.4f\n", sample_npe)
            @printf("Cell Count RMSE:        %.4f\n", sample_rmse)
            @printf("Cell Count NLL:         %.4f\n", sample_nll)
        end
    end

    # Prepare return statistics
    return_stats = (
        acc_mean=acc_mean, acc_std=acc_std,
        npe_mean=npe_mean, npe_std=npe_std,
        rmse_mean=rmse_mean, rmse_std=rmse_std,
        nll_mean=nll_mean, nll_std=nll_std,
        overall_mean=overall_mean, overall_std=overall_std,
        best_fold_idx=best_fold_idx,
        figure=fig, sample_metrics=sample_metrics
    )

    return return_stats
end

# Function for comparing multiple PKPD model performances
function compare_pkpd_models(model_stats_dict; sort_by="overall", ascending=false)
    """
    Compare performance of multiple PKPD models and display results in a formatted table.

    Arguments:
    - model_stats_dict: Dictionary with model names as keys and their assess_model_performance results as values
    - sort_by: Metric to sort by ("overall", "accuracy", "npe", "rmse", "nll")
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

    # Sort models by specified metric (no normalization or negation)
    if sort_by == "overall"
        sort_values = [stats.overall_mean for stats in model_stats]
    elseif sort_by == "accuracy"
        sort_values = [stats.acc_mean for stats in model_stats]
    elseif sort_by == "npe"
        sort_values = [stats.npe_mean for stats in model_stats]
    elseif sort_by == "rmse"
        sort_values = [stats.rmse_mean for stats in model_stats]
    elseif sort_by == "nll"
        sort_values = [stats.nll_mean for stats in model_stats]
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
    println("\n" * "="^100)
    println("PKPD Model Comparison Summary (sorted by $(uppercase(sort_by)))")
    println("="^100)
    @printf("%-15s | %-12s | %-12s | %-12s | %-12s | %-12s | %-6s\n",
        "Model", "Accuracy", "NPE", "RMSE", "NLL", "Overall", "Rank")
    println("-"^100)

    # Find best values for each metric (based on typical interpretation)
    best_acc = maximum([stats.acc_mean for stats in model_stats])      # Higher is better
    best_npe = maximum([stats.npe_mean for stats in model_stats])      # Higher is better
    best_rmse = minimum([stats.rmse_mean for stats in model_stats])    # Lower is better
    best_nll = minimum([stats.nll_mean for stats in model_stats])      # Lower is better
    best_overall = maximum([stats.overall_mean for stats in model_stats])  # Depends on metric mix

    for (rank, (name, stats)) in enumerate(zip(sorted_names, sorted_stats))
        # Add indicators for best performance
        acc_indicator = stats.acc_mean ≈ best_acc ? " ★" : ""
        npe_indicator = stats.npe_mean ≈ best_npe ? " ★" : ""
        rmse_indicator = stats.rmse_mean ≈ best_rmse ? " ★" : ""
        nll_indicator = stats.nll_mean ≈ best_nll ? " ★" : ""
        overall_indicator = stats.overall_mean ≈ best_overall ? " ★" : ""

        @printf("%-15s | %.3f±%.3f%s | %.3f±%.3f%s | %.3f±%.3f%s | %.3f±%.3f%s | %.3f±%.3f%s | %-6d\n",
            name[1:min(15, length(name))],  # Truncate long names
            stats.acc_mean, stats.acc_std, acc_indicator,
            stats.npe_mean, stats.npe_std, npe_indicator,
            stats.rmse_mean, stats.rmse_std, rmse_indicator,
            stats.nll_mean, stats.nll_std, nll_indicator,
            stats.overall_mean, stats.overall_std, overall_indicator,
            rank)
    end

    println("-"^100)
    println("★ = Best performance for that metric")
    println("Accuracy: Classification accuracy (higher is better)")
    println("NPE: Negative Predictive Entropy (higher is better)")
    println("RMSE: Root Mean Square Error (lower is better)")
    println("NLL: Negative Log-likelihood (lower is better)")
    println("Overall: Simple average of all metrics")
    println("="^100)

    # Print best model info
    best_model_name = sorted_names[1]
    @info "Best model by $(sort_by): $best_model_name"

    # Return sorted results
    return Dict(zip(sorted_names, sorted_stats))
end