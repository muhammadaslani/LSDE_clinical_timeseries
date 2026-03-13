# Evaluation functions for PKPD forecasting models
function eval_fn(model, θ, st, ts, data, config)
    u_obs, covars_obs, x_obs, y₁_obs, y₂_obs, mask₁_obs, mask₂_obs, u_for, covars_for, x_for, y₁_for, y₂_for, mask₁_for, mask₂_for = data
    batch_size = size(y₁_obs)[end]

    ts_obs, ts_for = ts
    (ŷ₁, ŷ₂), _, kl_pq = model(vcat(covars_obs, y₁_obs, log.(y₂_obs .+ 1)), u_for, (ts_obs, ts_for), θ, st)
    eval_loss_1 = CrossEntropy_Loss(ŷ₁, y₁_for, mask₁_for; agg=sum) / batch_size
    eval_loss_2 = -poisson_loglikelihood(ŷ₂, y₂_for, mask₂_for) / batch_size
    kl_val = kl_pq === nothing ? 0.0f0 : mean(kl_pq[end, :])
    eval_loss = eval_loss_1 + eval_loss_2
    return (eval_loss, eval_loss_1, eval_loss_2, kl_val)
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
    ŷ₂_crps = empirical_crps(y₂_f, ŷ₂_mc, mask₂_f)
    ŷ₂_count = rand.(Poisson.(clamp.(ŷ₂_mc, 0.0, 1000.0)))
    ŷ₂_count_m = dropmean(ŷ₂_count, dims=4)
    ŷ₂_count_rmse = sqrt.(mse(ŷ₂_count_m, y₂_f, mask₂_f))
    ŷ₂_count_nll = -poisson_loglikelihood_multiple_samples(ŷ₂_mc, y₂_f, mask₂_f; agg=mean)
    return (y₁_acc, y₁_npe), (ŷ₂_rmse, ŷ₂_crps), (ŷ₂_count_rmse, ŷ₂_count_nll)
end



# Performance assessment function for k-fold validation
function assess_model_performance(performances, variables_of_interest; model_name="Model",
    forecast_fn=forecast, plot_sample=false, sample_n=3, viz_fn=viz_fn_nde,
    models=nothing, params=nothing, states=nothing, data=nothing, normalization_stats, timepoints=nothing,
    config_path=nothing, best_fold_idx=nothing)
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

    config = isnothing(config_path) ? nothing : begin
        cfg = load_config(config_path)
        merge(cfg["model"]["validation"], cfg["training"]["validation"])
    end

    n_folds = length(performances)

    # Print header
    println("\n" * "="^80)
    println("$model_name Performance Summary ($n_folds-fold Cross-Validation)")
    println("="^80)

    # Extract metrics from new format: ((y₁_acc, y₁_npe), (ŷ₂_rmse, ŷ₂_nll))
    y1_acc_values = [perf[1][1] for perf in performances]
    y1_npe_values = [perf[1][2] for perf in performances]
    y2_rmse_values = [perf[2][1] for perf in performances]
    y2_crps_values = [perf[2][2] for perf in performances]
    y2_count_rmse_values = [perf[3][1] for perf in performances]
    y2_count_nll_values = [perf[3][2] for perf in performances]

    # Calculate statistics
    acc_mean, acc_std = mean(y1_acc_values), std(y1_acc_values)
    npe_mean, npe_std = mean(y1_npe_values), std(y1_npe_values)
    y2_rmse_mean, y2_rmse_std = mean(y2_rmse_values), std(y2_rmse_values)
    y2_crps_mean, y2_crps_std = mean(y2_crps_values), std(y2_crps_values)
    count_rmse_mean, count_rmse_std = mean(y2_count_rmse_values), std(y2_count_rmse_values)
    count_nll_mean, count_nll_std = mean(y2_count_nll_values), std(y2_count_nll_values)

    println("\nPerformance Metrics:")
    println("-"^50)
    @printf("Health Status Accuracy:        %.4f ± %.4f\n", acc_mean, acc_std)
    @printf("Health Status NPE:             %.4f ± %.4f\n", npe_mean, npe_std)
    @printf("Tumor size RMSE:               %.4f ± %.4f\n", y2_rmse_mean, y2_rmse_std)
    @printf("Tumor size CRPS:               %.4f ± %.4f\n", y2_crps_mean, y2_crps_std)
    @printf("Cell Count RMSE:               %.4f ± %.4f\n", count_rmse_mean, count_rmse_std)
    @printf("Cell Count Neg. Log-lik:       %.4f ± %.4f\n", count_nll_mean, count_nll_std)

    # Overall performance (simple average - no normalization or negation)
    overall_mean = mean([acc_mean, npe_mean, y2_rmse_mean, y2_crps_mean, count_rmse_mean, count_nll_mean])
    overall_std = sqrt(mean([acc_std^2, npe_std^2, y2_rmse_std^2, y2_crps_std^2, count_rmse_std^2, count_nll_std^2]))

    println("\nOverall Performance:")
    println("-"^50)
    @printf("Average across metrics: %.4f ± %.4f\n", overall_mean, overall_std)

    # Find best performing fold (using simple average)
    if isnothing(best_fold_idx)
        fold_scores = []
        for perf in performances
            acc, npe = perf[1]
            y2_rmse, y2_crps = perf[2]
            y2_count_rmse, y2_count_nll = perf[3]
            score = mean([acc, npe, y2_rmse, y2_crps, y2_count_rmse, y2_count_nll])  # Simple average
            push!(fold_scores, score)
        end
        best_fold_idx = argmin(fold_scores)  # Or argmin depending on your preference
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
            sample_data_obs = (u_obs[:, :, sample_n:sample_n], covars_obs[:, :, sample_n:sample_n], x_obs[:, :, sample_n:sample_n],
                y₁_obs[:, :, sample_n:sample_n], y₂_obs[:, :, sample_n:sample_n],
                mask₁_obs[:, :, sample_n:sample_n], mask₂_obs[:, :, sample_n:sample_n])
            sample_future_true_data = (u_forecast[:, :, sample_n:sample_n], covars_forecast[:, :, sample_n:sample_n], x_forecast[:, :, sample_n:sample_n],
                y₁_forecast[:, :, sample_n:sample_n], y₂_forecast[:, :, sample_n:sample_n],
                mask₁_forecast[:, :, sample_n:sample_n], mask₂_forecast[:, :, sample_n:sample_n])


            timepoints_obs, timepoints_forecast = timepoints

            # Generate forecast
            sample_forecasted_data = forecast_fn(best_model, best_params, best_state, sample_data_obs,
                u_forecast[:, :, sample_n:sample_n], timepoints, config)

            # Create visualization and get sample metrics
            fig = viz_fn(timepoints_obs, timepoints_forecast,
                sample_data_obs, sample_future_true_data, sample_forecasted_data, normalization_stats)

            # Evaluate sample performance
            sample_metrics = eval_forecast(sample_future_true_data, sample_forecasted_data)
            sample_acc, sample_npe = sample_metrics[1]
            sample_y2_rmse, sample_y2_crps = sample_metrics[2]
            sample_y2_count_rmse, sample_y2_count_nll = sample_metrics[3]

            display(fig)

            println("\nSample #$sample_n Forecast Metrics:")
            println("-"^50)
            @printf("Health Status Accuracy: %.4f\n", sample_acc)
            @printf("Health Status NPE:      %.4f\n", sample_npe)
            @printf("Tumor volume RMSE:       %.4f\n", sample_y2_rmse)
            @printf("Tumor volume CRPS:       %.4f\n", sample_y2_crps)
            @printf("Cell Count RMSE:        %.4f\n", sample_y2_count_rmse)
            @printf("Cell Count NLL:         %.4f\n", sample_y2_count_nll)
        end
    end

    # Prepare return statistics
    return_stats = (
        acc_mean=acc_mean, acc_std=acc_std,
        npe_mean=npe_mean, npe_std=npe_std,
        y2_rmse_mean=y2_rmse_mean, y2_rmse_std=y2_rmse_std,
        y2_crps_mean=y2_crps_mean, y2_crps_std=y2_crps_std,
        count_rmse_mean=count_rmse_mean, count_rmse_std=count_rmse_std,
        count_nll_mean=count_nll_mean, count_nll_std=count_nll_std,
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
    - sort_by: Metric to sort by ("overall", "accuracy", "npe", "y2_rmse", "y2_crps", "count_rmse", "count_nll")
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
    elseif sort_by == "y2_rmse"
        sort_values = [stats.y2_rmse_mean for stats in model_stats]
    elseif sort_by == "y2_crps"
        sort_values = [stats.y2_crps_mean for stats in model_stats]
    elseif sort_by == "count_rmse"
        sort_values = [stats.count_rmse_mean for stats in model_stats]
    elseif sort_by == "count_nll"
        sort_values = [stats.count_nll_mean for stats in model_stats]
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
    println("\n" * "="^130)
    println("PKPD Model Comparison Summary (sorted by $(uppercase(sort_by)))")
    println("="^130)
    @printf("%-12s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-6s\n",
        "Model", "Accuracy", "NPE", "T-RMSE", "T-CRPS", "C-RMSE", "C-NLL", "Overall", "Rank")
    println("-"^130)

    # Find best values for each metric (based on typical interpretation)
    best_acc = maximum([stats.acc_mean for stats in model_stats])              # Higher is better
    best_npe = maximum([stats.npe_mean for stats in model_stats])              # Higher is better
    best_y2_rmse = minimum([stats.y2_rmse_mean for stats in model_stats])      # Lower is better
    best_y2_crps = minimum([stats.y2_crps_mean for stats in model_stats])      # Lower is better
    best_count_rmse = minimum([stats.count_rmse_mean for stats in model_stats]) # Lower is better
    best_count_nll = minimum([stats.count_nll_mean for stats in model_stats])   # Lower is better
    best_overall = minimum([stats.overall_mean for stats in model_stats])       # Lower is better (since most metrics are loss-like)

    for (rank, (name, stats)) in enumerate(zip(sorted_names, sorted_stats))
        # Add indicators for best performance
        acc_indicator = stats.acc_mean ≈ best_acc ? " ★" : ""
        npe_indicator = stats.npe_mean ≈ best_npe ? " ★" : ""
        y2_rmse_indicator = stats.y2_rmse_mean ≈ best_y2_rmse ? " ★" : ""
        y2_crps_indicator = stats.y2_crps_mean ≈ best_y2_crps ? " ★" : ""
        count_rmse_indicator = stats.count_rmse_mean ≈ best_count_rmse ? " ★" : ""
        count_nll_indicator = stats.count_nll_mean ≈ best_count_nll ? " ★" : ""
        overall_indicator = stats.overall_mean ≈ best_overall ? " ★" : ""

        @printf("%-12s | %.3f±%.3f%s | %.3f±%.3f%s | %.3f±%.3f%s | %.3f±%.3f%s | %.3f±%.3f%s | %.3f±%.3f%s | %.3f±%.3f%s | %-6d\n",
            name[1:min(12, length(name))],  # Truncate long names
            stats.acc_mean, stats.acc_std, acc_indicator,
            stats.npe_mean, stats.npe_std, npe_indicator,
            stats.y2_rmse_mean, stats.y2_rmse_std, y2_rmse_indicator,
            stats.y2_crps_mean, stats.y2_crps_std, y2_crps_indicator,
            stats.count_rmse_mean, stats.count_rmse_std, count_rmse_indicator,
            stats.count_nll_mean, stats.count_nll_std, count_nll_indicator,
            stats.overall_mean, stats.overall_std, overall_indicator,
            rank)
    end

    println("-"^130)
    println("★ = Best performance for that metric")
    println("Accuracy: Health status classification accuracy (higher is better)")
    println("NPE: Negative Predictive Entropy (higher is better)")
    println("T-RMSE: Tumor volume RMSE (lower is better)")
    println("T-CRPS: Tumor volume CRPS (lower is better)")
    println("C-RMSE: Cell count RMSE (lower is better)")
    println("C-NLL: Cell count Negative Log-likelihood (lower is better)")
    println("Overall: Average of all metrics")
    println("="^130)

    # Print best model info
    best_model_name = sorted_names[1]
    @info "Best model by $(sort_by): $best_model_name"

    # Return sorted results
    return Dict(zip(sorted_names, sorted_stats))
end