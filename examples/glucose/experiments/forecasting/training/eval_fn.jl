function eval_fn(model, θ, st, ts, data, config)
    u_obs, covars_obs, x_obs, y_obs, mask_obs, u_forecast, covars_forecast, x_forecast, y_forecast, mask_forecast = data

    ts_obs, ts_for = ts
    y_enc = vcat(covars_obs, y_obs, mask_obs)
    ŷ, _, kl_pq = model(y_enc, u_forecast, (ts_obs, ts_for), θ, st)

    μ, log_σ² = ŷ
    eval_loss = normal_loglikelihood(μ .* mask_forecast, log_σ² .* mask_forecast, y_forecast .* mask_forecast)
    eval_rmse = sqrt.(mse(μ .* mask_forecast, y_forecast .* mask_forecast))
    kl_val = kl_pq === nothing ? 0.0f0 : mean(kl_pq[end, :])
    return (eval_loss, eval_loss, eval_rmse, kl_val)
end

function eval_forecast(true_data, forecasted_data)
    _, _, _, y_f, mask_f = true_data
    x̂_mc, ŷ_mc = forecasted_data
    μ_mc, log_σ²_mc = ŷ_mc
    # Mean prediction across MC samples
    μ_m = dropmean(μ_mc, dims=4)
    # RMSE
    ŷ_rmse = sqrt.(mse(μ_m, y_f, mask_f))

    # CRPS
    ŷ_crps = empirical_crps(y_f, μ_mc, mask_f)

    return (ŷ_rmse, ŷ_crps)
end

function assess_model_performance(performances, variables_of_interest; model_name="Model",
    forecast_fn=forecast, plot_sample=false, sample_n=1, viz_fn=nothing,
    models=nothing, params=nothing, states=nothing, data=nothing, normalization_stats=nothing,
    timepoints=nothing, config=nothing, best_fold_idx=nothing)

    n_folds = length(performances)

    println("\n" * "="^60)
    println("$model_name Performance Summary ($n_folds-fold Cross-Validation)")
    println("="^60)

    rmse_values = [perf[1] for perf in performances]
    crps_values = [perf[2] for perf in performances]

    rmse_mean, rmse_std = mean(rmse_values), std(rmse_values)
    crps_mean, crps_std = mean(crps_values), std(crps_values)

    println("\nPerformance Metrics:")
    println("-"^40)
    @printf("Glucose RMSE:  %.4e ± %.4e\n", rmse_mean, rmse_std)
    @printf("Glucose CRPS:  %.4e ± %.4e\n", crps_mean, crps_std)

    overall_mean = mean([rmse_mean, crps_mean])
    overall_std = sqrt(mean([rmse_std^2, crps_std^2]))

    println("\nOverall Performance:")
    println("-"^40)
    @printf("Average: %.4e ± %.4e\n", overall_mean, overall_std)

    if isnothing(best_fold_idx)
        best_fold_idx = argmin(rmse_values)
    end
    println("Best performing fold: $best_fold_idx")
    println("="^60)

    fig = nothing
    if plot_sample && !any(isnothing.([models, params, states, data, timepoints, config, viz_fn]))
        println("\nGenerating sample forecast from best model (Fold $best_fold_idx)...")
        best_model = models[best_fold_idx]
        best_params = params[best_fold_idx]
        best_state = states[best_fold_idx]

        u_obs, covars_obs, x_obs, y_obs, mask_obs,
        u_forecast, covars_forecast, x_forecast, y_forecast, mask_forecast = data

        sample_data_obs = (u_obs[:, :, sample_n:sample_n], covars_obs[:, :, sample_n:sample_n],
            x_obs[:, :, sample_n:sample_n], y_obs[:, :, sample_n:sample_n], mask_obs[:, :, sample_n:sample_n])
        sample_future_true = (u_forecast[:, :, sample_n:sample_n], covars_forecast[:, :, sample_n:sample_n],
            x_forecast[:, :, sample_n:sample_n], y_forecast[:, :, sample_n:sample_n], mask_forecast[:, :, sample_n:sample_n])

        timepoints_obs, timepoints_forecast = timepoints
        sample_forecasted = forecast_fn(best_model, best_params, best_state, sample_data_obs,
            u_forecast[:, :, sample_n:sample_n], timepoints, config)

        fig = viz_fn(timepoints_obs, timepoints_forecast,
            sample_data_obs, sample_future_true, sample_forecasted, normalization_stats)

        sample_metrics = eval_forecast(sample_future_true, sample_forecasted)
        display(fig)

        @printf("\nSample #%d Forecast: RMSE=%.4e, CRPS=%.4e\n", sample_n, sample_metrics[1], sample_metrics[2])
    end

    return (rmse_mean=rmse_mean, rmse_std=rmse_std,
        crps_mean=crps_mean, crps_std=crps_std,
        overall_mean=overall_mean, overall_std=overall_std,
        best_fold_idx=best_fold_idx, figure=fig)
end

function compare_glucose_models(model_stats_dict; sort_by="overall", ascending=true)
    if isempty(model_stats_dict)
        @warn "No models provided for comparison"
        return model_stats_dict
    end

    model_names = collect(keys(model_stats_dict))
    model_stats = collect(values(model_stats_dict))

    if sort_by == "overall"
        sort_values = [s.overall_mean for s in model_stats]
    elseif sort_by == "rmse"
        sort_values = [s.rmse_mean for s in model_stats]
    elseif sort_by == "crps"
        sort_values = [s.crps_mean for s in model_stats]
    else
        @warn "Invalid sort_by='$sort_by'. Using 'overall'."
        sort_values = [s.overall_mean for s in model_stats]
    end

    order = ascending ? sortperm(sort_values) : sortperm(sort_values, rev=true)
    sorted_names = model_names[order]
    sorted_stats = model_stats[order]

    println("\n" * "="^70)
    println("Model Comparison (sorted by $sort_by, $(ascending ? "ascending" : "descending"))")
    println("="^70)
    @printf("%-15s  %12s  %12s  %12s\n", "Model", "RMSE", "CRPS", "Overall")
    println("-"^70)
    for (name, s) in zip(sorted_names, sorted_stats)
        @printf("%-15s  %.4e±%.4e  %.4e±%.4e  %.4e±%.4e\n",
            name, s.rmse_mean, s.rmse_std, s.crps_mean, s.crps_std, s.overall_mean, s.overall_std)
    end
    println("="^70)

    return Dict(zip(sorted_names, sorted_stats))
end
