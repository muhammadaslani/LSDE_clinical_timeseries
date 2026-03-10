# Validation loss — mirrors loss_fn but without KL, computed on forecast targets
function eval_fn(model, θ, st, ts, data, config)
    x_obs, u_obs, y_obs, y_masks_obs, x_masks_obs, x_fut, u_fut, y_fut, y_masks_fut, x_fut_masks = data
    batch_size = size(y_fut)[end]

    n_static = size(x_obs, 1) - size(x_masks_obs, 1)
    static_ones = ones(Float32, n_static, size(x_obs, 2), batch_size)
    x_mask_full = vcat(static_ones, Float32.(x_masks_obs))
    x_aug = vcat(x_obs, x_mask_full)

    ŷ, _, kl_pq = model(x_aug, u_fut, ts, θ, st)

    eval_losses = map(eachindex(ŷ)) do i
        μ, log_σ² = ŷ[i][1], ŷ[i][2]
        normal_loglikelihood(
            μ[1, :, :],
            log_σ²[1, :, :],
            y_fut[i, :, :],
            y_masks_fut[i, :, :]
        ) / batch_size
    end
    eval_loss = sum(eval_losses)
    kl_val = kl_pq === nothing ? 0.0f0 : mean(kl_pq[end, :])

    return (eval_loss, eval_losses..., kl_val)
end


# Forecast evaluation — assesses future predictions against held-out targets
function eval_forecast(true_data, forecasted_data)
    _, _, y_for, y_masks_for, _ = true_data
    _, Ey = forecasted_data

    rmse = Float64[]
    crps = Float64[]

    for i in eachindex(Ey)
        μ_i, log_σ²_i = Ey[i]
        σ_i = sqrt.(exp.(log_σ²_i))

        # RMSE: mean prediction across MC (E[y] = μ for Gaussian)
        μ_bar = sum(μ_i, dims=4) ./ size(μ_i, 4)         # [1, T, N, 1]
        push!(rmse, sqrt(mse(μ_bar[1, :, :, 1], y_for[i, :, :], y_masks_for[i, :, :])))

        # CRPS: sample from predictive distribution
        ŷ_i = μ_i .+ σ_i .* randn!(similar(σ_i))          # [1, T, N, mc]
        push!(crps, empirical_crps(y_for[i:i, :, :], ŷ_i, y_masks_for[i:i, :, :]))
    end

    return rmse, crps
end


function assess_model_performance(performances, variables_of_interest; model_name="Model",
    forecast_fn=forecast, plot_sample=false, sample_n=1, viz_fn=viz_fn,
    models=nothing, params=nothing, states=nothing, data=nothing,
    normalization_stats=nothing, timepoints=nothing, config=nothing, best_fold_idx=nothing)

    n_folds = length(performances)
    n_features = length(variables_of_interest)

    rmse_values = zeros(n_folds, n_features)
    crps_values = zeros(n_folds, n_features)

    for (fold_idx, (rmse, crps)) in enumerate(performances)
        rmse_values[fold_idx, :] = rmse[1:n_features]
        crps_values[fold_idx, :] = crps[1:n_features]
    end

    rmse_means = mean(rmse_values, dims=1)[1, :]
    rmse_stds = std(rmse_values, dims=1)[1, :]
    crps_means = mean(crps_values, dims=1)[1, :]
    crps_stds = std(crps_values, dims=1)[1, :]

    println("\n" * "="^60)
    println("$model_name Performance Summary ($n_folds-fold Cross-Validation)")
    println("="^60)

    println("\nRMSE (Root Mean Square Error):")
    println("-"^40)
    for (i, var) in enumerate(variables_of_interest)
        @printf("%-8s: %.4f ± %.4f\n", var, rmse_means[i], rmse_stds[i])
    end

    println("\nCRPS (Continuous Ranked Probability Score):")
    println("-"^40)
    for (i, var) in enumerate(variables_of_interest)
        @printf("%-8s: %.4f ± %.4f\n", var, crps_means[i], crps_stds[i])
    end

    overall_rmse_mean = mean(rmse_means)
    overall_rmse_std = sqrt(mean(rmse_stds .^ 2))
    overall_crps_mean = mean(crps_means)
    overall_crps_std = sqrt(mean(crps_stds .^ 2))

    println("\nOverall (average across variables):")
    println("-"^40)
    @printf("RMSE: %.4f ± %.4f\n", overall_rmse_mean, overall_rmse_std)
    @printf("CRPS: %.4f ± %.4f\n", overall_crps_mean, overall_crps_std)
    println("="^60)

    fig = nothing
    if plot_sample && !any(isnothing.([models, params, states, data, timepoints, config]))
        isnothing(best_fold_idx) && (best_fold_idx = argmin([mean(p[1]) for p in performances]))

        best_model = models[best_fold_idx]
        best_params = params[best_fold_idx]
        best_state = states[best_fold_idx]

        x_obs, u_obs, y_obs, y_masks_obs, x_masks_obs, x_for, u_for, y_for, y_masks_for, x_fut_masks = data
        timepoints_obs, timepoints_for = timepoints

        sample_data_obs = (x_obs[:, :, sample_n:sample_n], u_obs[:, :, sample_n:sample_n],
            y_obs[:, :, sample_n:sample_n], y_masks_obs[:, :, sample_n:sample_n],
            x_masks_obs[:, :, sample_n:sample_n])
        sample_future_true = (x_for[:, :, sample_n:sample_n], u_for[:, :, sample_n:sample_n],
            y_for[:, :, sample_n:sample_n], y_masks_for[:, :, sample_n:sample_n],
            x_fut_masks[:, :, sample_n:sample_n])

        Ex, Ey = forecast_fn(best_model, best_params, best_state, sample_data_obs,
            u_for[:, :, sample_n:sample_n], timepoints, config)

        fig = viz_fn(timepoints_obs, timepoints_for, sample_data_obs, sample_future_true, (Ex, Ey);
            normalization_stats=normalization_stats)
        display(fig)

        sample_rmse, sample_crps = eval_forecast(sample_future_true, (Ex, Ey))
        println("\nSample #$sample_n Forecast Metrics:")
        println("-"^40)
        for (i, var) in enumerate(variables_of_interest)
            @printf("%-8s: RMSE=%.4f, CRPS=%.4f\n", var, sample_rmse[i], sample_crps[i])
        end
    end

    return (rmse_means=rmse_means, rmse_stds=rmse_stds,
        crps_means=crps_means, crps_stds=crps_stds,
        overall_rmse_mean=overall_rmse_mean, overall_rmse_std=overall_rmse_std,
        overall_crps_mean=overall_crps_mean, overall_crps_std=overall_crps_std,
        variables=variables_of_interest,
        figure=fig)
end


function compare_models(model_stats_dict; sort_by="rmse", ascending=true)
    if isempty(model_stats_dict)
        @warn "No models provided for comparison"
        return model_stats_dict
    end

    model_names = collect(keys(model_stats_dict))
    model_stats = collect(values(model_stats_dict))

    sort_values = if sort_by == "rmse"
        [s.overall_rmse_mean for s in model_stats]
    elseif sort_by == "crps"
        [s.overall_crps_mean for s in model_stats]
    else
        @warn "Invalid sort_by '$sort_by', defaulting to 'rmse'"
        [s.overall_rmse_mean for s in model_stats]
    end

    sorted_idx = sortperm(sort_values, rev=!ascending)
    sorted_names = model_names[sorted_idx]
    sorted_stats = model_stats[sorted_idx]

    best_rmse = minimum(s.overall_rmse_mean for s in model_stats)
    best_crps = minimum(s.overall_crps_mean for s in model_stats)

    println("\n" * "="^80)
    println("ICU Model Comparison (sorted by $(uppercase(sort_by)))")
    println("="^80)

    variables = sorted_stats[1].variables

    @printf("\n--- OVERALL PERFORMANCE ---\n")
    @printf("%-15s | %-22s | %-22s | %s\n", "Model", "Avg RMSE", "Avg CRPS", "Rank")
    println("-"^80)

    for (rank, (name, s)) in enumerate(zip(sorted_names, sorted_stats))
        rmse_star = s.overall_rmse_mean ≈ best_rmse ? " ★" : ""
        crps_star = s.overall_crps_mean ≈ best_crps ? " ★" : ""
        @printf("%-15s | %.4f ± %.4f%-4s | %.4f ± %.4f%-4s | %d\n",
            name,
            s.overall_rmse_mean, s.overall_rmse_std, rmse_star,
            s.overall_crps_mean, s.overall_crps_std, crps_star,
            rank)
    end

    for (i, var) in enumerate(variables)
        best_var_rmse = minimum(s.rmse_means[i] for s in model_stats)
        best_var_crps = minimum(s.crps_means[i] for s in model_stats)

        @printf("\n--- Output: %s ---\n", var)
        @printf("%-15s | %-22s | %-22s\n", "Model", "RMSE", "CRPS")
        println("-"^80)
        for (name, s) in zip(sorted_names, sorted_stats)
            rmse_star = s.rmse_means[i] ≈ best_var_rmse ? " ★" : ""
            crps_star = s.crps_means[i] ≈ best_var_crps ? " ★" : ""
            @printf("%-15s | %.4f ± %.4f%-4s | %.4f ± %.4f%-4s\n",
                name,
                s.rmse_means[i], s.rmse_stds[i], rmse_star,
                s.crps_means[i], s.crps_stds[i], crps_star)
        end
    end

    println("-"^80)
    println("★ = Best for that metric")
    println("="^80)

    @info "Best model ($(sort_by)): $(sorted_names[1])"
    return Dict(zip(sorted_names, sorted_stats))
end