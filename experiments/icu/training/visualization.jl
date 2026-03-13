
# Professional color palette for medical time series visualization
const MEDICAL_COLORS = (
    observed="#2E86AB",      # Deep blue for observations
    truth="#A23B72",         # Deep magenta for ground truth  
    predicted="#F18F01",     # Orange for predictions
    confidence="#C73E1D",    # Red for confidence intervals
    obs_period="#B8D4F0",    # More visible light blue for observation period
    forecast_period="#FFD4A3" # More visible light orange for forecast period
)



function viz_fn(t_obs, t_for, obs_data, future_true_data, forecasted_data; sample_n=1, plot=true, confidence_level=0.9, normalization_stats=nothing)
    _, _, y_obs, masks_obs = obs_data
    _, _, y_for, masks_for = future_true_data
    _, Ey = forecasted_data   # discard latent trajectories; Ey[i] = (μ_i, log_σ²_i)

    dt = t_for[2] - t_for[1]
    t_obs = t_obs / dt
    t_for = t_for / dt
    y_labels = ["MAP (mmHg)", "HR (bpm)", "Temperature (°C)"]
    n_mc_samples = size(Ey[1][1], 4)
    rmse = []
    crps = []
    n_features = length(y_labels)

    # Initialize plotting variables
    fig = nothing
    axes = CairoMakie.Axis[]

    # Display denormalization message once
    if normalization_stats !== nothing
        @info "Denormalizing all data for visualization"
    else
        @info "Using original (non-normalized) data for visualization"
    end

    for i in 1:n_features
        y_label = y_labels[i]
        t_obs_val = t_obs[masks_obs[i, :, 1] .== 1]
        t_for_val = t_for[masks_for[i, :, 1] .== 1]
        
        # Denormalize all data if normalization_stats is provided
        if normalization_stats !== nothing 
            y_stats = normalization_stats["y_stats"]
            
            # Denormalize predictions
            μ_i    = Ey[i][1]
            σ²_i   = exp.(Ey[i][2])
            μ_denorm  = μ_i  .* y_stats.σ[i] .+ y_stats.μ[i]
            σ²_denorm = σ²_i .* (y_stats.σ[i])^2
            dists = Normal.(μ_denorm, sqrt.(σ²_denorm))
            
            # Create denormalized copies (don't modify originals)
            y_obs_denorm = y_obs[i, :, :] .* y_stats.σ[i] .+ y_stats.μ[i]
            y_for_denorm = y_for[i, :, :] .* y_stats.σ[i] .+ y_stats.μ[i]
        else
            dists = Normal.(Ey[i][1], sqrt.(exp.(Ey[i][2])))
            # Use original data as is
            y_obs_denorm = y_obs[i, :, :]
            y_for_denorm = y_for[i, :, :]
        end
        
        ŷ = Float32.(rand.(dists))
        ŷ_mean = dropdims(mean(ŷ, dims=4), dims=4)
        ŷ_std = dropdims(std(ŷ, dims=4), dims=4)
        # Calculate confidence intervals

        ŷ_ci_lower, ŷ_ci_upper = ŷ_mean.*masks_for[i:i,:,:] .- 1.96 .* ŷ_std.*masks_for[i:i,:,:]/sqrt(n_mc_samples), ŷ_mean .+ 1.96 .* ŷ_std.*masks_for[i:i,:,:]/sqrt(n_mc_samples)
        # Calculate metrics using denormalized data
        if !isempty(masks_for[i, :, 1])
            sample_rmse_ = sqrt(mse(ŷ_mean[1, :, 1], y_for_denorm[:, 1], masks_for[i, :, 1]))
            sample_ŷ = ŷ[:, :, 1:1, :] # Keep 4D structure for empirical_crps
            sample_crps_ = empirical_crps(reshape(y_for_denorm[:, 1:1], 1, :, 1), sample_ŷ, masks_for[i:i, :, 1:1])
        else
            sample_rmse_ = NaN
            sample_crps_ = NaN
            println("Sample $sample_n - $(y_labels[i]): No data available")
        end

        push!(rmse, sample_rmse_)
        push!(crps, sample_crps_)

        # Unified plotting logic using denormalized data
        if plot && !isempty(t_obs_val) && !isempty(t_for_val)
            # Initialize figure only on the first valid iteration
            if fig === nothing
                fig = Figure(size=(1500, 800), fontsize=20,
                             backgroundcolor=:white,
                             figure_padding=20)
                axes = CairoMakie.Axis[]
            end

            ax = CairoMakie.Axis(fig[i, 1],
                xlabel=i == n_features ? "Time (hours)" : "",
                ylabel=y_labels[i],
                xgridvisible=true,
                ygridvisible=true,
                xgridcolor=("#E5E5E5", 0.8),
                ygridcolor=("#E5E5E5", 0.8),
                topspinevisible=false,
                rightspinevisible=false,
                xticklabelsize=16,
                yticklabelsize=16,
                xlabelsize=16,
                ylabelsize=16,
                titlesize=16,
                spinewidth=1.0)
            push!(axes, ax)

            # Calculate data range for background polygons using denormalized data
            temp_pred_vals = ŷ_mean[1, masks_for[i, :, 1] .== 1, 1]
            temp_ci_lower = ŷ_ci_lower[1, masks_for[i, :, 1] .== 1, 1]
            temp_ci_upper = ŷ_ci_upper[1, masks_for[i, :, 1] .== 1, 1]

            temp_all_y_values = vcat(
                y_obs_denorm[masks_obs[i, :, 1] .== 1, 1],
                y_for_denorm[masks_for[i, :, 1] .== 1, 1],
                temp_pred_vals,
                temp_ci_lower,
                temp_ci_upper
            )
            y_min_bg = minimum(temp_all_y_values) - 0.25 * (maximum(temp_all_y_values) - minimum(temp_all_y_values))
            y_max_bg = maximum(temp_all_y_values) + 0.25 * (maximum(temp_all_y_values) - minimum(temp_all_y_values))

            # Plot observation and forecast period backgrounds
            poly!(ax, [0, t_obs[end], t_obs[end], 0],
                 [y_min_bg, y_min_bg, y_max_bg, y_max_bg],
                 color=(MEDICAL_COLORS.obs_period, 0.3),
                 label=i == 1 ? "Observation Period" : "")
            poly!(ax, [t_obs[end], t_for_val[end], t_for_val[end], t_obs[end]],
                 [y_min_bg, y_min_bg, y_max_bg, y_max_bg],
                 color=(MEDICAL_COLORS.forecast_period, 0.3),
                 label=i == 1 ? "Forecasting Period" : "")

            # Plot historical observations using denormalized data
            scatter!(ax, t_obs_val, y_obs_denorm[masks_obs[i, :, 1] .== 1, 1],
                    color=MEDICAL_COLORS.observed,
                    label=i == 1 ? "Historical Observations" : "",
                    markersize=16)
            lines!(ax, t_obs_val, y_obs_denorm[masks_obs[i, :, 1] .== 1, 1],
                  color=(MEDICAL_COLORS.observed, 0.7),
                  linewidth=2,
                  linestyle=:dash)

            # Plot ground truth future data using denormalized data
            scatter!(ax, t_for_val, y_for_denorm[masks_for[i, :, 1] .== 1, 1],
                    color=MEDICAL_COLORS.truth,
                    label=i == 1 ? "Ground Truth" : "",
                    markersize=16)
            lines!(ax, t_for_val, y_for_denorm[masks_for[i, :, 1] .== 1, 1],
                  color=(MEDICAL_COLORS.truth, 0.7),
                  linewidth=2,
                  linestyle=:dash)

            # Plot model predictions with confidence intervals (already denormalized)
            pred_vals = ŷ_mean[1, masks_for[i, :, 1] .== 1, 1]
            ci_lower = ŷ_ci_lower[1, masks_for[i, :, 1] .== 1, 1]
            ci_upper = ŷ_ci_upper[1, masks_for[i, :, 1] .== 1, 1]

            band!(ax, t_for_val, ci_lower, ci_upper,
                 color=(MEDICAL_COLORS.confidence, 0.25),
                 label=i == 1 ? "$(Int(confidence_level*100))% Confidence Interval" : "")

            lines!(ax, t_for_val, pred_vals,
                  color=MEDICAL_COLORS.predicted,
                  linewidth=2,
                  linestyle=:dash)
            scatter!(ax, t_for_val, pred_vals,
                    color=MEDICAL_COLORS.predicted,
                    label=i == 1 ? "Model Predictions" : "",
                    markersize=16)

            # Set axis limits with proper padding using denormalized data
            all_y_values = vcat(
                y_obs_denorm[masks_obs[i, :, 1] .== 1, 1],
                y_for_denorm[masks_for[i, :, 1] .== 1, 1],
                pred_vals,
                ci_lower,
                ci_upper
            )
            y_range = maximum(all_y_values) - minimum(all_y_values)
            y_padding = max(0.15 * y_range, 0.01 * maximum(all_y_values))
            ylims!(ax, minimum(all_y_values) - y_padding, maximum(all_y_values) + y_padding)

            xlims!(ax, 0, t_for_val[end])

            # Add vertical separator line
            vlines!(ax, [t_obs[end]], color=("#666666", 0.8), linewidth=3, linestyle=:dash)
        elseif plot && (isempty(t_obs_val) || isempty(t_for_val))
            @warn "Insufficient data for $y_label in sample $sample_n"
        end
    end

    # Final plotting setup and return
    if plot && !isempty(axes)
        # Create unified legend below the figures
        legend = Legend(fig, axes[1],
                        orientation=:horizontal,
                        tellheight=true,
                        tellwidth=true,
                        margin=(10, 10, 10, 10),
                        framevisible=false,
                        labelsize=16,
                        halign=:center,
                        nbanks=1)
        fig[n_features + 1, 1] = legend

        linkxaxes!(axes...)
        colsize!(fig.layout, 1, Relative(1.0))
        rowgap!(fig.layout, 15)
        colgap!(fig.layout, 10)
        display(fig)
        
        # Display calculated metrics for each feature
        println("=== Sample $sample_n Metrics ===")
        for (i, y_label) in enumerate(y_labels)
            if !isnan(rmse[i]) && !isnan(crps[i])
                println("$(y_label): RMSE = $(round(rmse[i], digits=4)), CRPS = $(round(crps[i], digits=4))")
            else
                println("$(y_label): No data available for metrics calculation")
            end
        end
        println("================================")
        
        return fig, rmse, crps
    end

    return rmse, crps
end