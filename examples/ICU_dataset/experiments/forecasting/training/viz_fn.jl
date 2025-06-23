
# Professional color palette for medical time series visualization
const MEDICAL_COLORS = (
    observed="#2E86AB",      # Deep blue for observations
    truth="#A23B72",         # Deep magenta for ground truth  
    predicted="#F18F01",     # Orange for predictions
    confidence="#C73E1D",    # Red for confidence intervals
    obs_period="#B8D4F0",    # More visible light blue for observation period
    forecast_period="#FFD4A3" # More visible light orange for forecast period
)

function viz_fn_forecast_nde(t_obs, t_for, obs_data, future_true_data, forecasted_data; sample_n=1, plot=true, confidence_level=0.9)
    u_obs, x_obs, y_obs, masks_obs = obs_data
    u_for, x_for, y_for, masks_for = future_true_data
    μ, σ = forecasted_data
    t_obs = t_obs * 20 
    t_for = t_for * 20 

    y_labels = ["MAP (mmHg)", "HR (bpm)", "Temperature (°C)"]
    # Optimized figure size and font settings for 800x600
    fig = Figure(size=(1200, 800), fontsize=20, 
                 backgroundcolor=:white,
                 figure_padding=20)
    axes = CairoMakie.Axis[]
    rmse = []
    crps = []

    n_features = length(y_labels)  # Assuming one label per feature
    
    # Calculate quantile levels for confidence interval
    alpha = 1 - confidence_level
    lower_quantile = alpha / 2
    upper_quantile = 1 - alpha / 2

    for i in 1:n_features
        y_label = y_labels[i]
        t_obs_val = t_obs[masks_obs[i, :, sample_n] .== 1]
        t_for_val = t_for[masks_for[i, :, sample_n] .== 1]

        dists = Normal.(μ[i], σ[i])
        ŷ = rand.(dists)

        ŷ_mean = dropdims(mean(ŷ, dims=4), dims=4)
        
        # Calculate quantile-based confidence intervals
        ŷ_ci_lower = dropdims(mapslices(x -> quantile(x, lower_quantile), ŷ, dims=4), dims=4)
        ŷ_ci_upper = dropdims(mapslices(x -> quantile(x, upper_quantile), ŷ, dims=4), dims=4)

        crps_ = empirical_crps(y_for[i:i, :, :], ŷ, masks_for[i:i, :, :])
        rmse_ = sqrt(mse(ŷ_mean[1,:,:], y_for[i,:,:], masks_for[i,:,:]))

        push!(crps, crps_)
        push!(rmse, rmse_)

        if plot
       
        # Check if the sample has any data points
        if !isempty(masks_for[i, :, sample_n])
            sample_rmse_ = sqrt(mse(ŷ_mean[1,:, sample_n], y_for[i, :, sample_n], masks_for[i, :, sample_n]))
            
            # For sample CRPS, extract predictions for this specific sample
            sample_ŷ = ŷ[:, :,  sample_n:sample_n,:]  # Keep 4D structure for empirical_crps
            sample_crps_ = empirical_crps(y_for[i:i, :, sample_n:sample_n], sample_ŷ, masks_for[i:i, :, sample_n:sample_n])
            println("Sample $sample_n - $(y_labels[i]) RMSE: $(round(sample_rmse_, digits=4))")
            println("Sample $sample_n - $(y_labels[i]) CRPS: $(round(sample_crps_, digits=4))")
        else
            sample_rmse_ = NaN
            sample_crps_ = NaN
            println("Sample $sample_n - $(y_labels[i]): No data available")
        end
            if isempty(t_obs_val) || isempty(t_for_val)
                @warn "Insufficient data for $y_label in sample $sample_n"
                continue
            
            else
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
                
                # Calculate data range for background polygons
                temp_pred_vals = ŷ_mean[1, masks_for[i, :, sample_n] .== 1, sample_n]
                temp_ci_lower = ŷ_ci_lower[1, masks_for[i, :, sample_n] .== 1, sample_n]
                temp_ci_upper = ŷ_ci_upper[1, masks_for[i, :, sample_n] .== 1, sample_n]
                
                temp_all_y_values = vcat(
                    y_obs[i, masks_obs[i, :, sample_n] .== 1, sample_n],
                    y_for[i, masks_for[i, :, sample_n] .== 1, sample_n],
                    temp_pred_vals,
                    temp_ci_lower,
                    temp_ci_upper
                )
                y_min_bg = minimum(temp_all_y_values) - 0.25 * (maximum(temp_all_y_values) - minimum(temp_all_y_values))
                y_max_bg = maximum(temp_all_y_values) + 0.25 * (maximum(temp_all_y_values) - minimum(temp_all_y_values))
                
                # Plot observation period background
                if i == 1
                    poly!(ax, [0, t_obs[end], t_obs[end], 0], 
                         [y_min_bg, y_min_bg, y_max_bg, y_max_bg], 
                         color=(MEDICAL_COLORS.obs_period, 0.3), 
                         label="Observation Period")
                    poly!(ax, [t_obs[end], t_for_val[end], t_for_val[end], t_obs[end]], 
                         [y_min_bg, y_min_bg, y_max_bg, y_max_bg], 
                         color=(MEDICAL_COLORS.forecast_period, 0.3), 
                         label="Forecasting Period")
                else
                    poly!(ax, [0, t_obs[end], t_obs[end], 0], 
                         [y_min_bg, y_min_bg, y_max_bg, y_max_bg], 
                         color=(MEDICAL_COLORS.obs_period, 0.3))
                    poly!(ax, [t_obs[end], t_for_val[end], t_for_val[end], t_obs[end]], 
                         [y_min_bg, y_min_bg, y_max_bg, y_max_bg], 
                         color=(MEDICAL_COLORS.forecast_period, 0.3))
                end
                
                # Plot historical observations
                scatter!(ax, t_obs_val, y_obs[i, masks_obs[i, :, sample_n] .== 1, sample_n], 
                        color=MEDICAL_COLORS.observed, 
                        label=i == 1 ? "Historical Observations" : "", 
                        markersize=16)
                lines!(ax, t_obs_val, y_obs[i, masks_obs[i, :, sample_n] .== 1, sample_n], 
                      color=(MEDICAL_COLORS.observed, 0.7), 
                      linewidth=2,
                      linestyle=:dash)

                # Plot ground truth future data
                scatter!(ax, t_for_val, y_for[i, masks_for[i, :, sample_n] .== 1, sample_n], 
                        color=MEDICAL_COLORS.truth, 
                        label=i == 1 ? "Ground Truth" : "", 
                        markersize=16)
                lines!(ax, t_for_val, y_for[i, masks_for[i, :, sample_n] .== 1, sample_n], 
                      color=(MEDICAL_COLORS.truth, 0.7), 
                      linewidth=2,
                      linestyle=:dash)

                # Plot model predictions with confidence intervals
                pred_vals = ŷ_mean[1, masks_for[i, :, sample_n] .== 1, sample_n]
                ci_lower = ŷ_ci_lower[1, masks_for[i, :, sample_n] .== 1, sample_n]
                ci_upper = ŷ_ci_upper[1, masks_for[i, :, sample_n] .== 1, sample_n]
                
                # Plot confidence band first (so it's behind other elements)
                band!(ax, t_for_val, ci_lower, ci_upper, 
                     color=(MEDICAL_COLORS.confidence, 0.25),
                     label=i == 1 ? "$(Int(confidence_level*100))% Confidence Interval" : "")
                
                # Plot prediction line and points
                lines!(ax, t_for_val, pred_vals, 
                      color=MEDICAL_COLORS.predicted, 
                      linewidth=2,
                      linestyle=:dash)
                scatter!(ax, t_for_val, pred_vals, 
                        color=MEDICAL_COLORS.predicted, 
                        label=i == 1 ? "Model Predictions" : "", 
                        markersize=16,
                        strokewidth=2,
                        strokecolor=:white)

                # Set axis limits with proper padding
                all_y_values = vcat(
                    y_obs[i, masks_obs[i, :, sample_n] .== 1, sample_n],
                    y_for[i, masks_for[i, :, sample_n] .== 1, sample_n],
                    pred_vals,
                    ci_lower,
                    ci_upper
                )

                y_range = maximum(all_y_values) - minimum(all_y_values)
                y_padding = max(0.15 * y_range, 0.01 * maximum(all_y_values))
                ylims!(ax, minimum(all_y_values) - y_padding, maximum(all_y_values) + y_padding)
                
                # Set x-axis limits to start from 0 and end exactly where data ends
                xlims!(ax, 0, t_for_val[end])
                
                # Add vertical separator line
                vlines!(ax, [t_obs[end]], color=("#666666", 0.8), linewidth=3, linestyle=:dash)
            end
        end
    end

    if plot
        
        # Create unified legend below the figures
        if !isempty(axes)
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
        end
        
        linkxaxes!(axes...)
        colsize!(fig.layout, 1, Relative(1.0))
        rowgap!(fig.layout, 15)
        colgap!(fig.layout, 10)
        display(fig)
        return fig, rmse, crps
    else
        return rmse, crps
    end
end


function viz_fn_forecast_rnn(t_obs, t_for, obs_data, future_true_data, forecasted_data; sample_n=1, plot=true)
    u_obs, x_obs, y_obs, masks_obs = obs_data
    u_for, x_for, y_for, masks_for = future_true_data
    μ, σ = forecasted_data

    t_obs = t_obs * 20 
    t_for = t_for * 20 

    y_labels = ["MAP (mmHg)", "HR (bpm)", "Temperature (°C)"]
    fig = Figure(size=(800, 600), fontsize=8, 
                 backgroundcolor=:white,
                 figure_padding=8)
    axes = CairoMakie.Axis[]
    rmse = []

    n_features = length(y_labels)  # Assuming one label per feature

    for i in 1:n_features
        y_label = y_labels[i]

        t_obs_val = t_obs[masks_obs[i, :, sample_n] .== 1]
        t_for_val = t_for[masks_for[i, :, sample_n] .== 1]

        # Generate predicted distributions (RNN format)

        dists = Normal.(μ[i], σ[i])
        ŷ = rand.(dists)
        ŷ = ŷ[1, :, :]
        # Need to collect all valid predictions and ground truth across all samples
        rmse_ = sqrt(mse(ŷ, y_for[i, :, :], masks_for[i, :, :]))

        push!(rmse, rmse_)

        if plot

            if isempty(t_obs_val) || isempty(t_for_val)
                @warn "Insufficient data for $y_label in sample $sample_n"
                continue
            else            
                # Calculate RMSE for the intended sample (for printing when plot=true)
                sample_rmse_ = sqrt(mse(ŷ[:,sample_n], y_for[i, :, sample_n], masks_for[i, :, sample_n]))
                # Print RMSE for this feature and intended sample
                println("Sample $sample_n - $(y_labels[i]) RMSE: $(round(sample_rmse_, digits=4))")

                ax = CairoMakie.Axis(fig[i, 1], 
                    xlabel=i == n_features ? "Time (hours)" : "",
                    ylabel=y_labels[i], 
                    xgridvisible=true, 
                    ygridvisible=true,
                    xgridcolor=("#E5E5E5", 0.8),
                    ygridcolor=("#E5E5E5", 0.8),
                    topspinevisible=false,
                    rightspinevisible=false,
                    xticklabelsize=7,
                    yticklabelsize=7,
                    xlabelsize=8,
                    ylabelsize=8)
                push!(axes, ax)
                
                # Calculate data range for background polygons  
                temp_all_y_values = vcat(
                    y_obs[i, masks_obs[i, :, sample_n] .== 1, sample_n],
                    y_for[i, masks_for[i, :, sample_n] .== 1, sample_n],
                    ŷ[masks_for[i, :, sample_n] .== 1, sample_n]
                )
                y_min_bg = minimum(temp_all_y_values) - 0.25 * (maximum(temp_all_y_values) - minimum(temp_all_y_values))
                y_max_bg = maximum(temp_all_y_values) + 0.25 * (maximum(temp_all_y_values) - minimum(temp_all_y_values))
                
                # Plot observation period background
                if i == 1
                    poly!(ax, [0, t_obs[end], t_obs[end], 0], 
                         [y_min_bg, y_min_bg, y_max_bg, y_max_bg], 
                         color=(MEDICAL_COLORS.obs_period, 0.3), 
                         label="Observation Period")
                    poly!(ax, [t_obs[end], t_for_val[end], t_for_val[end], t_obs[end]], 
                         [y_min_bg, y_min_bg, y_max_bg, y_max_bg], 
                         color=(MEDICAL_COLORS.forecast_period, 0.3), 
                         label="Forecasting Period")
                else
                    poly!(ax, [0, t_obs[end], t_obs[end], 0], 
                         [y_min_bg, y_min_bg, y_max_bg, y_max_bg], 
                         color=(MEDICAL_COLORS.obs_period, 0.3))
                    poly!(ax, [t_obs[end], t_for_val[end], t_for_val[end], t_obs[end]], 
                         [y_min_bg, y_min_bg, y_max_bg, y_max_bg], 
                         color=(MEDICAL_COLORS.forecast_period, 0.3))
                end
                
                # Plot historical observations
                scatter!(ax, t_obs_val, y_obs[i, masks_obs[i, :, sample_n] .== 1, sample_n], 
                        color=MEDICAL_COLORS.observed, 
                        label=i == 1 ? "Historical Observations" : "", 
                        markersize=8,
                        strokewidth=0.8,
                        strokecolor=:white)
                lines!(ax, t_obs_val, y_obs[i, masks_obs[i, :, sample_n] .== 1, sample_n], 
                      color=(MEDICAL_COLORS.observed, 0.7), 
                      linewidth=1.2,
                      linestyle=:dash)

                # Plot ground truth future data
                scatter!(ax, t_for_val, y_for[i, masks_for[i, :, sample_n] .== 1, sample_n], 
                        color=MEDICAL_COLORS.truth, 
                        label=i == 1 ? "Ground Truth" : "", 
                        markersize=8,
                        strokewidth=0.8,
                        strokecolor=:white)
                lines!(ax, t_for_val, y_for[i, masks_for[i, :, sample_n] .== 1, sample_n], 
                      color=(MEDICAL_COLORS.truth, 0.7), 
                      linewidth=1.2,
                      linestyle=:dash)

                # Plot RNN predictions (no confidence intervals)
                pred_vals = ŷ[masks_for[i, :, sample_n] .== 1, sample_n]
                lines!(ax, t_for_val, pred_vals, 
                      color=MEDICAL_COLORS.predicted, 
                      linewidth=1.2,
                      linestyle=:dash)
                scatter!(ax, t_for_val, pred_vals, 
                        color=MEDICAL_COLORS.predicted, 
                        label=i == 1 ? "RNN Predictions" : "", 
                        markersize=8,
                        strokewidth=0.8,
                        strokecolor=:white)

                # Set axis limits with proper padding
                all_y_values = vcat(
                    y_obs[i, masks_obs[i, :, sample_n] .== 1, sample_n],
                    y_for[i, masks_for[i, :, sample_n] .== 1, sample_n],
                    pred_vals
                )

                y_range = maximum(all_y_values) - minimum(all_y_values)
                y_padding = max(0.15 * y_range, 0.01 * maximum(all_y_values))
                ylims!(ax, minimum(all_y_values) - y_padding, maximum(all_y_values) + y_padding)
                
                # Set x-axis limits to start from 0 and end exactly where data ends
                xlims!(ax, 0, t_for_val[end])
                
                # Add vertical separator line
                vlines!(ax, [t_obs[end]], color=("#666666", 0.6), linewidth=2, linestyle=:dash)
            end
        end
    end

    if plot
        
        # Create unified legend below the figures
        if !isempty(axes)
            legend = Legend(fig, axes[1], 
                          orientation=:horizontal,
                          tellheight=true,
                          tellwidth=true,
                          margin=(10, 10, 10, 10),
                          framevisible=false,
                          labelsize=12,
                          halign=:center,
                          nbanks=1)
            fig[n_features + 1, 1] = legend
        end
        
        linkxaxes!(axes...)
        rowgap!(fig.layout, 15)
        colgap!(fig.layout, 10)
        display(fig)
        # RNN models only return RMSE, no CRPS
        return fig, rmse
    else
        # RNN models only return RMSE, no CRPS
        return rmse
    end
end