# Professional color palette for PKPD time series visualization
const PKPD_COLORS = (
    observed = "#2E86AB",          # Deep blue for observations
    truth = "#A23B72",             # Deep magenta for ground truth  
    predicted = "#F18F01",         # Orange for predictions
    confidence = "#C73E1D",        # Red for confidence intervals
    obs_period = "#B8D4F0",        # Light blue for observation period
    forecast_period = "#FFD4A3",   # Light orange for forecast period
    tumor = "#E63946",             # Red for tumor volume
    health = "#06D6A0"             # Green for health score
)

# Neural differential equation visualization function
function viz_fn(obs_timepoints, for_timepoints, obs_data, future_true_data, forecasted_data; sample_n=3, plot=true)
    # Unpack observation data
    u_o, covars_o, x_o, y₁_o, y₂_o, mask₁_o, mask₂_o = obs_data
    
    # Unpack future data  
    u_t, covars_t, x_t, y₁_t, y₂_t, mask₁_t, mask₂_t = future_true_data
    
    # Unpack forecasted data
    u_p = u_t
    Ex, Ey_p = forecasted_data

    Ey₁_p, Ey₂_p = softmax(Ey_p[1], dims=1), Ey_p[2]
    
    # Convert time to days for plotting
    t_o, t_p = obs_timepoints *  7.0f0, for_timepoints * 7.0f0

    # Convert health status to classes
    y₁_o_class = onecold(softmax(y₁_o, dims=1), Array(0:5))
    y₁_t_class = onecold(softmax(y₁_t, dims=1), Array(0:5))

    # Calculate prediction results
    ŷ₁_m = dropmean(Ey₁_p, dims=4)
    ŷ₁_s = dropmean(std(Ey₁_p, dims=4), dims=4)
    ŷ₁_class = onecold(ŷ₁_m, Array(0:5))
    ŷ₁_conf = maximum(ŷ₁_m, dims=1)[1, :, :]
    ŷ₂_m = dropmean(Ey₂_p, dims=4)
    ŷ₂_s = dropmean(std(Ey₂_p, dims=4), dims=4)
    ŷ₂_count = rand.(Poisson.(Ey₂_p))
    ŷ₂_count_m = dropmean(ŷ₂_count, dims=4)

    # Find valid time points for each output
    t_o_valid₁ = t_o[mask₁_o[1, :, sample_n] .== 1]
    t_p_valid₁ = t_p[mask₁_t[1, :, sample_n] .== 1]
    max_t_o_valid₁ = isempty(t_o_valid₁) ? 0.0 : maximum(t_o_valid₁)
    max_t_p_valid₁ = isempty(t_p_valid₁) ? 0.0 : maximum(t_p_valid₁)
    
    t_o_valid₂ = t_o[mask₂_o[1, :, sample_n] .== 1]
    t_p_valid₂ = t_p[mask₂_t[1, :, sample_n] .== 1]
    max_t_o_valid₂ = isempty(t_o_valid₂) ? 0.0 : maximum(t_o_valid₂)
    max_t_p_valid₂ = isempty(t_p_valid₂) ? 0.0 : maximum(t_p_valid₂)

    # Extract valid data points
    y₁_o_class_valid = y₁_o_class[findall(i -> t_o[i] <= max_t_o_valid₁ && mask₁_o[1, i, sample_n] == 1, 1:length(t_o)), sample_n]
    y₁_t_class_valid = y₁_t_class[findall(i -> t_p[i] <= max_t_p_valid₁ && mask₁_t[1, i, sample_n] == 1, 1:length(t_p)), sample_n]
    ŷ₁_class_valid = ŷ₁_class[findall(i -> t_p[i] <= max_t_p_valid₁ && mask₁_t[1, i, sample_n] == 1, 1:length(t_p)), sample_n]
    ŷ₁_conf_valid = ŷ₁_conf[findall(i -> t_p[i] <= max_t_p_valid₁ && mask₁_t[1, i, sample_n] == 1, 1:length(t_p)), sample_n]

    y₂_o_valid = y₂_o[1, findall(i -> t_o[i] <= max_t_o_valid₂ && mask₂_o[1, i, sample_n] == 1, 1:length(t_o)), sample_n]
    y₂_t_valid = y₂_t[1, findall(i -> t_p[i] <= max_t_p_valid₂ && mask₂_t[1, i, sample_n] == 1, 1:length(t_p)), sample_n]
    ŷ₂_m_valid = ŷ₂_m[1, findall(i -> t_p[i] <= max_t_p_valid₂ && mask₂_t[1, i, sample_n] == 1, 1:length(t_p)), sample_n]
    ŷ₂_s_valid = ŷ₂_s[1, findall(i -> t_p[i] <= max_t_p_valid₂ && mask₂_t[1, i, sample_n] == 1, 1:length(t_p)), sample_n]
    ŷ₂_count_m_valid = ŷ₂_count_m[1, findall(i -> t_p[i] <= max_t_p_valid₂ && mask₂_t[1, i, sample_n] == 1, 1:length(t_p)), sample_n]
    ŷ₂_count_valid = ŷ₂_count[1, findall(i -> t_p[i] <= max_t_p_valid₂ && mask₂_t[1, i, sample_n] == 1, 1:length(t_p)), sample_n, :]
    # Calculate confidence intervals
    ŷ₂_CI_low = ŷ₂_m[1, :, sample_n] .- 1.96 * ŷ₂_s[1, :, sample_n]
    ŷ₂_CI_up = ŷ₂_m[1, :, sample_n] .+ 1.96 * ŷ₂_s[1, :, sample_n]
    ŷ₂_count_confidence_valid = 1.96 * sqrt.(ŷ₂_m_valid)

    # Calculate performance metrics only on valid time points
    crossentropy_health = 0.0
    rmse_tumor = 0.0
    nll_count = 0.0
    
    try
        # Health status (y₁): cross entropy - only on valid time points
        crossentropy_health = CrossEntropy_Loss(ŷ₁_m, y₁_t, mask₁_t; agg=sum, logits=false)/length(findall(mask₁_t .== 1))
        
        # Tumor volume (y₂_m): RMSE - only on valid time points
        rmse_tumor = sqrt(mse(ŷ₂_m, y₂_t, mask₂_t))

        # Cell count (y₂_count): negative log likelihood - only on valid time points
        # nll_count = -poisson_loglikelihood(ŷ₂_m, y₂_t, mask₂_t)/length(findall(mask₂_t .== 1))
        nll_count = -poisson_loglikelihood_multiple_samples(clamp.(Ey₂_p, 0.0, 100.0), y₂_t) / length(findall(mask₂_t .== 1))
        
        if plot
            println("Health status cross entropy: ", crossentropy_health)
            println("Tumor volume RMSE: ", rmse_tumor)
            println("Cell count negative log likelihood: ", nll_count)
        end
    catch e
        if plot
            println("Warning: Could not calculate performance metrics: ", e)
        end
    end

    # Early return if not plotting - return the three specific metrics
    if !plot
        return crossentropy_health, rmse_tumor, nll_count
    end

    # Find treatment indices
    valid_indices_chemo_o = findall(i -> u_o[1, i, sample_n] == 1 && t_o[i] <= max_t_o_valid₁, 1:length(t_o))
    valid_indices_radio_o = findall(i -> u_o[2, i, sample_n] == 1 && t_o[i] <= max_t_o_valid₁, 1:length(t_o))
    valid_indices_chemo_p = findall(i -> u_p[1, i, sample_n] == 1 && t_p[i] <= max_t_p_valid₁, 1:length(t_p))
    valid_indices_radio_p = findall(i -> u_p[2, i, sample_n] == 1 && t_p[i] <= max_t_p_valid₁, 1:length(t_p))

    # Calculate dynamic plot limits based on actual data (ICU style)
    x_min = 0.0
    x_max = max_t_p_valid₁ > 0 ? max_t_p_valid₁ + 0.05 * max_t_p_valid₁ : 10.0  # Add 5% padding or default
    
    # Calculate y-limits for each panel based on actual data with padding
    # Panel 1: Health status - based on class range with padding
    all_health_values = vcat(y₁_o_class_valid, y₁_t_class_valid, ŷ₁_class_valid)
    if !isempty(all_health_values)
        health_range = maximum(all_health_values) - minimum(all_health_values)
        health_padding = max(0.25 * health_range, 0.5)  # At least 0.5 padding
        y1_min = minimum(all_health_values) - health_padding
        y1_max = maximum(all_health_values) + health_padding
    else
        y1_min, y1_max = -0.5, 3.5  # Default range for health status
    end
    
    # Panel 2: Tumor size - based on all tumor data with padding
    all_tumor_values = vcat(x_o[1, :, sample_n], x_t[1, :, sample_n], ŷ₂_m_valid, ŷ₂_CI_low, ŷ₂_CI_up)
    if !isempty(all_tumor_values)
        tumor_range = maximum(all_tumor_values) - minimum(all_tumor_values)
        tumor_padding = max(0.25 * tumor_range, 0.1 * maximum(all_tumor_values))
        y2_min = minimum(all_tumor_values) - tumor_padding
        y2_max = maximum(all_tumor_values) + tumor_padding
    else
        y2_min, y2_max = -0.5, 5.0  # Default range for tumor size
    end
    
    # Panel 3: Cell count - based on all count data with padding  
    all_count_values = vcat(y₂_o_valid, y₂_t_valid, ŷ₂_count_m_valid)
    if !isempty(all_count_values)
        count_range = maximum(all_count_values) - minimum(all_count_values)
        count_padding = max(0.25 * count_range, 0.1 * maximum(all_count_values))
        y3_min = minimum(all_count_values) - count_padding
        y3_max = maximum(all_count_values) + count_padding
    else
        y3_min, y3_max = -5.0, 50.0  # Default range for cell count
    end

    # Create the 3-panel figure with professional styling (integrated timeline approach)
    fig = Figure(size=(1200, 800), fontsize=20,
                 backgroundcolor=:white,
                 figure_padding=10)
    
    # Panel 1: Health status
    ax1 = CairoMakie.Axis(fig[1, 1], 
                         xlabel="", 
                         ylabel="Health Status",
                         xgridvisible=true, 
                         ygridvisible=true,
                         xgridcolor=("#E5E5E5", 0.8),
                         ygridcolor=("#E5E5E5", 0.8),
                         topspinevisible=false,
                         rightspinevisible=false,
                         xticklabelsize=16,
                         yticklabelsize=16,
                         xlabelsize=20,
                         ylabelsize=20)

    # Panel 2: Tumor size
    ax2 = CairoMakie.Axis(fig[2, 1], 
                         xlabel="", 
                         ylabel="Tumor Size (Unobserved)",
                         xgridvisible=true, 
                         ygridvisible=true,
                         xgridcolor=("#E5E5E5", 0.8),
                         ygridcolor=("#E5E5E5", 0.8),
                         topspinevisible=false,
                         rightspinevisible=false,
                         xticklabelsize=16,
                         yticklabelsize=16,
                         xlabelsize=20,
                         ylabelsize=20)

    # Panel 3: Cell count
    ax3 = CairoMakie.Axis(fig[3, 1], 
                         xlabel="Time (days)", 
                         ylabel="Cell Count",
                         xgridvisible=true, 
                         ygridvisible=true,
                         xgridcolor=("#E5E5E5", 0.8),
                         ygridcolor=("#E5E5E5", 0.8),
                         topspinevisible=false,
                         rightspinevisible=false,
                         xticklabelsize=16,
                         yticklabelsize=16,
                         xlabelsize=20,
                         ylabelsize=20)

    # Add background periods and intervention lines for all panels with dynamic y-limits
    y_limits = [(y1_min, y1_max), (y2_min, y2_max), (y3_min, y3_max)]
    
    for (i, (ax, (y_bg_min, y_bg_max))) in enumerate(zip([ax1, ax2, ax3], y_limits))
        # Observation period background
        if i == 1
            poly!(ax, [x_min, max_t_o_valid₁, max_t_o_valid₁, x_min], 
                 [y_bg_min, y_bg_min, y_bg_max, y_bg_max], 
                 color=(PKPD_COLORS.obs_period, 0.3), 
                 label="Observation Period")
            poly!(ax, [max_t_o_valid₁, x_max, x_max, max_t_o_valid₁], 
                 [y_bg_min, y_bg_min, y_bg_max, y_bg_max], 
                 color=(PKPD_COLORS.forecast_period, 0.3), 
                 label="Forecasting Period")
        else
            poly!(ax, [x_min, max_t_o_valid₁, max_t_o_valid₁, x_min], 
                 [y_bg_min, y_bg_min, y_bg_max, y_bg_max], 
                 color=(PKPD_COLORS.obs_period, 0.3))
            poly!(ax, [max_t_o_valid₁, x_max, x_max, max_t_o_valid₁], 
                 [y_bg_min, y_bg_min, y_bg_max, y_bg_max], 
                 color=(PKPD_COLORS.forecast_period, 0.3))
        end
        
        # Add vertical separator line like in ICU version
        vlines!(ax, [max_t_o_valid₁], color=("#666666", 0.8), linewidth=5, linestyle=:dash)
        
        # Add intervention lines across all panels for integrated timeline
        if i == 1  # Only add to legend once
            # Chemotherapy intervention lines
            vlines!(ax, t_o[valid_indices_chemo_o], color=:navy, linewidth=4.5, alpha=0.8,
                   label="Chemotherapy Sessions")
            vlines!(ax, t_p[valid_indices_chemo_p], color=:navy, linewidth=4.5, alpha=0.8)
            
            # Radiotherapy intervention lines  
            vlines!(ax, t_o[valid_indices_radio_o], color=:darkred, linewidth=4.5, alpha=0.8,
                   label="Radiotherapy Sessions")
            vlines!(ax, t_p[valid_indices_radio_p], color=:darkred, linewidth=4.5, alpha=0.8)
        else
            # No labels for subsequent panels
            vlines!(ax, t_o[valid_indices_chemo_o], color=:navy, linewidth=4.5, alpha=0.8)
            vlines!(ax, t_p[valid_indices_chemo_p], color=:navy, linewidth=4.5, alpha=0.8)
            vlines!(ax, t_o[valid_indices_radio_o], color=:darkred, linewidth=4.5, alpha=0.8)
            vlines!(ax, t_p[valid_indices_radio_p], color=:darkred, linewidth=4.5, alpha=0.8)
        end
    end

    # Plot health status
    if !isempty(t_o_valid₁) && !isempty(y₁_o_class_valid)
        scatter!(ax1, t_o_valid₁, y₁_o_class_valid, 
                color=PKPD_COLORS.observed, markersize=20,
                label="Historical Observations")
        lines!(ax1, t_o_valid₁, y₁_o_class_valid, 
              color=(PKPD_COLORS.observed, 0.7), linewidth=3.5, linestyle=:dash)
    end
          
    if !isempty(t_p_valid₁) && !isempty(y₁_t_class_valid)
        scatter!(ax1, t_p_valid₁, y₁_t_class_valid, 
                color=PKPD_COLORS.truth, markersize=20,
                label="Ground Truth")
        lines!(ax1, t_p_valid₁, y₁_t_class_valid, 
              color=(PKPD_COLORS.truth, 0.7), linewidth=3.5, linestyle=:dash)
    end
          
    if !isempty(t_p_valid₁) && !isempty(ŷ₁_class_valid)
        scatter!(ax1, t_p_valid₁, ŷ₁_class_valid, 
                color=PKPD_COLORS.predicted, markersize=20,
                label="Model Predictions")
        lines!(ax1, t_p_valid₁, ŷ₁_class_valid, 
              color=PKPD_COLORS.predicted, linewidth=3.5, linestyle=:dash)
    end
          
    if !isempty(t_p_valid₁) && !isempty(ŷ₁_class_valid) && !isempty(ŷ₁_conf_valid)
        errorbars!(ax1, t_p_valid₁, ŷ₁_class_valid, ŷ₁_conf_valid, 
                  color=(PKPD_COLORS.confidence, 0.6), whiskerwidth=12)
    end

    # Plot observed tumor size (historical data) - use index-based plotting for underlying tumor size
    lines!(ax2, Array(0:length(vcat(x_o[1, :, sample_n], x_t[1, :, sample_n]))-1), 
          vcat(x_o[1, :, sample_n], x_t[1, :, sample_n]), 
          color=(PKPD_COLORS.observed, 0.7), linewidth=3.5, linestyle=:solid,
          label="Historical Observations")
    # Plot confidence band first (so it's behind other elements)
    band!(ax2, t_p, ŷ₂_CI_low, ŷ₂_CI_up, 
         color=(PKPD_COLORS.confidence, 0.25))
         
    lines!(ax2, t_p, ŷ₂_m[1, :, sample_n], 
          color=PKPD_COLORS.predicted, linewidth=2.5, linestyle=:solid,
          label="Model Predictions")
    # Plot cell count with professional styling
    scatter!(ax3, t_o_valid₂, y₂_o_valid,
            color=PKPD_COLORS.observed, markersize=20)
    lines!(ax3, t_o_valid₂, y₂_o_valid, 
          color=(PKPD_COLORS.observed, 0.7), linewidth=3.5, linestyle=:dash)

    scatter!(ax3, t_p_valid₂, y₂_t_valid, 
            color=PKPD_COLORS.truth, markersize=20)
    lines!(ax3, t_p_valid₂, y₂_t_valid, 
          color=(PKPD_COLORS.truth, 0.7), linewidth=3.5, linestyle=:dash)
    scatter!(ax3, t_p_valid₂, ŷ₂_count_m_valid, 
            color=PKPD_COLORS.predicted, markersize=20)
    lines!(ax3, t_p_valid₂, ŷ₂_count_m_valid, 
          color=PKPD_COLORS.predicted, linewidth=3.5, linestyle=:dash)

    errorbars!(ax3, t_p_valid₂, ŷ₂_count_m_valid, ŷ₂_count_confidence_valid, 
              color=(PKPD_COLORS.confidence, 0.6), whiskerwidth=20)

    # Link x-axes for synchronized zooming/panning
    #linkxaxes!(ax1, ax2, ax3)
    
    # Set axis limits based on actual data range (ICU style)
    x_data_max = max_t_p_valid₁ > 0 ? max_t_p_valid₁ : 10.0
    xlims!(ax1, 0, x_data_max)
    xlims!(ax2, 0, x_data_max)
    xlims!(ax3, 0, x_data_max)
    ylims!(ax1, y1_min, y1_max)
    ylims!(ax2, y2_min, y2_max)
    ylims!(ax3, y3_min, y3_max)
    
    # Create consolidated legend at the bottom (ICU style)
    custom_elements = [
        PolyElement(color=(PKPD_COLORS.obs_period, 0.3)),
        PolyElement(color=(PKPD_COLORS.forecast_period, 0.3)),
        MarkerElement(color=PKPD_COLORS.observed, marker=:circle, markersize=20),
        MarkerElement(color=PKPD_COLORS.truth, marker=:circle, markersize=20), 
        MarkerElement(color=PKPD_COLORS.predicted, marker=:circle, markersize=20),
        LineElement(color=(PKPD_COLORS.confidence, 0.4), linewidth=16, linestyle=:solid),  # Thicker line to represent error bars
        LineElement(color=:navy, linewidth=4.5),
        LineElement(color=:darkred, linewidth=4.5)
    ]
    
    custom_labels = [
        "Observation Period",
        "Forecasting Period", 
        "Historical Observations",
        "Ground Truth", 
        "Model Predictions",
        "Prediction Uncertainty",
        "Chemotherapy Sessions",
        "Radiotherapy Sessions"
    ]
    
    legend = Legend(fig, custom_elements, custom_labels,
                  orientation=:horizontal,
                  tellheight=true,
                  tellwidth=true,
                  margin=(10, 10, 10, 10),
                  framevisible=false,
                  labelsize=24,
                  halign=:center,
                  nbanks=2)  # Use 2 rows to fit all legend items nicely
    fig[4, 1] = legend  # Place legend below all 3 panels
    
   colsize!(fig.layout, 1, Relative(1.0))

    
    # Add spacing 
    rowgap!(fig.layout, 15)
    colgap!(fig.layout, 10)
    
    
    return fig, crossentropy_health, rmse_tumor, nll_count
end
