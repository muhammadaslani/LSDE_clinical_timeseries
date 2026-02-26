# Professional color palette for PKPD time series visualization
const PKPD_COLORS = (
    observed="#2E86AB",          # Deep blue for observations
    truth="#A23B72",             # Deep magenta for ground truth
    predicted="#F18F01",         # Orange for predictions
    confidence="#C73E1D",        # Red for confidence intervals
    obs_period="#B8D4F0",        # Light blue for observation period
    forecast_period="#FFD4A3",   # Light orange for forecast period
    tumor="#E63946",             # Red for tumor volume
    health="#06D6A0"             # Green for health score
)

# Neural differential equation visualization function
function viz_fn(obs_timepoints, for_timepoints, obs_data, future_true_data, forecasted_data, normalization_stats)
    # Denormalize timepoints back to actual day values
    if normalization_stats !== nothing && haskey(normalization_stats, "T_stats")
        t_min_val = normalization_stats["T_stats"].min_val
        t_max_val = normalization_stats["T_stats"].max_val
        t_o = Float32.(obs_timepoints .* (t_max_val - t_min_val) .+ t_min_val)
        t_p = Float32.(for_timepoints .* (t_max_val - t_min_val) .+ t_min_val)
    else
        t_o = Float32.(obs_timepoints)
        t_p = Float32.(for_timepoints)
    end

    # Unpack observation data
    u_o, _, x_o, y₁_o, y₂_o, mask₁_o, mask₂_o = obs_data
    # Unpack future data
    u_t, _, x_t, y₁_t, y₂_t, mask₁_t, mask₂_t = future_true_data
    # Unpack forecasted data
    Ex, Ey_p = forecasted_data
    ŷ₁_mc, ŷ₂_mc = softmax(Ey_p[1], dims=1), Ey_p[2]

    if normalization_stats !== nothing && haskey(normalization_stats, "Y₂_stats")
        y_max = normalization_stats["Y₂_stats"].max_val
        ŷ₂_mc = ŷ₂_mc .* y_max
        y₂_o = y₂_o .* y_max
        y₂_t = y₂_t .* y_max
    end

    # Convert health status to classes
    y₁_o_class = onecold(softmax(y₁_o, dims=1), Array(0:5))
    y₁_t_class = onecold(softmax(y₁_t, dims=1), Array(0:5))

    # Calculate prediction results
    ŷ₁_m = dropmean(ŷ₁_mc, dims=4)
    ŷ₁_class = onecold(ŷ₁_m, Array(0:5))
    ŷ₁_conf = 1 .- npe_per_timepoint(Ey_p[1], mask₁_t)
    ŷ₂_m = dropmean(ŷ₂_mc, dims=4)
    ŷ₂_s = dropmean(std(ŷ₂_mc, dims=4), dims=4)
    ŷ₂_count = rand.(Poisson.(clamp.(ŷ₂_mc, 0.0, 1000.0)))
    ŷ₂_count_m = dropmean(ŷ₂_count, dims=4)

    # Observation/forecast boundary: use the actual last observation timepoint
    t_boundary = maximum(t_o)

    # Build time grids for the state trajectories (x_o, x_t) which may have a
    # different (finer) time resolution than the observation timepoints (t_o, t_p).
    n_x_o = size(x_o, 2)
    n_x_t = size(x_t, 2)
    t_x_o = Float32.(range(minimum(t_o), t_boundary, length=n_x_o))
    t_x_t = Float32.(range(t_boundary, maximum(t_p), length=n_x_t))

    # Find valid (mask==1) time points for each output
    t_o_valid₁ = t_o[mask₁_o[1, :, 1] .== 1]
    t_p_valid₁ = t_p[mask₁_t[1, :, 1] .== 1]

    t_o_valid₂ = t_o[mask₂_o[1, :, 1] .== 1]
    t_p_valid₂ = t_p[mask₂_t[1, :, 1] .== 1]

    # Extract valid observed data points (filtered by mask)
    y₁_o_class_valid = Float32.(y₁_o_class[findall(i -> mask₁_o[1, i, 1] == 1, 1:length(t_o)), 1])
    y₂_o_valid = Float32.(y₂_o[1, findall(i -> mask₂_o[1, i, 1] == 1, 1:length(t_o)), 1])

    # Extract valid forecast ground truth (filtered by mask)
    y₁_t_class_valid = Float32.(y₁_t_class[findall(i -> mask₁_t[1, i, 1] == 1, 1:length(t_p)), 1])
    y₂_t_valid = Float32.(y₂_t[1, findall(i -> mask₂_t[1, i, 1] == 1, 1:length(t_p)), 1])

    # Extract valid forecast predictions (filtered by mask)
    ŷ₁_class_valid = Float32.(ŷ₁_class[findall(i -> mask₁_t[1, i, 1] == 1, 1:length(t_p)), 1])
    ŷ₁_conf_valid = Float32.(ŷ₁_conf[findall(i -> mask₁_t[1, i, 1] == 1, 1:length(t_p)), 1])
    ŷ₂_m_valid = Float32.(ŷ₂_m[1, findall(i -> mask₂_t[1, i, 1] == 1, 1:length(t_p)), 1])
    ŷ₂_s_valid = Float32.(ŷ₂_s[1, findall(i -> mask₂_t[1, i, 1] == 1, 1:length(t_p)), 1])
    ŷ₂_count_m_valid = Float32.(ŷ₂_count_m[1, findall(i -> mask₂_t[1, i, 1] == 1, 1:length(t_p)), 1])

    # Confidence intervals for tumor predictions (over all forecast timepoints)
    ŷ₂_CI_low = ŷ₂_m[1, :, 1] .- 1.96 * ŷ₂_s[1, :, 1]
    ŷ₂_CI_up = ŷ₂_m[1, :, 1] .+ 1.96 * ŷ₂_s[1, :, 1]

    # Find treatment indices across the full time range
    valid_indices_chemo_o = findall(i -> u_o[1, i, 1] > 0.5, 1:length(t_o))
    valid_indices_radio_o = findall(i -> u_o[2, i, 1] > 0.5, 1:length(t_o))
    valid_indices_chemo_p = findall(i -> u_t[1, i, 1] > 0.5, 1:length(t_p))
    valid_indices_radio_p = findall(i -> u_t[2, i, 1] > 0.5, 1:length(t_p))

    # Calculate plot limits: x_max is the last valid timepoint across all outputs
    x_min = 0.0
    all_max_times = [
        isempty(t_o_valid₁) ? 0.0f0 : maximum(t_o_valid₁),
        isempty(t_p_valid₁) ? 0.0f0 : maximum(t_p_valid₁),
        isempty(t_o_valid₂) ? 0.0f0 : maximum(t_o_valid₂),
        isempty(t_p_valid₂) ? 0.0f0 : maximum(t_p_valid₂)
    ]
    t_last_valid = maximum(all_max_times) > 0 ? maximum(all_max_times) : maximum(t_p)
    x_max = t_last_valid + 0.05 * t_last_valid

    # Truncate state trajectories to only show up to x_max
    x_o_mask = t_x_o .<= x_max
    x_t_mask = t_x_t .<= x_max
    t_x_o_plot = t_x_o[x_o_mask]
    t_x_t_plot = t_x_t[x_t_mask]
    x_o_plot = x_o[1, x_o_mask, 1]
    x_t_plot = x_t[1, x_t_mask, 1]

    # Panel 1: Health status y-limits
    all_health_values = vcat(y₁_o_class_valid, y₁_t_class_valid, ŷ₁_class_valid)
    if !isempty(all_health_values)
        health_range = maximum(all_health_values) - minimum(all_health_values)
        health_padding = max(0.25 * health_range, 0.5)
        y1_min = minimum(all_health_values) - health_padding
        y1_max = maximum(all_health_values) + health_padding
    else
        y1_min, y1_max = -0.5, 3.5
    end

    # Panel 2: Tumor size y-limits (using truncated data)
    all_tumor_values = vcat(x_o_plot, x_t_plot, ŷ₂_m[1, :, 1], ŷ₂_CI_low, ŷ₂_CI_up)
    if !isempty(all_tumor_values)
        tumor_range = maximum(all_tumor_values) - minimum(all_tumor_values)
        tumor_padding = max(0.25 * tumor_range, 0.1 * abs(maximum(all_tumor_values)))
        y2_min = minimum(all_tumor_values) - tumor_padding
        y2_max = maximum(all_tumor_values) + tumor_padding
    else
        y2_min, y2_max = -0.5, 5.0
    end

    # Panel 3: Cell count y-limits
    all_count_values = vcat(y₂_o_valid, y₂_t_valid, ŷ₂_count_m_valid)
    if !isempty(all_count_values)
        count_range = maximum(all_count_values) - minimum(all_count_values)
        count_padding = max(0.25 * count_range, 0.1 * abs(maximum(all_count_values)))
        y3_min = minimum(all_count_values) - count_padding
        y3_max = maximum(all_count_values) + count_padding
    else
        y3_min, y3_max = -5.0, 50.0
    end

    # Create the 3-panel figure
    fig = Figure(size=(1200, 800), fontsize=20,
        backgroundcolor=:white,
        figure_padding=10)

    # Panel 1: Health status
    ax1 = CairoMakie.Axis(fig[1, 1],
        xlabel="",
        ylabel="Performance score",
        xgridvisible=true, ygridvisible=true,
        xgridcolor=("#E5E5E5", 0.8), ygridcolor=("#E5E5E5", 0.8),
        topspinevisible=false, rightspinevisible=false,
        xticklabelsize=16, yticklabelsize=16,
        xlabelsize=20, ylabelsize=20)

    # Panel 2: Tumor size
    ax2 = CairoMakie.Axis(fig[2, 1],
        xlabel="",
        ylabel="Tumor Size (Unobserved)",
        xgridvisible=true, ygridvisible=true,
        xgridcolor=("#E5E5E5", 0.8), ygridcolor=("#E5E5E5", 0.8),
        topspinevisible=false, rightspinevisible=false,
        xticklabelsize=16, yticklabelsize=16,
        xlabelsize=20, ylabelsize=20)

    # Panel 3: Cell count
    ax3 = CairoMakie.Axis(fig[3, 1],
        xlabel="Time (days)",
        ylabel="Cell Count",
        xgridvisible=true, ygridvisible=true,
        xgridcolor=("#E5E5E5", 0.8), ygridcolor=("#E5E5E5", 0.8),
        topspinevisible=false, rightspinevisible=false,
        xticklabelsize=16, yticklabelsize=16,
        xlabelsize=20, ylabelsize=20)

    # Add background periods and intervention lines for all panels
    y_limits = [(y1_min, y1_max), (y2_min, y2_max), (y3_min, y3_max)]

    for (i, (ax, (y_bg_min, y_bg_max))) in enumerate(zip([ax1, ax2, ax3], y_limits))
        # Observation period background
        if i == 1
            poly!(ax, [x_min, t_boundary, t_boundary, x_min],
                [y_bg_min, y_bg_min, y_bg_max, y_bg_max],
                color=(PKPD_COLORS.obs_period, 0.3),
                label="Observation Period")
            poly!(ax, [t_boundary, x_max, x_max, t_boundary],
                [y_bg_min, y_bg_min, y_bg_max, y_bg_max],
                color=(PKPD_COLORS.forecast_period, 0.3),
                label="Forecasting Period")
        else
            poly!(ax, [x_min, t_boundary, t_boundary, x_min],
                [y_bg_min, y_bg_min, y_bg_max, y_bg_max],
                color=(PKPD_COLORS.obs_period, 0.3))
            poly!(ax, [t_boundary, x_max, x_max, t_boundary],
                [y_bg_min, y_bg_min, y_bg_max, y_bg_max],
                color=(PKPD_COLORS.forecast_period, 0.3))
        end

        vlines!(ax, [t_boundary], color=("#666666", 0.8), linewidth=5, linestyle=:dash)

        # Add intervention lines across all panels
        if i == 1
            vlines!(ax, t_o[valid_indices_chemo_o], color=:navy, linewidth=4.5, alpha=0.8,
                label="Chemotherapy Sessions")
            vlines!(ax, t_p[valid_indices_chemo_p], color=:navy, linewidth=4.5, alpha=0.8)
            vlines!(ax, t_o[valid_indices_radio_o], color=:darkred, linewidth=4.5, alpha=0.8,
                label="Radiotherapy Sessions")
            vlines!(ax, t_p[valid_indices_radio_p], color=:darkred, linewidth=4.5, alpha=0.8)
        else
            vlines!(ax, t_o[valid_indices_chemo_o], color=:navy, linewidth=4.5, alpha=0.8)
            vlines!(ax, t_p[valid_indices_chemo_p], color=:navy, linewidth=4.5, alpha=0.8)
            vlines!(ax, t_o[valid_indices_radio_o], color=:darkred, linewidth=4.5, alpha=0.8)
            vlines!(ax, t_p[valid_indices_radio_p], color=:darkred, linewidth=4.5, alpha=0.8)
        end
    end

    # --- Panel 1: Health status ---
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

    # --- Panel 2: Tumor size (latent/unobserved) ---
    # Plot observation-period tumor trajectory using truncated state time grid
    lines!(ax2, t_x_o_plot, x_o_plot,
        color=(PKPD_COLORS.observed, 0.8), linewidth=3.5, linestyle=:dash,
        label="Historical Observations")

    # Plot ground truth tumor trajectory in forecast period
    lines!(ax2, t_x_t_plot, x_t_plot,
        color=(PKPD_COLORS.truth, 0.8), linewidth=3.5, linestyle=:dash,
        label="Ground Truth")

    # Plot model prediction confidence band
    band!(ax2, t_p, ŷ₂_CI_low, ŷ₂_CI_up,
        color=(PKPD_COLORS.confidence, 0.25))

    # Plot model prediction mean
    lines!(ax2, t_p, ŷ₂_m[1, :, 1],
        color=PKPD_COLORS.predicted, linewidth=3.5, linestyle=:dash,
        label="Model Predictions")

    # --- Panel 3: Cell count ---
    if !isempty(t_o_valid₂) && !isempty(y₂_o_valid)
        scatter!(ax3, t_o_valid₂, y₂_o_valid,
            color=PKPD_COLORS.observed, markersize=20)
        lines!(ax3, t_o_valid₂, y₂_o_valid,
            color=(PKPD_COLORS.observed, 0.8), linewidth=3.5, linestyle=:dash)
    end

    if !isempty(t_p_valid₂) && !isempty(y₂_t_valid)
        scatter!(ax3, t_p_valid₂, y₂_t_valid,
            color=PKPD_COLORS.truth, markersize=20)
        lines!(ax3, t_p_valid₂, y₂_t_valid,
            color=(PKPD_COLORS.truth, 0.8), linewidth=3.5, linestyle=:dash)
    end

    if !isempty(t_p_valid₂) && !isempty(ŷ₂_count_m_valid)
        scatter!(ax3, t_p_valid₂, ŷ₂_count_m_valid,
            color=PKPD_COLORS.predicted, markersize=20)
        lines!(ax3, t_p_valid₂, ŷ₂_count_m_valid,
            color=PKPD_COLORS.predicted, linewidth=3.5, linestyle=:dash)
    end

    # Set axis limits
    xlims!(ax1, x_min, x_max)
    xlims!(ax2, x_min, x_max)
    xlims!(ax3, x_min, x_max)
    ylims!(ax1, y1_min, y1_max)
    ylims!(ax2, y2_min, y2_max)
    ylims!(ax3, y3_min, y3_max)

    # Create consolidated legend at the bottom
    custom_elements = [
        PolyElement(color=(PKPD_COLORS.obs_period, 0.3)),
        PolyElement(color=(PKPD_COLORS.forecast_period, 0.3)),
        MarkerElement(color=PKPD_COLORS.observed, marker=:circle, markersize=20),
        MarkerElement(color=PKPD_COLORS.truth, marker=:circle, markersize=20),
        MarkerElement(color=PKPD_COLORS.predicted, marker=:circle, markersize=20),
        LineElement(color=(PKPD_COLORS.confidence, 0.4), linewidth=16, linestyle=:solid),
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
        nbanks=2)
    fig[4, 1] = legend

    colsize!(fig.layout, 1, Relative(1.0))

    # Add spacing
    rowgap!(fig.layout, 15)
    colgap!(fig.layout, 10)

    return fig
end
