const GLUCOSE_COLORS = (
    observed="#2E86AB",
    truth="#A23B72",
    predicted="#F18F01",
    confidence="#C73E1D",
    obs_period="#B8D4F0",
    forecast_period="#FFD4A3",
)

function viz_fn(obs_timepoints, for_timepoints, obs_data, future_true_data, forecasted_data, normalization_stats)
    # Convert normalized [0,1] timepoints back to original scale (minutes)
    t_min = normalization_stats["T_stats"].min_val
    t_max = normalization_stats["T_stats"].max_val
    t_o = obs_timepoints .* (t_max - t_min) .+ t_min
    t_p = for_timepoints .* (t_max - t_min) .+ t_min

    # Unpack data
    u_o, _, x_o, y_o, mask_o = obs_data
    u_t, _, x_t, y_t, mask_t = future_true_data
    Ex, Ey_p = forecasted_data
    ŷ_mc = Ey_p
    μ_mc, log_σ²_mc = ŷ_mc

    # Denormalize if needed
    if normalization_stats !== nothing && haskey(normalization_stats, "Y_stats")
        y_max = normalization_stats["Y_stats"].max_val
        μ_mc = μ_mc .* y_max
        y_o = y_o .* y_max
        y_t = y_t .* y_max
    end

    # Mean and std across MC samples
    μ_m = dropmean(μ_mc, dims=4)
    μ_s = dropmean(std(μ_mc, dims=4), dims=4)

    # Valid time points
    t_o_valid = t_o[mask_o[1, :, 1].==1]
    t_p_valid = t_p[mask_t[1, :, 1].==1]
    max_t_o = isempty(t_o_valid) ? 0.0 : maximum(t_o_valid)
    max_t_p = isempty(t_p_valid) ? 0.0 : maximum(t_p_valid)

    y_o_valid = y_o[1, findall(i -> t_o[i] <= max_t_o && mask_o[1, i, 1] == 1, 1:length(t_o)), 1]
    y_t_valid = y_t[1, findall(i -> t_p[i] <= max_t_p && mask_t[1, i, 1] == 1, 1:length(t_p)), 1]
    μ_m_valid = μ_m[1, findall(i -> t_p[i] <= max_t_p && mask_t[1, i, 1] == 1, 1:length(t_p)), 1]
    μ_s_valid = μ_s[1, findall(i -> t_p[i] <= max_t_p && mask_t[1, i, 1] == 1, 1:length(t_p)), 1]

    # Confidence intervals
    CI_low = μ_m[1, :, 1] .- 1.96 * μ_s[1, :, 1]
    CI_up = μ_m[1, :, 1] .+ 1.96 * μ_s[1, :, 1]

    # Plot limits
    x_max = max_t_p > 0 ? max_t_p * 1.05 : 100.0
    all_y = vcat(y_o_valid, y_t_valid, μ_m_valid)
    y_range = isempty(all_y) ? 50.0 : maximum(all_y) - minimum(all_y)
    y_pad = max(0.15 * y_range, 5.0)
    y_min = isempty(all_y) ? 60.0 : minimum(all_y) - y_pad
    y_max_val = isempty(all_y) ? 200.0 : maximum(all_y) + y_pad

    fig = Figure(size=(1200, 800), fontsize=16, backgroundcolor=:white, figure_padding=10)

    # Row 1: Glucose
    ax1 = CairoMakie.Axis(fig[1, 1],
        ylabel="Glucose [mg/dL]",
        xgridvisible=true, ygridvisible=true,
        xgridcolor=("#E5E5E5", 0.8), ygridcolor=("#E5E5E5", 0.8),
        topspinevisible=false, rightspinevisible=false)

    # Background periods
    poly!(ax1, [0, max_t_o, max_t_o, 0], [y_min, y_min, y_max_val, y_max_val],
        color=(GLUCOSE_COLORS.obs_period, 0.3), label="Observation Period")
    poly!(ax1, [max_t_o, x_max, x_max, max_t_o], [y_min, y_min, y_max_val, y_max_val],
        color=(GLUCOSE_COLORS.forecast_period, 0.3), label="Forecast Period")
    vlines!(ax1, [max_t_o], color=("#666666", 0.8), linewidth=3, linestyle=:dash)

    # Confidence band
    band!(ax1, t_p, CI_low, CI_up, color=(GLUCOSE_COLORS.confidence, 0.2))

    # Observations
    if !isempty(t_o_valid)
        scatter!(ax1, t_o_valid, y_o_valid, color=GLUCOSE_COLORS.observed, markersize=12, label="Observed")
        lines!(ax1, t_o_valid, y_o_valid, color=(GLUCOSE_COLORS.observed, 0.6), linewidth=2, linestyle=:dash)
    end

    # Ground truth
    if !isempty(t_p_valid)
        scatter!(ax1, t_p_valid, y_t_valid, color=GLUCOSE_COLORS.truth, markersize=12, label="Ground Truth")
        lines!(ax1, t_p_valid, y_t_valid, color=(GLUCOSE_COLORS.truth, 0.6), linewidth=2, linestyle=:dash)
    end

    # Predictions
    lines!(ax1, t_p, μ_m[1, :, 1], color=GLUCOSE_COLORS.predicted, linewidth=3, label="Predicted (mean)")
    if !isempty(t_p_valid)
        scatter!(ax1, t_p_valid, μ_m_valid, color="#3CB371", markersize=10, label="Predicted (points)")
    end
    xlims!(ax1, 0, x_max)
    ylims!(ax1, y_min, y_max_val)

    # Add legend outside the plot
    Legend(fig[1, 2], ax1, tellheight=false, tellwidth=true, labelsize=12)

    # Row 2: Meals
    ax2 = CairoMakie.Axis(fig[2, 1],
        ylabel="Meal [mg/dL/min]",
        xgridvisible=true, ygridvisible=true,
        xgridcolor=("#E5E5E5", 0.8), ygridcolor=("#E5E5E5", 0.8),
        topspinevisible=false, rightspinevisible=false)
    vlines!(ax2, [max_t_o], color=("#666666", 0.8), linewidth=3, linestyle=:dash)
    barplot!(ax2, t_o, u_o[1, :, 1], color=:orange)
    barplot!(ax2, t_p, u_t[1, :, 1], color=:orange)
    xlims!(ax2, 0, x_max)

    # Row 3: Insulin
    ax3 = CairoMakie.Axis(fig[3, 1],
        xlabel="Time [min]", ylabel="Insulin [μU/mL/min]",
        xgridvisible=true, ygridvisible=true,
        xgridcolor=("#E5E5E5", 0.8), ygridcolor=("#E5E5E5", 0.8),
        topspinevisible=false, rightspinevisible=false)
    vlines!(ax3, [max_t_o], color=("#666666", 0.8), linewidth=3, linestyle=:dash)
    barplot!(ax3, t_o, u_o[2, :, 1], color=:steelblue)
    barplot!(ax3, t_p, u_t[2, :, 1], color=:steelblue)
    xlims!(ax3, 0, x_max)

    return fig
end
