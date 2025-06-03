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

# Neural differential equation visualization function following PkPd_latent_SDE structure
function viz_fn_nde(obs_timepoints, for_timepoints, obs_data, future_true_data, forecasted_data; sample_n=3, plot=true)
    # Unpack observation data
    u_o, covars_o, x_o, y₁_o, y₂_o, mask₁_o, mask₂_o = obs_data
    
    # Unpack future data  
    u_t, covars_t, x_t, y₁_t, y₂_t, mask₁_t, mask₂_t = future_true_data
    
    # Unpack forecasted data
    u_p = u_t
    Ex, Ey_p = forecasted_data
    Ey₁_p, Ey₂_p = softmax(Ey_p[1], dims=1), Ey_p[2]
    
    # Convert time to days for plotting
    t_o, t_p = obs_timepoints * 7, for_timepoints * 7

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
    max_t_o_valid₁ = maximum(t_o[mask₁_o[1, :, sample_n] .== 1])
    max_t_p_valid₁ = maximum(t_p[mask₁_t[1, :, sample_n] .== 1])
    
    t_o_valid₂ = t_o[mask₂_o[1, :, sample_n] .== 1]
    t_p_valid₂ = t_p[mask₂_t[1, :, sample_n] .== 1]
    max_t_o_valid₂ = maximum(t_o[mask₂_o[1, :, sample_n] .== 1])
    max_t_p_valid₂ = maximum(t_p[mask₂_t[1, :, sample_n] .== 1])

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
        if !isempty(ŷ₂_m_valid) && !isempty(y₂_t_valid)
            rmse_tumor = sqrt(sum((ŷ₂_m_valid .- y₂_t_valid).^2) / length(y₂_t_valid))
        end
        
        # Cell count (y₂_count): negative log likelihood - only on valid time points  
        nll_count = -poisson_loglikelihood(ŷ₂_m, y₂_t, mask₂_t)/length(findall(mask₂_t .== 1))
        
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

    # Set plot limits
    x_min, x_max = -2.0, max_t_p_valid₁ + max_t_p_valid₁/50
    y_max_fig₃ = maximum([maximum(ŷ₂_m_valid), maximum(x_o[1, :, sample_n]), maximum(x_t[1, :, sample_n])]) + 0.5
    y_max_fig₄ = maximum([maximum(ŷ₂_count_m_valid), maximum(y₂_o_valid), maximum(y₂_t_valid)]) + 0.5
    y_min = -0.5

    # Create the 4-panel figure
    fig = Figure(size=(1200, 900), fontsize=20)
    
    # Panel 1: Interventions
    ax1 = CairoMakie.Axis(fig[1, 1], 
                         xlabel="Time (days)", 
                         ylabel="Interventions",
                         limits=((x_min, x_max), (0.0, 1.5)), 
                         yticks=[0, 1],
                         xgridvisible=false, 
                         ygridvisible=false)
    
    # Panel 2: Health status
    ax2 = CairoMakie.Axis(fig[2, 1], 
                         xlabel="Time (days)", 
                         ylabel="Health status",
                         limits=((x_min, x_max), (-2.0, 6)),
                         xgridvisible=false, 
                         ygridvisible=false)
    
    # Panel 3: Tumor size
    ax3 = CairoMakie.Axis(fig[3, 1], 
                         xlabel="Time (days)", 
                         ylabel="Tumor size",
                         limits=((x_min, x_max), (y_min, y_max_fig₃)),
                         xgridvisible=false, 
                         ygridvisible=false)
    
    # Panel 4: Cell count
    ax4 = CairoMakie.Axis(fig[4, 1], 
                         xlabel="Time (days)", 
                         ylabel="Cell count",
                         limits=((x_min, x_max), (y_min, y_max_fig₄)),
                         xgridvisible=false, 
                         ygridvisible=false)

    # Plot interventions
    scatter!(ax1, t_o[valid_indices_chemo_o], ones(length(valid_indices_chemo_o)), 
            marker=:utriangle, markersize=10, color=:blue, label="Chemotherapy regimen")
    scatter!(ax1, t_o[valid_indices_radio_o], ones(length(valid_indices_radio_o)), 
            marker=:star5, markersize=10, color=:red, label="Radiotherapy regimen")
    scatter!(ax1, t_p[valid_indices_chemo_p], ones(length(valid_indices_chemo_p)), 
            marker=:utriangle, markersize=10, color=:blue)
    scatter!(ax1, t_p[valid_indices_radio_p], ones(length(valid_indices_radio_p)), 
            marker=:star5, markersize=10, color=:red)

    # Plot health status
    scatter!(ax2, t_o_valid₁, y₁_o_class_valid, color=:blue, label="Observed")
    scatter!(ax2, t_p_valid₁, y₁_t_class_valid, color=(:green, 0.5), markersize=15, label="True")
    scatter!(ax2, t_p_valid₁, ŷ₁_class_valid, color=(:red, 0.9), label="Predicted")
    errorbars!(ax2, t_p_valid₁, ŷ₁_class_valid, ŷ₁_conf_valid, 
              color=(PKPD_COLORS.confidence, 0.5), whiskerwidth=8, label="Prediction uncertainty")

    # Plot tumor size
    lines!(ax3, Array(1:length(vcat(x_o[1, :, sample_n], x_t[1, :, sample_n]))), 
          vcat(x_o[1, :, sample_n], x_t[1, :, sample_n]), 
          color=:blue, label="Observed (underlying tumor size)")
    lines!(ax3, t_p, ŷ₂_m[1, :, sample_n], color=:red, label="Predicted (continuous)")
    scatter!(ax3, t_p_valid₂, ŷ₂_m_valid, color=:red, label="Predicted (weekly irregular)")
    band!(ax3, t_p, ŷ₂_CI_low, ŷ₂_CI_up, 
         color=(PKPD_COLORS.confidence, 0.5), label="Prediction uncertainty")

    # Plot cell count
    scatter!(ax4, t_o_valid₂, y₂_o_valid, color=:blue, label="Observed")
    scatter!(ax4, t_p_valid₂, y₂_t_valid, color=(:green, 0.5), markersize=15, label="True")
    scatter!(ax4, t_p_valid₂, ŷ₂_count_m_valid, color=(:red, 0.9), label="Predicted")
    errorbars!(ax4, t_p_valid₂, ŷ₂_count_m_valid, ŷ₂_count_confidence_valid, 
              color=(PKPD_COLORS.confidence, 0.5), whiskerwidth=8, label="Prediction uncertainty")

    # Add background periods for all panels
    for ax in [ax1, ax2, ax3, ax4]
        poly!(ax, [-10, t_o[end], t_o[end], -10], [-10, -10, 500, 500], 
             color=(:blue, 0.05), label="observation period (history)")
        poly!(ax, [t_o[end], t_o[end] + max_t_p_valid₁, t_o[end] + max_t_p_valid₁, t_o[end]], 
             [-10, -10, 500, 500], color=(:red, 0.05), label="prediction period (future)")
    end

    # Link x-axes
    linkxaxes!(ax1, ax2, ax3, ax4)
    
    # Add legends
    fig[1, 2] = Legend(fig, ax1, framevisible=false, halign=:left)
    fig[3, 2] = Legend(fig, ax3, framevisible=false, halign=:left)
    fig[4, 2] = Legend(fig, ax4, framevisible=false, halign=:left)
    
    display(fig)
    return fig, crossentropy_health, rmse_tumor, nll_count
end

