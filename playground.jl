using StatsBase
y₁_all = hcat(data[4], data[11])  # combine obs + forecast health scores
mask_all = hcat(data[6], data[13]) # combine masks
classes = onecold(y₁_all, 0:5)
# Only count valid (unmasked) entries
valid = mask_all[1, :, :] .== 1
valid_classes = classes[valid]
println("Class distribution (ECOG 0–5):")
println(countmap(valid_classes))
println("\nProportions:")
for (k, v) in sort(countmap(valid_classes))
    @printf("  Score %d: %5d  (%.1f%%)\n", k, v, 100 * v / length(valid_classes))
end



using CairoMakie

function plot_tumor_and_treatment(data, ts_obs; patients=1:3)
    # X is at daily resolution — create matching time grid
    x_obs, x_for = data[3], data[10]
    x_full = hcat(x_obs, x_for)
    t_daily = range(0, 1, length=size(x_full, 2))
    fig = Figure(size=(900, 600))
    ax1 = CairoMakie.Axis(fig[1, 1], ylabel="Tumor Volume", title="Tumor Volume Trajectories")
    for (idx, p_idx) in enumerate(patients)
        if p_idx <= size(x_full, 3)
            lines!(ax1, t_daily, x_full[1, :, p_idx], linewidth=1.5, label="Patient $p_idx")
        end
    end
    vlines!(ax1, [ts_obs[end]], color=:gray, linestyle=:dash, label="Obs/Forecast split")
    Legend(fig[1, 2], ax1, framevisible=false)

    # Treatment inputs (at observation resolution)
    u_obs, u_for = data[1], data[8]
    u_full = hcat(u_obs, u_for)
    t_obs_full = range(0, 1, length=size(u_full, 2))

    ax2 = CairoMakie.Axis(fig[2, 1], ylabel="Chemotherapy")
    for (idx, p_idx) in enumerate(patients)
        if p_idx <= size(u_full, 3)
            stairs!(ax2, t_obs_full, u_full[1, :, p_idx], linewidth=1.2, color=Cycled(idx), label="P$p_idx")
        end
    end
    vlines!(ax2, [ts_obs[end]], color=:gray, linestyle=:dash)

    ax3 = CairoMakie.Axis(fig[3, 1], xlabel="Time (normalized)", ylabel="Radiotherapy")
    for (idx, p_idx) in enumerate(patients)
        if p_idx <= size(u_full, 3)
            stairs!(ax3, t_obs_full, u_full[2, :, p_idx], linewidth=1.2, color=Cycled(idx), label="P$p_idx")
        end
    end
    vlines!(ax3, [ts_obs[end]], color=:gray, linestyle=:dash)

    linkxaxes!(ax1, ax2, ax3)
    display(fig)
    return fig
end

plot_tumor_and_treatment(data, ts_obs, patients=1:5)



# X states are in data[3] (obs) and data[10] (forecast)
x_obs, x_for = data[3], data[10];
x_full = hcat(x_obs, x_for);
n_patients = size(x_full, 3);

# Count outcomes
died = sum(x_full[5, end, :] .<= 0.12);

remission = sum(x_full[1, end, :] .<= 0.6);
println("Died: $died / $n_patients ($(round(100*died/n_patients, digits=1))%)");
println("Remission: $remission / $n_patients ($(round(100*remission/n_patients, digits=1))%)");
println("Survived with tumor: $(n_patients - died - remission) / $n_patients");

# === Dataset Heterogeneity Analysis ===
state_names = ["Tumor", "Chemo", "Radio", "Immune", "Health"];
n_times = size(x_full, 2);
timepoints_check = [1, n_times ÷ 4, n_times ÷ 2, 3 * n_times ÷ 4, n_times];

println("\n=== Heterogeneity Analysis (CV = std/mean) ===");
println("State        | t=start | t=25%   | t=50%   | t=75%   | t=end   | Overall");
println("-------------|---------|---------|---------|---------|---------|--------");
for (s, name) in enumerate(state_names)
    cvs = []
    for t_idx in timepoints_check
        vals = x_full[s, t_idx, :]
        μ = mean(vals)
        σ = std(vals)
        cv = μ ≈ 0.0 ? 0.0 : σ / abs(μ)
        push!(cvs, cv)
    end
    # Overall CV: across all timepoints and patients
    all_vals = vec(x_full[s, :, :])
    overall_cv = mean(all_vals) ≈ 0.0 ? 0.0 : std(all_vals) / abs(mean(all_vals))
    println("$(rpad(name, 13))| $(join([lpad(round(cv*100, digits=1), 6) * "%" for cv in cvs], " | ")) | $(lpad(round(overall_cv*100, digits=1), 5))%")
end

# Trajectory spread: range of final tumor values








tumor_final = x_full[1, end, :];
println("\nTumor at end: min=$(round(minimum(tumor_final), digits=1)), " *
        "median=$(round(median(tumor_final), digits=1)), " *
        "max=$(round(maximum(tumor_final), digits=1)), " *
        "IQR=$(round(quantile(tumor_final, 0.25), digits=1))-$(round(quantile(tumor_final, 0.75), digits=1))");