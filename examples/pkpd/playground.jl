using CairoMakie
include("data/data_prep.jl")

"""
    plot_pkpd(patient_idx; n_samples=50, seed=1234, tspan=(0.0, 90.0), sample_rate=3)

Plot tumor volume and cell count for a patient.
- Blue line: true tumor volume (state x)
- Orange: observed cell counts (Poisson-sampled from true volume)
"""
function plot_pkpd(patient_idx::Int;
    n_samples=50, seed=1234,
    tspan=(0.0, 180.0), sample_rate=1)

    U, X, Y₁, Y₂, T, covariates = generate_dataset(;
        n_samples=max(patient_idx, n_samples), tspan, sample_rate, seed)

    i = patient_idx
    t_obs = Float64.(T[i])
    cell_counts = Float64.(Y₂[i][1, :])
    x_states = Float64.(X[i])  # 5 × T_full
    # Use actual solution time points (saveat=1.0), not range — handles early termination correctly
    t_full = range(tspan[1], step=1.0, length=size(x_states, 2))

    tumor_true = x_states[1, :]   # true tumor volume
    immune_true = x_states[4, :]  # true immune cells
    n_obs = length(t_obs)
    inputs = Float64.(U[i])       # 2 × T_obs: [chemo; radio]
    chemo_input = inputs[1, 1:n_obs]
    radio_input = inputs[2, 1:n_obs]

    fig = Figure(size=(1000, 1000))

    # --- Tumor volume & cell counts ---
    ax1 = CairoMakie.Axis(fig[1, 1];
        ylabel="Tumor volume / Cell count",
        title="Patient $i — PKPD Dynamics")

    lines!(ax1, collect(t_full), tumor_true; color=:dodgerblue, linewidth=3, label="Tumor volume (true)")
    lines!(ax1, t_obs, cell_counts; color=:orange, linewidth=1)
    scatter!(ax1, t_obs, cell_counts; color=:orange, markersize=8, marker=:circle, label="Cell count (obs, n=$(length(t_obs)))")

    axislegend(ax1; position=:rt)

    # --- Immune response ---
    ax2 = CairoMakie.Axis(fig[2, 1]; ylabel="Immune level")

    lines!(ax2, collect(t_full), immune_true; color=:green, linewidth=3, label="Immune (true)")

    axislegend(ax2; position=:rt)

    # --- Health status ---
    health_true = x_states[5, :]
    ax3 = CairoMakie.Axis(fig[3, 1]; ylabel="Health")

    lines!(ax3, collect(t_full), health_true; color=:teal, linewidth=3, label="Health (true)")

    axislegend(ax3; position=:rt)

    # --- Chemo input ---
    ax4 = CairoMakie.Axis(fig[4, 1]; ylabel="Chemo")
    stem!(ax4, t_obs, chemo_input; color=:purple, stemcolor=:purple, stemwidth=2, markersize=6, label="Chemo")
    axislegend(ax4; position=:rt)

    # --- Radio input ---
    ax5 = CairoMakie.Axis(fig[5, 1]; xlabel="Time (days)", ylabel="Radio")
    stem!(ax5, t_obs, radio_input; color=:red, stemcolor=:red, stemwidth=2, markersize=6, label="Radio")
    axislegend(ax5; position=:rt)

    linkxaxes!(ax1, ax2, ax3, ax4, ax5)

    fig
end

# --- Try it ---
fig = plot_pkpd(1)
# fig = plot_pkpd(5)

"""
    plot_multi_patients(patient_ids; kwargs...)

Plot tumor volume and health for multiple patients in a grid.
Each column is a patient, rows are: tumor+obs, health.
"""
function plot_multi_patients(patient_ids::AbstractVector{Int};
    n_samples=50, seed=1234,
    tspan=(0.0, 365.0), sample_rate=3)

    U, X, Y₁, Y₂, T, covariates = generate_dataset(;
        n_samples=max(maximum(patient_ids), n_samples), tspan, sample_rate, seed)

    n = length(patient_ids)
    ncols = min(n, 4)
    nrows = ceil(Int, n / ncols)
    fig = Figure(size=(350 * ncols, 300 * nrows))

    for (idx, i) in enumerate(patient_ids)
        row = (idx - 1) ÷ ncols + 1
        col = (idx - 1) % ncols + 1

        x_states = Float64.(X[i])
        t_full = range(tspan[1], step=1.0, length=size(x_states, 2))
        t_obs = Float64.(T[i])
        cell_counts = Float64.(Y₂[i][1, :])
        tumor_true = x_states[1, :]
        health_true = x_states[5, :]

        ax = CairoMakie.Axis(fig[row, col]; title="Patient $i",
            xlabel=row == nrows ? "Time (days)" : "", ylabel=col == 1 ? "Tumor / Cells" : "")
        lines!(ax, collect(t_full), tumor_true; color=:dodgerblue, linewidth=2)
        scatter!(ax, t_obs, cell_counts; color=:orange, markersize=5)

        # Health on secondary y-axis via twin
        ax2 = CairoMakie.Axis(fig[row, col]; ylabel=col == ncols ? "Health" : "",
            yaxisposition=:right, yticklabelcolor=:teal)
        hidespines!(ax2)
        hidexdecorations!(ax2)
        lines!(ax2, collect(t_full), health_true; color=:teal, linewidth=2, linestyle=:dash)
    end

    fig
end

# Plot 16 patients to scan for non-curing cases
fig2 = plot_multi_patients(1:16)
