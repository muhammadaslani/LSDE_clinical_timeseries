using CairoMakie
include("data/data_prep.jl")

"""
    plot_glucose(patient_idx; n_samples=50, seed=1234, kwargs...)

Plot glucose trajectory for a patient, optionally overriding model parameters.

# Keyword arguments (parameter overrides)
- `kabs`, `p1`, `p2`, `p3`, `n`, `γ`, `Vi`, `Gb`, `Ib`: override any ModelParameters field
- `σ_obs`: observation noise (default 2.0)
- `tspan`: simulation time span (default (0.0, 720.0))
- `sample_rate`: observation interval in minutes (default 5)

# Example
```julia
fig = plot_glucose(1)
fig = plot_glucose(1; p1=0.04, Gb=95.0)  # investigate parameter effects
```
"""
function plot_glucose(patient_idx::Int;
    n_samples=50, seed=1234,
    tspan=(0.0, 1440.0), sample_rate=10, n_meals=3,
    obs_rate=0.5, irregularity_seed=42)

    U, X, Y, T, covariates = generate_dataset(;
        n_samples=max(patient_idx, n_samples), tspan, sample_rate, n_meals, seed)

    i = patient_idx
    t_obs = Float64.(T[i])
    g_obs = Float64.(Y[i])
    x_states = Float64.(X[i])  # 4 × T_full
    t_full = range(tspan[1], tspan[2], length=size(x_states, 2))

    # Simulate irregularity mask (same logic as data_utils.jl)
    irr_rng = Random.MersenneTwister(irregularity_seed + i)
    mask = [rand(irr_rng) < obs_rate for _ in eachindex(t_obs)]

    fig = Figure(size=(900, 400))
    ax = CairoMakie.Axis(fig[1, 1];
        xlabel="Time (min)", ylabel="Glucose (mg/dL)",
        title="Patient $i — Glucose Trajectory")

    n_regular = length(t_obs)
    n_irreg = sum(mask)

    # True glucose (state 2) at full resolution
    lines!(ax, collect(t_full), x_states[2, :]; color=:dodgerblue, linewidth=3, label="G(t) true")

    # Regular measurement grid
    lines!(ax, t_obs, g_obs; color=:orange, linewidth=1)
    scatter!(ax, t_obs, g_obs; color=:orange, markersize=6, marker=:circle, label="Regular grid (n=$n_regular)")

    # Irregular observations (kept after masking)
    t_irr = t_obs[mask]
    g_irr = g_obs[mask]
    lines!(ax, t_irr, g_irr; color=:crimson, linewidth=2)
    scatter!(ax, t_irr, g_irr; color=:crimson, markersize=12, marker=:diamond, label="Irregular obs (n=$n_irreg)")

    # Basal glucose reference
    gb_val = covariates[1, i]
    hlines!(ax, [gb_val]; color=:gray, linestyle=:dash, linewidth=2, label="Gb=$(round(gb_val; digits=1))")

    axislegend(ax; position=:rt)
    fig
end

# --- Try it ---
fig = plot_glucose(1)
# fig = plot_glucose(1; p1=0.05)
# fig = plot_glucose(3; Gb=95.0, kabs=0.05)
