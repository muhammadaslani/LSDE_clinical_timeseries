using Rhythm
using DifferentialEquations, Random, Distributions, CairoMakie, MLUtils, Printf, Statistics

include("examples/glucose/data/data_prep.jl")

# ── Generate dataset ──────────────────────────────────────────────────────────
U, X, Y, T, covariates = generate_dataset(; n_samples = 10, seed = 42);

# ── Plot helper ───────────────────────────────────────────────────────────────
"""
Plot states, inputs, and CGM observations for a single patient sample.
"""
function plot_sample(i::Int; U, X, Y, T, covariates, tspan = (0.0, 1440.0))
    t_full  = range(tspan[1], tspan[2], length = size(X[i], 2))  # 1-min grid
    t_obs   = T[i]

    # States: X[i] is (4 × T_full) — rows: D, G, X, I
    Gb    = covariates[1, i]
    D_gut = X[i][1, :]           # gut glucose [mg/dL]
    G_abs = X[i][2, :]          # absolute plasma glucose [mg/dL]
    Xi    = X[i][3, :]           # remote insulin [1/min]
    I_pl  = X[i][4, :]           # plasma insulin [μU/mL]

    fig = Figure(size = (1400, 900), figure_padding = 20)
    Label(fig[0, :],
          "Patient $i  |  Gb = $(round(Gb, digits=1)) mg/dL  |  Ib = $(round(covariates[2,i], digits=1)) μU/mL" *
          "  |  1st meal = $(round(covariates[3,i]/60, digits=1)) h  |  interval = $(round(covariates[4,i], digits=0)) min" *
          "  |  bolus offset = $(round(covariates[5,i], digits=0)) min  |  basal at = $(round(covariates[6,i]/60, digits=1)) h",
          fontsize = 12, font = :bold)

    # ── States ────────────────────────────────────────────────────────────────
    ax1 = Axis(fig[1, 1]; ylabel = "Glucose [mg/dL]", title = "G (plasma glucose)")
    lines!(ax1, t_full, G_abs; color = :steelblue, linewidth = 2, label = "G(t) + Gb")
    scatter!(ax1, t_obs, Float64.(Y[i]); color = (:tomato, 0.7), markersize = 5, label = "G_obs (CGM)")
    hlines!(ax1, [Gb]; color = :gray, linestyle = :dash, linewidth = 1, label = "Basal Gb")
    axislegend(ax1; position = :rt, labelsize = 11)

    ax2 = Axis(fig[1, 2]; ylabel = "Gut glucose [mg/dL]", title = "D (gut absorption compartment)")
    lines!(ax2, t_full, D_gut; color = :chocolate, linewidth = 2)

    ax3 = Axis(fig[2, 1]; ylabel = "Remote insulin [1/min]", title = "X (remote insulin)")
    lines!(ax3, t_full, Xi; color = :darkorange, linewidth = 2)

    ax4 = Axis(fig[2, 2]; ylabel = "Insulin [μU/mL]", title = "I (plasma insulin)")
    lines!(ax4, t_full, I_pl; color = :purple, linewidth = 2)

    # ── Inputs ────────────────────────────────────────────────────────────────
    t_u = range(tspan[1], tspan[2], length = size(U[i], 2))

    ax5 = Axis(fig[3, 1]; ylabel = "Meal input [mg/dL/min]", title = "Carbohydrate ingestion rate", xlabel = "Time [min]")
    stairs!(ax5, t_u, Float64.(U[i][1, :]); color = :seagreen, linewidth = 2)

    ax6 = Axis(fig[3, 2]; ylabel = "u_I [μU/mL/min]", title = "Insulin injection (u_I)", xlabel = "Time [min]")
    stairs!(ax6, t_u, Float64.(U[i][2, :]); color = :crimson, linewidth = 2)

    linkxaxes!(ax1, ax2, ax3, ax4, ax5, ax6)
    colsize!(fig.layout, 1, Relative(0.5))
    colsize!(fig.layout, 2, Relative(0.5))

    fig
end

# ── Plot a few samples ────────────────────────────────────────────────────────
n_plot = 2
for i in 1:n_plot
    fig = plot_sample(i; U, X, Y, T, covariates)
    display(fig)
end



include("examples/glucose/data/data_prep.jl")
include("examples/glucose/data/data_utils.jl")
data, train_loader, val_loader, test_loader, dims, ts_obs, ts_for, stats = generate_dataloader(n_samples=100);
