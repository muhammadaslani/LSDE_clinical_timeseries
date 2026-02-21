# Neural CDE Encoder-Decoder experiment on glucose dataset
#
# Architecture:
# - Encoder: GRU backward → h, build_control_path → dXdt, NeuralCDE → z_enc
# - Decoder: build_control_path(future controls) → dXdt, NeuralCDE(z_enc_final) → z_dec
# - Output: Dense(z_dec) → ŷ

using Rhythm
using Lux, DifferentialEquations, Random, ComponentArrays, Optimisers
using Statistics, Printf, MLUtils, Distributions, CairoMakie, Zygote

include("../data/data_prep.jl")
include("../data/data_utils.jl")
include(joinpath(@__DIR__, "../../../src/core/neural_cde_encoder_decoder.jl"))

rng = Random.MersenneTwister(42)

# -----------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------
@info "Loading glucose data..."

data, train_loader, val_loader, test_loader, dims, ts_obs, ts_for, normalization_stats =
    generate_dataloader(n_samples=512, batchsize=64, split=(0.6, 0.2),
                       obs_fraction=0.5, normalization=true, seed=42);

control_dim = dims["input_dim"]    # insulin + meal (2)
covars_dim = 6                     # static covariates

# -----------------------------------------------------------------------
# Model setup
# -----------------------------------------------------------------------
latent_dim = 32
hidden_size = 32
depth = 1
cde_dt = nothing         # Euler step size (nothing = auto from saveat)

model = CDEEncoderDecoder(;
    obs_dim=1,
    covars_dim=covars_dim,
    control_dim=control_dim,
    latent_dim=latent_dim,
    hidden_size=hidden_size,
    depth=depth
)

rng_init = Random.MersenneTwister(123)
θ, st = Lux.setup(rng_init, model);
θ = ComponentArray(θ);

# -----------------------------------------------------------------------
# Loss function
# -----------------------------------------------------------------------
function loss_fn(model, θ, st, batch)
    U_obs, Covars_obs, X_obs, Y_obs, Masks_obs,
    U_forecast, Covars_forecast, X_forecast, Y_forecast, Masks_forecast = batch

    ŷ_forecast, st_new = model(U_obs, Covars_obs, Y_obs, Masks_obs,
                                U_forecast, Y_forecast, Masks_forecast,
                                ts_obs, ts_for, θ, st; dt=cde_dt)

    loss = mean((ŷ_forecast .- Y_forecast) .^ 2)
    return loss, st_new
end

# -----------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------
lr0 = 5e-3
lr_min = 1e-5
n_epochs = 500
opt = Adam(lr0)
opt_st = Optimisers.setup(opt, θ);

@info "Training started"

for epoch in 1:n_epochs
    # Cosine annealing: lr decays from lr0 to lr_min over n_epochs
    lr = lr_min + 0.5f0 * (lr0 - lr_min) * (1 + cos(Float32(π) * epoch / n_epochs))
    Optimisers.adjust!(opt_st, lr)

    train_loss = 0.0f0

    for batch in train_loader
        loss, grads = Zygote.withgradient(θ) do θ_in
            l, _ = loss_fn(model, θ_in, st, batch)
            l
        end

        opt_st, θ = Optimisers.update(opt_st, θ, grads[1])
        train_loss += loss
    end

    if epoch % 10 == 0
        @printf("Epoch %3d | Train Loss: %.4e | LR: %.2e\n", epoch, train_loss / length(train_loader), lr)
    end
end


# -----------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------
function plot_predictions(model, θ, st, loader, ts_obs, ts_for, normalization_stats; sample_idx=1)
    batch = first(loader)
    U_obs, Covars_obs, X_obs, Y_obs, Masks_obs,
    U_forecast, Covars_forecast, X_forecast, Y_forecast, Masks_forecast = batch

    ŷ_forecast, _ = model(U_obs, Covars_obs, Y_obs, Masks_obs,
                           U_forecast, Y_forecast, Masks_forecast,
                           ts_obs, ts_for, θ, st; dt=cde_dt)

    # Rescale to original units
    Y_max = normalization_stats["Y_stats"].max_val
    t_min = normalization_stats["T_stats"].min_val
    t_max = normalization_stats["T_stats"].max_val

    ts_obs_orig = ts_obs .* (t_max - t_min) .+ t_min
    ts_for_orig = ts_for .* (t_max - t_min) .+ t_min
    Y_obs_orig = Y_obs .* Y_max
    Y_for_orig = Y_forecast .* Y_max
    ŷ_for_orig = ŷ_forecast .* Y_max

    fig = CairoMakie.Figure(size=(1200, 700))

    # Row 1: Glucose — observation and forecast
    ax1 = CairoMakie.Axis(fig[1, 1], ylabel="Glucose (mg/dL)", title="Observation Window")
    CairoMakie.lines!(ax1, ts_obs_orig, Y_obs_orig[1, :, sample_idx], label="Observed glucose", linewidth=2)
    CairoMakie.axislegend(ax1)

    ax2 = CairoMakie.Axis(fig[1, 2], ylabel="Glucose (mg/dL)", title="Forecast Window")
    CairoMakie.lines!(ax2, ts_for_orig, Y_for_orig[1, :, sample_idx], label="True glucose", linewidth=2)
    CairoMakie.lines!(ax2, ts_for_orig, ŷ_for_orig[1, :, sample_idx], label="Predicted glucose",
                       linewidth=2, linestyle=:dash)
    CairoMakie.axislegend(ax2)

    # Row 2: Meal signal
    ax3 = CairoMakie.Axis(fig[2, 1], ylabel="Meal [mg/dL/min]")
    CairoMakie.barplot!(ax3, ts_obs_orig, U_obs[1, :, sample_idx], color=:orange, label="Meal")
    CairoMakie.axislegend(ax3)

    ax4 = CairoMakie.Axis(fig[2, 2], ylabel="Meal [mg/dL/min]")
    CairoMakie.barplot!(ax4, ts_for_orig, U_forecast[1, :, sample_idx], color=:orange, label="Meal")
    CairoMakie.axislegend(ax4)

    # Row 3: Insulin signal
    ax5 = CairoMakie.Axis(fig[3, 1], xlabel="Time (min)", ylabel="Insulin [μU/mL/min]")
    CairoMakie.barplot!(ax5, ts_obs_orig, U_obs[2, :, sample_idx], color=:steelblue, label="Insulin")
    CairoMakie.axislegend(ax5)

    ax6 = CairoMakie.Axis(fig[3, 2], xlabel="Time (min)", ylabel="Insulin [μU/mL/min]")
    CairoMakie.barplot!(ax6, ts_for_orig, U_forecast[2, :, sample_idx], color=:steelblue, label="Insulin")
    CairoMakie.axislegend(ax6)

    CairoMakie.save("glucose_cde_encoder_decoder_predictions.png", fig)
    println("Plot saved to glucose_cde_encoder_decoder_predictions.png")
    return fig
end

plot_predictions(model, θ, st, test_loader, ts_obs, ts_for, normalization_stats; sample_idx=1)
