# Neural CDE experiment on the glucose dataset.
# Framing: given the full glucose trajectory (obs + covariates) for a patient,
# predict the mean glucose level (a scalar regression task), following the
# original Neural CDE paper (classification/regression on a full sequence).
# Input path X(t): vcat(covariates, glucose_obs) → (7, T, B)
# Output: mean glucose over the forecast window → (1, B)

using Rhythm
using Lux, DifferentialEquations, Random, ComponentArrays, Optimisers
using Statistics, Printf, Distributions, MLUtils, CairoMakie

include("../data/data_prep.jl")
include("../data/data_utils.jl")
include(joinpath(@__DIR__, "../../../src/core/neural_cde.jl"))

rng = Random.MersenneTwister(42)

# -----------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------
data, train_loader, val_loader, test_loader, dims, ts_obs, ts_for, norm_stats =
    generate_dataloader(; n_samples=128, batchsize=16, split=(0.6, 0.2),
                          obs_fraction=0.5, normalization=true, seed=123);

# Full timepoints [0,1] covering the obs window
ts = Float32.(ts_obs);   # (T_obs,)  — CDE sees the obs window

obs_dim   = dims["obs_dim"]    # 7 (covariates + glucose)
output_dim = 1                  # predict scalar: mean future glucose

#printf("check1")
# -----------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------
model = NeuralCDE(; obs_dim=obs_dim, latent_dim=32, output_dim=output_dim,
                    hidden_size=128, depth=2)
θ, st = Lux.setup(rng, model);
θ = ComponentArray{Float32}(θ);
#printf("check2")
# -----------------------------------------------------------------------
# Loss: MSE between predicted mean glucose and true mean future glucose
# -----------------------------------------------------------------------
function loss_fn(model, θ, st, batch)
    u_obs, covars_obs, _, y_obs, mask_obs,
    _, _, _, y_forecast, mask_forecast = batch

    Y = vcat(covars_obs, y_obs)          # (7, T_obs, B)
    ŷ, st_new = model(Y, ts, θ, st)      # (1, B)

    # Target: mean observed glucose in the forecast window per sample
    n_valid = max.(sum(mask_forecast[1, :, :], dims=1), 1)           # (1, B)
    y_target = sum(y_forecast[1, :, :] .* mask_forecast[1, :, :], dims=1) ./ n_valid  # (1, B)

    loss = mean((ŷ[1, :] .- y_target[1, :]).^2)
    return loss, st_new
end

# -----------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------
opt   = Adam(1e-3)
opt_st = Optimisers.setup(opt, θ)


n_epochs = 100
for epoch in 1:n_epochs
    losses = Float32[]
    for batch in train_loader
        loss_val, loss_back = Zygote.pullback(θ) do θ_in
            l, _ = loss_fn(model, θ_in, st, batch)
            l
        end
        grads = loss_back(1f0)[1]
        global opt_st, θ
        opt_st, θ = Optimisers.update(opt_st, θ, grads)
        push!(losses, loss_val)
    end

    if epoch % max(1, n_epochs ÷ 5) == 0 || epoch == 1
        val_losses = Float32[]
        for batch in val_loader
            loss, _ = loss_fn(model, θ, st, batch)
            push!(val_losses, loss)
        end
        @printf("Epoch %3d | train MSE: %.4f | val MSE: %.4f\n",
                epoch, mean(losses), mean(val_losses))
    end
end

println("Training complete.")
# -----------------------------------------------------------------------
# Test evaluation
# -----------------------------------------------------------------------
test_losses = Float32[]
for batch in test_loader
    loss, _ = loss_fn(model, θ, st, batch)
    push!(test_losses, loss)
end
@printf("\nTest MSE: %.4f | Test RMSE: %.4f\n",
        mean(test_losses), sqrt(mean(test_losses)))

# -----------------------------------------------------------------------
# Plotting function
# -----------------------------------------------------------------------
using CairoMakie

function plot_predictions(model, θ, st, test_loader)
    test_batch = first(test_loader)

    _, covars_obs, _, y_obs, _,
    _, _, _, y_forecast, mask_forecast = test_batch

    Y = vcat(covars_obs, y_obs)
    ŷ_pred, _ = model(Y, ts, θ, st)

    # Get true mean glucose
    n_valid = max.(sum(mask_forecast[1, :, :], dims=1), 1)
    y_true = sum(y_forecast[1, :, :] .* mask_forecast[1, :, :], dims=1) ./ n_valid

    n_samples = size(ŷ_pred, 2)

    # Create plot
    fig = CairoMakie.Figure(size=(800, 400))
    ax = CairoMakie.Axis(fig[1, 1], xlabel="Sample", ylabel="Mean Glucose (normalized)")

    x_pos = 1:n_samples
    CairoMakie.scatter!(ax, x_pos, y_true[1, 1:n_samples], label="True", markersize=8, color=:blue)
    CairoMakie.scatter!(ax, x_pos, ŷ_pred[1, 1:n_samples], label="Predicted", markersize=8, color=:red)

    CairoMakie.axislegend(ax)
    return fig
end

# Display predictions
fig = plot_predictions(model, θ, st, test_loader)
display(fig)
