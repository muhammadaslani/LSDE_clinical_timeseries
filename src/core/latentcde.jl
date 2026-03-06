"""
    LatentCDE(obs_encoder, ctrl_encoder, init_map, dynamics, state_map, obs_decoder, ctrl_decoder)

Constructs a Latent Controlled Differential Equation model.

The CDE encoder processes history (observations + covariates + controls) through a GRU backward
pass and an encoder CDE to produce probabilistic initial conditions. The CDE decoder evolves
the latent state forward using future controls via a decoder CDE.

Arguments:

  - `obs_encoder`: CDE_Encoder that encodes history into probabilistic initial conditions.
  - `ctrl_encoder`: NoOpLayer (controls handled via CDE path construction).
  - `init_map`: NoOpLayer (sampling handled in forward pass).
  - `dynamics`: CDE dynamics component.
  - `state_map`: NoOpLayer.
  - `obs_decoder`: Decoder (e.g. MultiHeadLinearDecoder) that maps latent states to observations.
  - `ctrl_decoder`: NoOpLayer.
  - `device`: Device on which the model runs (cpu or gpu).
"""
@with_kw struct LatentCDE <: LatentVariableModel
  obs_encoder = Identity_Encoder()
  ctrl_encoder = NoOpLayer()
  init_map = NoOpLayer()
  dynamics
  state_map = NoOpLayer()
  obs_decoder = Identity_Decoder()
  ctrl_decoder = NoOpLayer()
  device = cpu_device()
end


"""
    (model::LatentCDE)(y, u, ts, ps, st)

Forward pass of the LatentCDE model.

Arguments:

  - `y`: Encoder input — concatenated history `(covars + obs + controls, T_obs, B)`.
  - `u`: Future control inputs `(control_dim, T_for, B)`.
  - `ts`: Tuple `(ts_obs, ts_for)` — observation and forecast time vectors.
  - `ps`: Parameters.
  - `st`: NamedTuple of states.

Returns:

  - `ŷ`: Decoded observations from the latent trajectory.
  - `px₀`: Probabilistic initial condition `(μ, σ)`.
  - `kl_path`: Always `nothing` (CDE has no path-wise KL).
"""
function (model::LatentCDE)(y::AbstractArray, u::AbstractArray, ts::Tuple, ps::ComponentArray, st::NamedTuple)
  ts_obs, ts_for = ts

  # 1. Encode history → probabilistic initial conditions
  px₀, _ = model.obs_encoder(y, ts_obs, ps.obs_encoder, st.obs_encoder)[1]
  # 2. Sample initial latent state
  x₀ = sample_rp(px₀)
  # 3. Reconstruct: evolve latent state over observation window
  z_dec, st_dyn = model.dynamics(x₀, u, ts_for, ps.dynamics, st.dynamics)
  # z_dec: (latent_dim, T_for, B) — already in correct shape
  # 4. Map latent states to observations
  ŷ = model.obs_decoder(z_dec, ps.obs_decoder, st.obs_decoder)[1]

  return ŷ, px₀, nothing
end


"""
    predict(model::LatentCDE, y, u, ts, ps, st, n_samples, dev)

Generate Monte Carlo forecasts from the LatentCDE model.

Arguments:

  - `model`: The LatentCDE model.
  - `y`: Encoder input — concatenated history `(covars + obs + controls, T_obs, B)`.
  - `u`: Future control inputs `(control_dim, T_for, B)`.
  - `ts`: Tuple `(ts_obs, ts_for)` — observation and forecast time vectors.
  - `ps`: Parameters.
  - `st`: NamedTuple of states.
  - `n_samples`: Number of MC samples.
  - `dev`: Device (CPU or GPU).

Returns:

  - `x_pred`: Predicted latent states `(latent_dim, T_for, B, n_samples)`.
  - `y_pred`: Predicted observations `(obs_dim, T_for, B, n_samples)`.
"""
function predict(model::LatentCDE, solver, y::AbstractArray, u::AbstractArray, ts::Tuple,
  ps::ComponentArray, st::NamedTuple, n_samples::Int, dev::Any; kwargs...)
  ts_obs, ts_for = ts

  # Encode history → probabilistic initial conditions
  px₀, _ = model.obs_encoder(y, ts_obs, ps.obs_encoder, st.obs_encoder)[1]

  # Sample multiple trajectories
  x_pred = sample_dynamics(model.dynamics, px₀, u, ts_for, ps.dynamics, st.dynamics, n_samples) |> model.device

  # Decode latent trajectories to observations
  x_pred_ = model.state_map(x_pred, ps.state_map, st.state_map)[1]
  y_pred = model.obs_decoder(x_pred_, ps.obs_decoder, st.obs_decoder)[1]

  return x_pred, y_pred
end
