"""
    LatentODE(obs_encoder, ctrl_encoder, dynamics, ofunction forward!(model::LatentODE, y::AbstractArray, u::Union{Nothing, AbstractArray}, ts::AbstractArray, ps::ComponentArray, st::NamedTuple, dynamics::ODE)
    x̂₀, _ = model.obs_encoder(y, ts_obs, ps.obs_encoder, st.obs_encoder)[1] 
    x₀ = model.init_map(sample_rp(x̂₀), ps.init_map, st.init_map)[1]
    u_enc = model.ctrl_encoder(u, ps.ctrl_encoder, st.ctrl_encoder)[1]
    x_sol = dynamics(x₀, u_enc, ts, ps.dynamics, st.dynamics)[1] |> model.device
    x = permutedims(x_sol, (1, 3, 2))
    x = model.state_map(x, ps.state_map, st.state_map)[1]
    kl_path = nothing
    ŷ = model.obs_decoder(x, ps.obs_decoder, st.obs_decoder)[1]
    û = model.ctrl_decoder(x, ps.ctrl_decoder, st.ctrl_decoder)[1]
    return ŷ, û, x̂₀, kl_path 
end ctrl_decoder)

Constructs a Latent Ordinary Differential Equation model.

Arguments:

  - `obs_encoder`: A function that encodes the observations `y` to get the initial hidden state `x₀` and context for the dynamics if needed (Partial observability) 
  - `ctrl_encoder`: A function that encodes (high-dimensional) inputs/controls to a lower-dimensional representation if needed.
  - `dynamics`: A function that models the dynamics of the system (your ODE/SDE).
  - `obs_decoder`: A function that decodes the hidden states `x` to the observations `y`.
  - 'ctrl_decoder': A function that decodes the control representation to the original control space if needed.
  - 'device': The device on which the model is stored. Default is `cpu`. 

"""
@with_kw struct LatentODE <: LatentVariableModel
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
    (model::LatentODE)(y::AbstractArray, u::Union{Nothing, AbstractArray}, ts::AbstractArray, ps::ComponentArray, st::NamedTuple)

The forward pass of the LatentODE model.

Arguments:

  - `y`: Observations
  - `u`: Control inputs
  - `ts`: Time points
  - `ps`: Parameters
  - `st`: NamedTuple of states 

Returns:

  - `ŷ`: Decoded observations from the hidden states.
  - `px₀`: Encoded initial hidden state.
  - `kl_path`: KL divergence path. (Only for SDE dynamics, otherwise `nothing`)
"""
function (model::LatentODE)(y::AbstractArray, u::Union{Nothing,AbstractArray}, ts::Tuple, ps::ComponentArray, st::NamedTuple)
  ts_obs, _ = ts
  px₀, _ = model.obs_encoder(y, ts_obs, ps.obs_encoder, st.obs_encoder)[1]
  x₀ = model.init_map(sample_rp(px₀), ps.init_map, st.init_map)[1]
  u_enc = model.ctrl_encoder(u, ps.ctrl_encoder, st.ctrl_encoder)[1]
  x_sol = model.dynamics(x₀, u_enc, ts_obs, ps.dynamics, st.dynamics)[1]
  x = permutedims(Array(x_sol), (1, 3, 2))
  kl_path = nothing
  ŷ = model.obs_decoder(x, ps.obs_decoder, st.obs_decoder)[1]
  return ŷ, px₀, kl_path
end




"""
    predict(model::LatentODE, solver::DiffEqBase.DEAlgorithm, y::AbstractArray, t_obs::AbstractArray, t_pred::AbstractArray, u::Union{Nothing, AbstractArray}, ps::ComponentArray, st::NamedTuple, n_samples::Int, dev::Device)

  Predicts the future trajectory of the system from time `T` to `T+k` given observations from time `1` to `T` and control inputs from time `1` to `T+k`.
  Used for forecasting and control applications.

      ```math
      p(y_{T:T+k \\mid y_{1:T}, u_{1:T+k}}) \\quad p(x_{T:T+k \\mid y_{1:T}, u_{1:T+k}})
      ```

Arguments:

  - `model`: The `LatentODE` model to sample from.
  - `solver`: The nummerical solver to solve the ODE.
  - `y`: Observations. ``y_{1:T}``
  - `t_pred`: Time points at which to predict the future trajectory. ``t\\_{T:T+k}``
  - `u`: Control inputs from time. ``u_{1:T+k}``
  - `ps`: Parameters for the model.
  - `st`: NamedTuple of states for different components of the model.
  - `n_samples`: Number of samples used to make the prediction.
  - `dev`: Device on which to perform the computations (CPU or GPU).
  - `kwargs`: Additional keyword arguments to pass to the solver.

Returns:

  - `x_pred`: Predicted hidden states. ``x_{T:T+k}``
  - `y_pred`: Predicted observations. ``y_{T:T+k}``
"""
function predict(model::LatentODE, solver::DiffEqBase.DEAlgorithm, y::AbstractArray, u::Union{Nothing,AbstractArray}, ts::Tuple, ps::ComponentArray, st::NamedTuple, n_samples::Int, dev::Any; kwargs...)
  ts_obs, ts_for = ts
  x̂₀, _ = model.obs_encoder(y, ts_obs, ps.obs_encoder, st.obs_encoder)[1]
  u_enc = model.ctrl_encoder(u, ps.ctrl_encoder, st.ctrl_encoder)[1]
  x̂ = sample_dynamics(model.dynamics, x̂₀, u_enc, ts_for, ps.dynamics, st.dynamics, n_samples) |> model.device
  x_pred = model.state_map(x̂, ps.state_map, st.state_map)[1]
  y_pred = model.obs_decoder(x_pred, ps.obs_decoder, st.obs_decoder)[1]
  return x_pred, y_pred
end









