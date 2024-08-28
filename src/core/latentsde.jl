"""
    LatentSDE(obs_encoder, ctrl_encoder, dynamics, obs_decoder, ctrl_decoder)

Constructs a Latent Stochastic Differential Equation model.

Arguments:

  - `obs_encoder`: A function that encodes the observations `y` to get the initial hidden state `x₀` and context.
  - `ctrl_encoder`: A function that encodes (high-dimensional) inputs/controls to a lower-dimensional representation if needed.
  - `init_map`: A function that maps the sampled initial conditions to plausible valubles for the dynamics. 
  - `dynamics`: A function that models the dynamics of the system (the SDE)
  - `state_map`: A function that selects the relevant hidden states for the observation decoder.
  - `obs_decoder`: A function that decodes the hidden states `x` to the observations `y`.
  - 'ctrl_decoder': A function that decodes the control representation to the original control space if needed.

"""
@with_kw struct LatentSDE <: LatentVariableModel
    obs_encoder = Identity_Encoder()
    ctrl_encoder = NoOpLayer()
    init_map = NoOpLayer()
    dynamics 
    state_map = NoOpLayer()
    obs_decoder = Identity_Decoder()
    ctrl_decoder = NoOpLayer()
end 

"""
    (model::LatentSDE)(y::AbstractArray, u::Union{Nothing, AbstractArray}, ts::AbstractArray, ps::ComponentArray, st::NamedTuple)

The forward pass of the LatentSDE model. 
Used for fitting the model to data.

Arguments:

  - `y`: Observations
  - `u`: Control inputs
  - `ts`: Time points
  - `ps`: Parameters
  - `st`: NamedTuple of states 

Returns:

  - `ŷ`: Decoded observations from the hidden states.
  - `ū`: Decoded control inputs from the hidden states.
  - `x̂₀`: Encoded initial hidden state.
  - `kl_path`: KL divergence path. (Only for SDE dynamics, otherwise `nothing`)ƒ
"""
function (model::LatentSDE)(y::AbstractArray, u::Union{Nothing, AbstractArray}, ts::AbstractArray, ps::ComponentArray, st::NamedTuple)
    px₀, context = model.obs_encoder(y, ps.obs_encoder, st.obs_encoder)[1]
    x₀ = model.init_map(sample_rp(px₀), ps.init_map, st.init_map)[1]
    x₀_aug = CRC.@ignore_derivatives fill!(similar(x₀, 1, size(x₀)[2]), 0.0f0)
    x₀  = vcat(x₀, x₀_aug)
    u_enc = model.ctrl_encoder(u, ps.ctrl_encoder, st.ctrl_encoder)[1]
    x_sol = model.dynamics(x₀, u_enc, context, ts, ps.dynamics, st.dynamics)[1]
    x_arr = cat(x_sol.u..., dims = 3)
    x_ = permutedims(x_arr, (1, 3, 2))
    x = x_[1:end-1, :, :]
    kl_path = x_[end, :, :]
    x = model.state_map(x, ps.state_map, st.state_map)[1]
    ŷ = model.obs_decoder(x, ps.obs_decoder, st.obs_decoder)[1]
    return ŷ, px₀, kl_path 
end




"""
    predict(model::LatentSDE, solver::DiffEqBase.DEAlgorithm, y::AbstractArray, t_obs::AbstractArray, t_pred::AbstractArray, u::Union{Nothing, AbstractArray}, ps::ComponentArray, st::NamedTuple, n_samples::Int, dev::Device)

  Predicts the future trajectory of the system given the observations from time `T` to `T+k` given observations from time `1` to `T` and control inputs from time `1` to `T+k`.
  Used for forecasting and control applications.

      ```math
      p(y_{T:T+k \\mid y_{1:T}, u_{1:T+k}}) \\quad p(x_{T:T+k \\mid y_{1:T}, u_{1:T+k}})
      ```

Arguments:

  - `model`: The `LatentSDE` model to sample from.
  - `solver`: The nummerical solver to solve the SDE.
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
function predict(model::LatentSDE, solver::DiffEqBase.DEAlgorithm, y::AbstractArray, t_pred::AbstractArray, u::Union{Nothing, AbstractArray}, ps::ComponentArray, st::NamedTuple, n_samples::Int, dev::Any; kwargs...)
    pxₜ, _ = model.obs_encoder(y, ps.obs_encoder, st.obs_encoder)[1] 
    u_enc = model.ctrl_encoder(u, ps.ctrl_encoder, st.ctrl_encoder)[1]
    x_pred = sample_generative(model.dynamics, model.init_map, solver, pxₜ, u_enc, t_pred, ps, st, n_samples, dev; kwargs...)
    x_pred = model.state_map(x_pred, ps.state_map, st.state_map)[1]
    y_pred = model.obs_decoder(x_pred, ps.obs_decoder, st.obs_decoder)[1]
    return x_pred, y_pred
end 


""" 
    filter(model::LatentSDE, solver::DiffEqBase.DEAlgorithm, y::AbstractArray, u::Union{Nothing, AbstractArray}, ts::AbstractArray, ps::ComponentArray, st::NamedTuple, n_samples::Int, dev::Device)

    Estimates the hidden state of the system at time `t` given the observations from time `1` to `t` and control inputs from time `1` to `t`.
    Typically used for online tracking/monitoring.

      ```math
      p(x_{t \\mid y_{1:t}, u_{1:t}})
      ```

Arguments:

  - `model`: The `LatentSDE` model to sample from.
  - `solver`: The nummerical solver to solve the SDE.
  - `y`: Observations. ``y_{1:t}``
  - `u`: Control inputs from time. ``u_{1:t}``
  - `ts`: Time points at which the observations are made. ``t\\_{0:t}``
  - `ps`: Parameters for the model.
  - `st`: NamedTuple of states for different components of the model.
  - `n_samples`: Number of samples used to make the prediction.
  - `dev`: Device on which to perform the computations (CPU or GPU).
  - `kwargs`: Additional keyword arguments to pass to the solver.

Returns:
  - `x̂`: Estimated hidden states. ``x_{t}``
"""
function filter(model::LatentSDE, solver::DiffEqBase.DEAlgorithm, y::AbstractArray, u::Union{Nothing, AbstractArray}, ts::AbstractArray, ps::ComponentArray, st::NamedTuple, n_samples::Int, dev::Any; kwargs...)
    px₀, context = model.obs_encoder(y, ps.obs_encoder, st.obs_encoder)[1]
    u_enc = model.ctrl_encoder(u, ps.ctrl_encoder, st.ctrl_encoder)[1]
    x_ = sample_augmented(model.dynamics, model.init_map, solver, px₀, u_enc, context, ts, ps, st, n_samples, dev; kwargs...)
    x = model.state_map(x_, ps.state_map, st.state_map)[1]
    return x[:,end, :, :]
end



""" 
    smooth(model::LatentSDE, solver::DiffEqBase.DEAlgorithm, y::AbstractArray, u::Union{Nothing, AbstractArray}, ts::AbstractArray, ps::ComponentArray, st::NamedTuple, n_samples::Int, dev::Device)

    Estimates the hidden states of the system at times `1` to `T` given the observations from time `1` to `T` and control inputs from time `1` to `T`.
    Typically used for offline applications. i.e. understanding system dynamics.

      ```math
      p(x_{1:t \\mid y_{1:T}, u_{1:T}})
      ```

Arguments:

  - `model`: The `LatentSDE` model to sample from.
  - `solver`: The nummerical solver to solve the SDE.
  - `y`: Observations. ``y_{1:T}``
  - `u`: Control inputs from time. ``u_{1:T}``
  - `ts`: Time points at which the observations are made. ``t\\_{1:T}``
  - `ps`: Parameters for the model.
  - `st`: NamedTuple of states for different components of the model.
  - `n_samples`: Number of samples used to make the prediction.
  - `dev`: Device on which to perform the computations (CPU or GPU).

Returns:
  - `x̃`: Estimated hidden states. ``x_{1:T}``
  - `ỹ`: Reconstructed observations. ``y_{1:T}``
"""
function smooth(model::LatentSDE, solver::DiffEqBase.DEAlgorithm, y::AbstractArray, u::Union{Nothing, AbstractArray}, ts::AbstractArray, ps::ComponentArray, st::NamedTuple, n_samples::Int, dev::Any; kwargs...)
    px₀, context = model.obs_encoder(y, ps.obs_encoder, st.obs_encoder)[1]
    u_enc = model.ctrl_encoder(u, ps.ctrl_encoder, st.ctrl_encoder)[1]
    x̃ = sample_augmented(model.dynamics, model.init_map, solver, px₀, u_enc, context, ts, ps, st, n_samples, dev; kwargs...)
    x̃ = model.state_map(x̃, ps.state_map, st.state_map)[1]
    ỹ = model.obs_decoder(x̃, ps.obs_decoder, st.obs_decoder)[1]
    return x̃, ỹ
end




""" 
    generate(model::LatentSDE, solver::DiffEqBase.DEAlgorithm, px₀::Tuple, u::Union{Nothing, AbstractArray}, ts::AbstractArray, ps::ComponentArray, st::NamedTuple, n_samples::Int, dev::Device)

    Generates new samples from the generative model (potentially conditioned on control inputs)

      ```math
      q(x_{1:T}, y_{1:T} | u_{1:T})
      ```

Arguments:

  - `model`: The `LatentSDE` model to sample from.
  - `solver`: The nummerical solver to solve the SDE.
  - `px₀`: The distribution of the initial condition (mean and vaiance)
  - `u`: Control inputs from time. ``u_{1:t}``
  - `ts`: Time points at which the observations are made. ``t\\_{0:t}``
  - `ps`: Parameters for the model.
  - `st`: NamedTuple of states for different components of the model.
  - `n_samples`: Number of samples used to make the prediction.
  - `dev`: Device on which to perform the computations (CPU or GPU).
  - `kwargs`: Additional keyword arguments to pass to the solver.

Returns:
  - `x̂`: Generated hidden states. ``x_{1:T}``
  - `ŷ`: Generated observations. ``y_{1:T}``
"""

function generate(model::LatentSDE, solver::DiffEqBase.DEAlgorithm, px₀::Tuple, u::Union{Nothing, AbstractArray}, ts::AbstractArray, ps::ComponentArray, st::NamedTuple, n_samples::Int, dev::Any; kwargs...)
  u_enc = model.ctrl_encoder(u, ps.ctrl_encoder, st.ctrl_encoder)[1]
  x̂ = sample_generative(model.dynamics, model.init_map, solver, px₀, u_enc, ts, ps, st, n_samples, dev; kwargs...)
  x̂ = model.state_map(x̂, ps.state_map, st.state_map)[1]
  ŷ = model.obs_decoder(x̂, ps.obs_decoder, st.obs_decoder)[1]
  return x̂, ŷ
end