"""
    Latent LSTM(obs_encoder, ctrl_encoder, dynamics, obs_decoder, ctrl_decoder)

Constructs a Latent LSTM model.

Arguments:

  - `obs_encoder`: A function that encodes the observations `y` to get the initial hidden state `x₀` and context.
  - `ctrl_encoder`: A function that encodes (high-dimensional) inputs/controls to a lower-dimensional representation if needed.
  - `init_map`: A function that maps the sampled initial conditions to plausible valubles for the dynamics. 
  - `dynamics`: A function that models the dynamics of the system (the LSTM)
  - `state_map`: A function that selects the relevant hidden states for the observation decoder.
  - `obs_decoder`: A function that decodes the hidden states `x` to the observations `y`.
  - 'ctrl_decoder': A function that decodes the control representation to the original control space if needed.
"""

@with_kw struct LatentLSTM <: LatentVariableModel
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
    (model::LatentLSTM)(y::AbstractArray, u::Union{Nothing, AbstractArray}, ts::AbstractArray, ps::ComponentArray, st::NamedTuple)

The forward pass of the LatentLSTM model. 
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
  - `kl_path`: KL divergence path. (Only for LSTM dynamics, otherwise `nothing`)
"""

function (model::LatentLSTM)(y::AbstractArray, u::Union{Nothing, AbstractArray}, ts::AbstractArray, ps::ComponentArray, st::NamedTuple)
    px₀, _ = model.obs_encoder(y, ps.obs_encoder, st.obs_encoder)[1]
    x₀ = model.init_map(sample_rp(px₀), ps.init_map, st.init_map)[1]
    u_enc = model.ctrl_encoder(u, ps.ctrl_encoder, st.ctrl_encoder)[1]
    x = model.dynamics(x₀, u_enc, ts, ps.dynamics, st.dynamics)[1]
    kl_path = nothing
    x = model.state_map(x, ps.state_map, st.state_map)[1]
    ŷ = model.obs_decoder(x, ps.obs_decoder, st.obs_decoder)[1]
    return ŷ, px₀, kl_path 
end


"""
    predict(model::LatentLSTM,  y::AbstractArray, t_obs::AbstractArray, t_pred::AbstractArray, u::Union{Nothing, AbstractArray}, ps::ComponentArray, st::NamedTuple, n_samples::Int, dev::Device)

  Predicts the future trajectory of the system from time `T` to `T+k` given observations from time `1` to `T` and control inputs from time `1` to `T+k`.
  Used for forecasting and control applications.

      ```math
      p(y_{T:T+k \\mid y_{1:T}, u_{1:T+k}}) \\quad p(x_{T:T+k \\mid y_{1:T}, u_{1:T+k}})
      ```

Arguments:

  - `model`: The `LatentLSTM` model to sample from.
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


function predict(model::LatentLSTM, y::AbstractArray, u::Union{Nothing, AbstractArray}, t_pred::AbstractArray, ps::ComponentArray, st::NamedTuple, n_samples::Int, dev::Any; kwargs...)
    x̂₀, _ = model.obs_encoder(y, ps.obs_encoder, st.obs_encoder)[1]
    u_enc = model.ctrl_encoder(u, ps.ctrl_encoder, st.ctrl_encoder)[1]
    x̂ = sample_dynamics(model.dynamics, x̂₀, u_enc, t_pred, ps.dynamics, st.dynamics, n_samples) |> model.device
    x_pred = model.state_map(x̂, ps.state_map, st.state_map)[1]
    y_pred = model.obs_decoder(x_pred, ps.obs_decoder, st.obs_decoder)[1]
    return x_pred, y_pred
end







