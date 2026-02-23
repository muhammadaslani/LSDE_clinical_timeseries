
"""
    Encoder

An encoder is a container layer that contains three networks: `linear_net`, `init_net`, and `context_net`.

# Fields

- `linear_net`: A linear network that maps observations to a hidden representation.
- `init_net`: A network that maps the hidden representation to the hidden state
- `context_net`: A network that maps the hidden representation to the context.
"""
struct Encoder <: AbstractLuxContainerLayer{(:linear_net, :init_net, :context_net)}
    linear_net
    init_net
    context_net
end


"""
    (model::Encoder)(x::AbstractArray, p::ComponentVector, st::NamedTuple)

The forward pass of the encoder.

Arguments:

- `x`: The input to the encoder (e.g. observations).
- `p`: The parameters.
- `st`: The state of the encoder.

returns:

    - `x̂₀`: The initial hidden state.
    - `context`: The context.

"""
function (model::Encoder)(x, p, st)
    x_, st1 = model.linear_net(x, p.linear_net, st.linear_net)
    px₀, st2 = model.init_net(x_, p.init_net, st.init_net)
    context, st3 = model.context_net(x_, p.context_net, st.context_net)
    st = (st1, st2, st3)
    return (px₀, context), st
end


"""
    Identity_Encoder()

Constructs an identity encoder. Useful for fully observed systems.
    
"""
function Identity_Encoder()
    linear_net = Lux.NoOpLayer()
    init_net = Lux.BranchLayer(Lux.SelectDim(2, 1), Lux.SelectDim(2, 1))
    context_net = Lux.NoOpLayer()
    return Encoder(linear_net, init_net, context_net)
end


"""
    Recurrent_Encoder(obs_dim, latent_dim, context_dim, hidden_dim)

Constructs a recurrent encoder. Useful for partially observed systems. 

Arguments:

- `obs_dim`: Dimension of the observations.
- `latent_dim`: Dimension of the latent space.
- `context_dim`: Dimension of the context.
- `hidden_dim`: Dimension of the hidden state.
"""
function Recurrent_Encoder(obs_dim, latent_dim, context_dim; hidden_size)
    linear_net = Dense(obs_dim => hidden_size)

    init_net = Chain(
        ReverseSequence(dim=2),
        Recurrence(LSTMCell(hidden_size => hidden_size), return_sequence=true),
        Recurrence(LSTMCell(hidden_size => hidden_size)),
        BranchLayer(Dense(hidden_size => latent_dim), Dense(hidden_size => latent_dim)))

    context_net = Chain(
        Recurrence(LSTMCell(hidden_size => hidden_size); return_sequence=true),
        Recurrence(LSTMCell(hidden_size => context_dim); return_sequence=true),
        x -> stack(x; dims=2))

    return Encoder(linear_net, init_net, context_net)

end


# -----------------------------------------------------------------------
# CDE Encoder: linear_net → init_net (GRU backward) → CDE → proj_net (μ, σ)
# -----------------------------------------------------------------------

"""
    CDE_Encoder

An encoder that follows the same pattern as the standard `Encoder` and `Recurrent_Encoder`:

1. `linear_net`: Projects raw observations into `hidden_size`.
2. `init_net`: Runs a LSTM backward over the projected history and returns a
   **deterministic** initial condition for the encoder CDE.
3. `cde`: A CDE that integrates forward over the observation window,
   driven by the projected history as a control path.
4. `proj_net`: A BranchLayer that maps the CDE's terminal state to
   probabilistic initial conditions `(μ, σ)` for the latent model.

# Fields

- `linear_net`: Dense layer that projects observations to hidden size.
- `init_net`: Chain with backward LSTM that produces deterministic CDE initial conditions.
- `cde`: CDE dynamics that encodes the observation trajectory.
- `proj_net`: BranchLayer that maps CDE output to `(μ, σ)`.
"""
struct CDE_Encoder <: AbstractLuxContainerLayer{(:linear_net, :init_net, :cde, :proj_net)}
    linear_net    # Dense: obs_dim → hidden_size
    init_net      # Chain(ReverseSequence, LSTM backward) → deterministic z₀ for CDE
    cde           # CDE dynamics: encode trajectory
    proj_net      # BranchLayer: z_enc_final → (μ, σ)
    hidden_size::Int
end


"""
    CDE_Encoder(encoder_path_dim, latent_dim, context_dim=0; hidden_size=64, depth=1)

Construct a CDE encoder following the standard encoder pattern.

Arguments:

- `encoder_path_dim`: Total dimension of the encoder input (covars + obs + controls).
- `latent_dim`: Dimension of the latent space.
- `context_dim`: Dimension of context (ignored for now, provided for API compatibility).
- `hidden_size`: Hidden layer size for the projection and CDE vector field.
- `depth`: Number of hidden layers in the CDE vector field.
"""
function CDE_Encoder(encoder_path_dim::Int, latent_dim::Int, context_dim::Int=0;
    hidden_size::Int=64, depth::Int=1)

    # 1. Project raw observations to hidden_size
    linear_net = Dense(encoder_path_dim => hidden_size)

    # 2. Backward LSTM over projected history → deterministic initial condition for the encoder CDE
    init_net = Chain(
        ReverseSequence(dim=2),
        Recurrence(LSTMCell(hidden_size => hidden_size); return_sequence=true),
        Recurrence(LSTMCell(hidden_size => hidden_size)),
    )

    # 3. Encoder CDE: vector field built here (like ODE factory)
    #    input_dim = hidden_size (projected observations), path_dim = hidden_size + 1 (time channel)
    encoder_vf = CDEField(hidden_size, hidden_size + 1; hidden_size, depth)
    cde = CDE(encoder_vf)

    # 4. Project CDE terminal state to (μ, σ) for latent model
    proj_net = BranchLayer(Dense(hidden_size => latent_dim), Dense(hidden_size => latent_dim))

    return CDE_Encoder(linear_net, init_net, cde, proj_net, hidden_size)
end


"""
    (enc::CDE_Encoder)(y, t_obs, ps, st)

Forward pass of the CDE encoder.

Arguments:

- `y`: History data `(covars + obs + controls, T_obs, B)` — concatenated history.
- `t_obs`: Observation time points `(T_obs,)`.
- `ps`: Parameters.
- `st`: States.

Returns:

- `(px₀, nothing)`: where `px₀ = (μ, σ)` is the probabilistic initial condition.
- `st_new`: Updated states.
"""
function (enc::CDE_Encoder)(y::AbstractArray{<:Real,3}, t_obs::AbstractVector,
    ps::ComponentArray, st::NamedTuple)
    _, T_obs, B = size(y)

    # 1. Project observations to hidden_size
    y_proj, st_linear = enc.linear_net(y, ps.linear_net, st.linear_net)
    # y_proj: (hidden_size, T_obs, B)

    # 2. Run backward LSTM over projected history → deterministic z₀ for CDE
    h0, st_init = enc.init_net(y_proj, ps.init_net, st.init_net)
    # h0: (hidden_size, B) — deterministic initial condition

    # 3. Encoder CDE: pass y_proj as input, CDE handles interpolation internally
    z_enc, st_cde = enc.cde(h0, y_proj, t_obs, ps.cde, st.cde)
    # z_enc: (hidden_size, T_obs, B)

    # 4. Extract final state and project to (μ, σ)
    z_enc_final = z_enc[:, end, :]   # (hidden_size, B)
    px₀, st_proj = enc.proj_net(z_enc_final, ps.proj_net, st.proj_net)

    st_new = (linear_net=st_linear, init_net=st_init, cde=st_cde, proj_net=st_proj)
    return (px₀, nothing), st_new
end

