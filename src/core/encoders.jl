
"""
    Encoder

An encoder is a container layer that contains three networks: `linear_net`, `init_net`, and `context_net`.

# Fields

- `linear_net`: A linear network that maps observations to a hidden representation.
- `init_net`: A network that maps the hidden representation to the hidden state
- `context_net`: A network that maps the hidden representation to the context.
"""
struct Encoder  <: Lux.AbstractExplicitContainerLayer{(:linear_net, :init_net, :context_net)}
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
function(model::Encoder)(x, p, st)
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
function Recurrent_Encoder(obs_dim, latent_dim, context_dim, hidden_size)
    linear_net = Dense(obs_dim => hidden_size)

    init_net = Chain(
                    x -> reverse(x, dims=2),
                    Recurrence(LSTMCell(hidden_size=>hidden_size)), 
                    BranchLayer(Dense(hidden_size => latent_dim), Dense(hidden_size => latent_dim, softplus)))
    
    context_net = Chain(x -> reverse(x, dims=2),
                        Recurrence(LSTMCell(hidden_size=>context_dim); return_sequence=true),
                        x -> stack(x; dims=2))
    
    return Encoder(linear_net, init_net, context_net)

end