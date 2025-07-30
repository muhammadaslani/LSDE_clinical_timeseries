"""
    Decoder

A decoder is a function that takes latent variables and produces outputs (Observations or Control inputs).    

"""
struct Decoder{ON} <: AbstractLuxWrapperLayer{(:output_net)}
    output_net::ON      
end

"""
    (model::Decoder)(x::AbstractArray, p::ComponentVector, st::NamedTuple)


The forward pass of the decoder.

Arguments:

- `x`: The input to the decoder.
- `p`: The parameters.
- `st`: The state.

returns:

    - 'ŷ': The output of the decoder.
    - 'st': The state of the decoder.

"""
function(model::Decoder)(x, p, st)
    ŷ, st = model.output_net(x, p, st)
    return ŷ, st
end


"""
    Identity_Decoder()

Constructs an identity decoder. Useful for fully observable systems.

"""
function Identity_Decoder()
    output_net = Lux.NoOpLayer()
    return Decoder(output_net)
end

"""
    Linear_Decoder(obs_dim, latent_dim) 

Constructs a linear decoder (mapping from latent space to observation space without any hidden layers).

Arguments:

- `obs_dim`: Dimension of the observations.
- `latent_dim`: Dimension of the latent space. 
- `dist`: Type of observation noise. Default is Gaussian. Options are Gaussian, Poisson, None.

returns: 

    - The decoder.
        
"""
function Linear_Decoder(latent_dim, obs_dim, dist="Gaussian")
    if dist == "Gaussian"
        # Assumption: the output is a Gaussian distribution with mean and log variance
        output_net = BranchLayer(Dense(latent_dim, obs_dim), Dense(latent_dim, obs_dim))
    elseif dist == "Poisson"
        # Assumption: the output is a Poisson distribution mean rate, should be positive
        output_net = Chain(Dense(latent_dim, obs_dim), x -> exp.(x))
    elseif dist == "None"
        # Assumption: can be used for Classification tasks as well, where the output is raw logits (softmax applied later (loss function))
        output_net = Dense(latent_dim, obs_dim)
    else
        error("Unknown Observation noise: $dataset_name \n Currnet supported dist: Gaussian, Poisson, None")
    end
    return Decoder(output_net)
    
end


"""
    MLP_Decoder(obs_dim, latent_dim, hidden_dim, n_hidden)

Constructs an MLP decoder (mapping from latent space to observation space with hidden layers).

Arguments:

- `obs_dim`: Dimension of the observations.
- `latent_dim`: Dimension of the latent space.
- `hidden_size`: Dimension of the hidden layers.
- `depth`: Number of hidden layers.
- `dist`: Type of observation noise. Default is Gaussian. Options are Gaussian, Poisson, None.

returns: 

    - The decoder.
        
"""
function MLP_Decoder(latent_dim, obs_dim; hidden_size, depth, dist)

    mlp = Chain(Dense(latent_dim => hidden_size), [Dense(hidden_size, hidden_size) for i in 1:depth]...)
    if dist == "Gaussian"
        # Assumption: the output is a Gaussian distribution with mean and log variance
        output_net = Chain(mlp, BranchLayer(Dense(hidden_size, obs_dim), Dense(hidden_size, obs_dim)))
    elseif dist == "Poisson"
        # Assumption: the output is a Poisson distribution mean rate, should be positive
        output_net = Chain(mlp, Dense(hidden_size, obs_dim), x -> exp.(x))
    elseif dist == "None"
        # Assumption: can be used for Classification tasks as well, where the output is raw logits (softmax applied later (loss function))
        output_net = Chain(mlp, Dense(hidden_size, obs_dim))

    else
        error("Unknown Observation noise: $dist \n Currnet supported distributions: Gaussian, Poisson, None")
    end

    return Decoder(output_net)
end

"""
    MultiHeadMLPDecoder(obs_dims, latent_dim, hidden_dim, n_hidden)

Constructs a multi-head MLP decoder (mapping from latent space to observation space with hidden layers).

Arguments:

- `obs_dim`: Dimension of the observations.
- `latent_dim`: Dimension of the latent space.
- `hidden_size`: Dimension of the hidden layers.
- `depth`: Number of hidden layers.
- `dists`: Type of observation noises. Default is Gaussian. Options are Gaussian, Poisson, None.

returns: 

    - The decoders.
      
"""


function MultiHeadMLPDecoder(latent_dim, output_dims; hidden_size, depth, dists)
    decoders = [MLP_Decoder(latent_dim, output_dims[i]; hidden_size=hidden_size, depth=depth, dist=dists[i]) for i in eachindex(dists)]
    return Decoder(Parallel(nothing, decoders...))
end


"""
    MultiHeadLinearDecoder(latent_dim, obs_dims; dists)

Constructs a decoder with multiple branches from a single latent space.

Arguments:

- `latent_dim`: Dimension of the latent space.
- `obs_dims`: List of dimensions of the observations.
- `dists`: List of observation noise. Default is Gaussian. Options are Gaussian, Poisson, None.

returns: 

    - The decoders.
        
"""
function MultiHeadLinearDecoder(latent_dim, output_dims; dists)
    decoders = [Linear_Decoder(latent_dim, output_dims[i], dists[i]) for i in eachindex(dists)]
    return Decoder(Parallel(nothing, decoders...))
end


