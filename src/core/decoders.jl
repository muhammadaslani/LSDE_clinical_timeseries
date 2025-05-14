"""
    Decoder

A decoder is a function that takes a latent variable and produces an output (Observations or Control inputs).    

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

Constructs a linear decoder.

Arguments:

- `obs_dim`: Dimension of the observations.
- `latent_dim`: Dimension of the latent space. 
- `dist`: Type of observation noise. Default is Gaussian. Options are Gaussian, Poisson, None.

returns: 

    - The decoder.
        
"""
function Linear_Decoder(latent_dim, obs_dim, dist="Gaussian")
    if dist == "Gaussian"
        output_net = BranchLayer(Dense(latent_dim, obs_dim), Dense(latent_dim, obs_dim, softplus))
    elseif dist == "Poisson"
        output_net = Chain(Dense(latent_dim, obs_dim), x -> exp.(x))
    elseif dist == "None" 
        output_net = Dense(latent_dim, obs_dim)
    else
        error("Unknown Observation noise: $dataset_name \n Currnet supported dist: Gaussian, Poisson, None")
    end
    return Decoder(output_net)
    
end


"""
    MLP_Decoder(obs_dim, latent_dim, hidden_dim, n_hidden)

Constructs an MLP decoder.

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
        output_net = Chain(mlp, BranchLayer(Dense(hidden_size, obs_dim), Dense(hidden_size, obs_dim)))
    elseif dist == "Poisson"
        output_net = Chain(mlp, Dense(hidden_size, obs_dim), x -> exp.(x))
    elseif dist== "Classification"
        output_net = Chain(mlp, Dense(hidden_size, obs_dim))
    elseif dist == "None"
        output_net = Chain(mlp, Dense(hidden_size, obs_dim), softplus)

    else
        error("Unknown Observation noise: $dist \n Currnet supported distributions: Gaussian, Poisson, None")
    end

    return Decoder(output_net)
end



function MultiDecoder(latent_dims, obs_dims; hidden_size, depth, dist)
    decoders = [MLP_Decoder(latent_dims[i], obs_dims[i]; hidden_size=hidden_size, depth=depth, dist=dist) for i in 1:length(latent_dims)]
    return Decoder(Parallel(vcat, decoders...))
end

"""
    MultiDecoder_linear(latent_dims, obs_dims; dist)

TBW
"""
function MultiDecoder_linear(latent_dims, obs_dims; dist)
    decoders = [Linear_Decoder(latent_dims[i], obs_dims[i], dist) for i in 1:length(latent_dims)]
    return Decoder(Parallel(vcat, decoders...))
end


"""
    BranchDecoder(latent_dim, obs_dims; hidden_size, depth, dists)

Constructs a decoder with multiple branches from a single latent space.

Arguments:

- `latent_dim`: Dimension of the latent space.
- `obs_dims`: List of dimensions of the observations.
- `hidden_size`: Dimension of the hidden layers.
- `depth`: Number of hidden layers.
- `dists`: List of observation noise. Default is Gaussian. Options are Gaussian, Poisson, None.

returns: 

    - The decoder.
        
"""
function BranchDecoder(latent_dim, output_dims; hidden_size, depth, dists)
    decoders = [MLP_Decoder(latent_dim, output_dims[i]; hidden_size=hidden_size, depth=depth, dist=dists[i]) for i in eachindex(output_dims)]
    return Decoder(Parallel(nothing, decoders...))
end



"""
    BranchDecoder_linear(latent_dim, obs_dims; dists)

Constructs a decoder with multiple branches from a single latent space.

Arguments:

- `latent_dim`: Dimension of the latent space.
- `obs_dims`: List of dimensions of the observations.
- `dists`: List of observation noise. Default is Gaussian. Options are Gaussian, Poisson, None.

returns: 

    - The decoder.
        
"""
function BranchDecoder_linear(latent_dim, obs_dims; dists)
    decoders = [Linear_Decoder(latent_dim, obs_dims[i], dists[i]) for i in 1:length(obs_dims)]
    return Decoder(Parallel(nothing, decoders...))
end
