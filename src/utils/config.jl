function create_object(object_dict::Dict, args...)
    if haskey(TYPE_MAP, object_dict["type"])
        type = TYPE_MAP[object_dict["type"]]
        symbol_params = Dict(Symbol(k) => v for (k, v) in object_dict if k != "type")
        return type(args...; symbol_params...)
    else
        error("Unknown type: $type_name")
    end
end


function create_latentsde(config::Dict, dims::Dict)
    latent_dim = config["latent_dim"]::Int
    context_dim = config["context_dim"]::Int
    input_dim = dims["input_dim"]::Int
    output_dim = dims["output_dim"]::Int

    obs_encoder = create_object(config["obs_encoder"], output_dim, latent_dim, context_dim)

    drift = create_object(config["SDE"]["drift"], [latent_dim, input_dim], latent_dim)
    drift_aug = create_object(config["SDE"]["drift_aug"], [latent_dim, context_dim, input_dim], latent_dim)
    diffusion = create_object(config["SDE"]["diffusion"], latent_dim, latent_dim)

    dynamics = SDE(drift, diffusion, drift_aug, eval(Meta.parse(config["SDE"]["solver"])))

    obs_decoder = create_object(config["obs_decoder"], latent_dim, output_dim)

    return LatentSDE(;obs_encoder, dynamics, obs_decoder)
end

