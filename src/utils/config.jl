function create_object(object_dict::Dict, args...)
    if haskey(TYPE_MAP, object_dict["type"])
        type = TYPE_MAP[object_dict["type"]]
        symbol_params = Dict{Symbol, Any}()
        for (k, v) in object_dict
            if k != "type"
                key = Symbol(k)
                if k == "activation"
                    symbol_params[key] = eval(Meta.parse(v))
                else
                    symbol_params[key] = v
                end
            end
        end
        
        return type(args...; symbol_params...)
    else
        error("Unknown type: $(object_dict["type"])")
    end
end



function create_latentsde(config::Dict, dims::Dict, rng::AbstractRNG)
    latent_dim = config["latent_dim"]::Int
    context_dim = config["context_dim"]::Int
    input_dim = dims["input_dim"]::Int
    obs_dim= dims["obs_dim"]::Union{Int, Vector{Int}}
    output_dim = dims["output_dim"]::Union{Int, Vector{Int}}
    
    if output_dim isa Int
        state_map = NoOpLayer()
    else
        #state_map = Parallel(nothing, [NoOpLayer() for _ in output_dim]...)
        state_map = NoOpLayer()
    end
    obs_encoder = create_object(config["obs_encoder"], sum(obs_dim), latent_dim, context_dim)
    drift = create_object(config["SDE"]["drift"], [latent_dim, input_dim], latent_dim)
    drift_aug = create_object(config["SDE"]["drift_aug"], [latent_dim, context_dim, input_dim], latent_dim)
    diffusion = create_object(config["SDE"]["diffusion"], latent_dim, latent_dim)

    sde_kwargs = Dict{Symbol, Any}(Symbol(k) => Float32.(v) for (k, v) in config["SDE"]["kwargs"])
    dynamics = SDE(drift, drift_aug, diffusion, eval(Meta.parse(config["SDE"]["solver"])), sde_kwargs)

    obs_decoder = create_object(config["obs_decoder"], latent_dim, output_dim)

    model =  LatentSDE(;obs_encoder, dynamics, state_map, obs_decoder)
    #println(model)
    θ, st = Lux.setup(rng, model);
    θ = θ |> ComponentArray{Float32};
    return model, θ, st

end 



function create_latentode(config::Dict, dims::Dict, rng::AbstractRNG)
    latent_dim = config["latent_dim"]::Int
    context_dim = config["context_dim"]::Int
    input_dim = dims["input_dim"]::Int
    obs_dim= dims["obs_dim"]::Union{Int, Vector{Int}}
    output_dim = dims["output_dim"]::Union{Int, Vector{Int}}
    
    if output_dim isa Int
        state_map = NoOpLayer()
    else
        state_map = NoOpLayer()
    end
    obs_encoder = create_object(config["obs_encoder"], sum(obs_dim), latent_dim, context_dim)
    vector_field = create_object(config["ODE"]["vector_field"], [latent_dim, input_dim], latent_dim)
    ode_kwargs = Dict{Symbol, Any}(Symbol(k) => Float32.(v) for (k, v) in config["ODE"]["kwargs"])
    dynamics = ODE(vector_field, eval(Meta.parse(config["ODE"]["solver"])), ode_kwargs)

    obs_decoder = create_object(config["obs_decoder"], latent_dim, output_dim)

    model = LatentODE(obs_encoder=obs_encoder, dynamics=dynamics, init_map=NoOpLayer(), state_map=state_map, obs_decoder=obs_decoder)
    θ, st = Lux.setup(rng, model);
    θ = θ |> ComponentArray{Float32};
    return model, θ, st

end 