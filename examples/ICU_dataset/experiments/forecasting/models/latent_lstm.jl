using Random
using Lux

struct obs_encoder <: AbstractLuxContainerLayer{(:backbone, :mu_head, :logvar_head)}
    backbone::Chain
    mu_head::Dense
    logvar_head::Dense
end

function obs_encoder(obs_dim::Int, hidden_dim::Int, latent_dim::Int, depth::Int=2)
    # Create LSTM layers
    layers = []
    
    # First layer: input_dim -> hidden_dim
    push!(layers, Recurrence(LSTMCell(obs_dim => hidden_dim); return_sequence=true))
    
    # Middle layers: hidden_dim -> hidden_dim
    for i in 2:(depth-1)
        push!(layers, Recurrence(LSTMCell(hidden_dim => hidden_dim); return_sequence=true))
    end
    
    # Last layer: hidden_dim -> hidden_dim, return only final state
    if depth > 1
        push!(layers, Recurrence(LSTMCell(hidden_dim => hidden_dim); return_sequence=false))
    else
        # If only one layer, go directly from input to hidden
        layers[1] = Recurrence(LSTMCell(obs_dim => hidden_dim); return_sequence=false)
    end
    
    # Create LSTM chain
    lstm_chain = Chain(layers...)

    mu_head = Dense(hidden_dim => latent_dim)
    logvar_head = Dense(hidden_dim => latent_dim)

    return obs_encoder(lstm_chain, mu_head, logvar_head)
end

function (encoder::obs_encoder)(obs_data, ps, st)
    hidden_features, backbone_st = encoder.backbone(obs_data, ps.backbone, st.backbone)

    μ, mu_st = encoder.mu_head(hidden_features, ps.mu_head, st.mu_head)
    logσ², logvar_st = encoder.logvar_head(hidden_features, ps.logvar_head, st.logvar_head)

    new_st = (backbone = backbone_st, mu_head = mu_st, logvar_head = logvar_st)
    return (μ, logσ²), new_st
end

function reparameterize(μ, logσ², rng::AbstractRNG)
    σ = exp.(0.5f0 .* logσ²)
    ε = randn(Float32, size(μ)...)
    return μ .+ σ .* ε
end

struct dynamics <: AbstractLuxContainerLayer{(:model, )}
    model::LSTMCell
end

function dynamics(control_dim::Int, latent_dim::Int)
    lstm_cell = LSTMCell(latent_dim + control_dim => latent_dim)
    return dynamics(lstm_cell)
end

function (dynamics::dynamics)(u_inputs, latent_init_states, forecast_length::Int, ps, st)
    h= latent_init_states
    c = zeros(eltype(latent_init_states), size(latent_init_states))
    lstm_st = st  # Initialize LSTM state
    output = zeros(eltype(latent_init_states), size(latent_init_states))

    outputs = [begin
        u_t = u_inputs[:, t, :]
        (output, (h, c)), lstm_st = dynamics.model((vcat(output, u_t), (h, c)), ps.model, lstm_st)
        output
    end for t in 1:forecast_length]

    new_st = (model = lstm_st)
    return stack(outputs, dims=2), new_st  # Stack outputs along the second dimension
end

struct obs_decoder <: AbstractLuxContainerLayer{(:backbone, :output_heads)}
    backbone::Chain
    output_heads::BranchLayer
end

function obs_decoder(latent_dim::Int, hidden_dim::Int, output_dim::Vector{Int}, depth::Int=2)
    # Create shared backbone layers
    backbone_layers = []
    push!(backbone_layers, Dense(latent_dim => hidden_dim, tanh))
    for i in 2:depth
        push!(backbone_layers, Dense(hidden_dim => hidden_dim, tanh))
    end
    backbone = Chain(backbone_layers...)
    
    # Create output heads
    output_heads = BranchLayer(
        BranchLayer(
            Dense(hidden_dim => output_dim[1]),    # First output head: mean (μ)
            Dense(hidden_dim => output_dim[2])     # First output head: log variance (logσ²)
        ),
        BranchLayer(
            Dense(hidden_dim => output_dim[1]),    # Second output head: mean (μ)
            Dense(hidden_dim => output_dim[2])     # Second output head: log variance (logσ²)
        ),
        BranchLayer(
            Dense(hidden_dim => output_dim[1]),     # Third output head: mean (μ)
            Dense(hidden_dim => output_dim[2])     # Third output head: log variance (logσ²)
        )
    )
    
    return obs_decoder(backbone, output_heads)
end

function (decoder::obs_decoder)(x, ps, st)
    # First pass through shared backbone
    hidden_features, backbone_st = decoder.backbone(x, ps.backbone, st.backbone)

    # Then through output heads
    outputs, heads_st = decoder.output_heads(hidden_features, ps.output_heads, st.output_heads)
    
    new_st = (backbone = backbone_st, output_heads = heads_st)
    return outputs, new_st
end



struct latent_lstm <: AbstractLuxContainerLayer{(:obs_encoder, :dynamics, :obs_decoder)}
    obs_encoder:: obs_encoder
    dynamics:: dynamics
    obs_decoder:: obs_decoder
end

function latent_lstm(obs_dim::Int, control_dim::Int, hidden_dim::Int, latent_dim::Int, output_dim::Vector{Int}, encoder_depth::Int=2, decoder_depth::Int=2)
    encoder = obs_encoder(obs_dim, hidden_dim, latent_dim, encoder_depth)
    dyn = dynamics(LSTMCell(latent_dim + control_dim => latent_dim))
    decoder = obs_decoder(latent_dim, hidden_dim, output_dim, decoder_depth)
    return latent_lstm(encoder, dyn, decoder)
end

function (model::latent_lstm)(history_data, u_inputs, forecast_length::Int, ps, st; rng::AbstractRNG=Random.default_rng())
    (μ, logσ²), obs_encoder_st = model.obs_encoder(history_data, ps.obs_encoder, st.obs_encoder)
    latent_init_states = reparameterize(μ, logσ², rng)
    latent_states, dynamics_st = model.dynamics(u_inputs, latent_init_states, forecast_length, ps.dynamics, st.dynamics)
    predictions, obs_decoder_st = model.obs_decoder(latent_states, ps.obs_decoder, st.obs_decoder)

    new_st = (
        obs_encoder = obs_encoder_st,
        dynamics = dynamics_st,
        obs_decoder = obs_decoder_st
    )
    return predictions, new_st, (μ = μ, logσ² = logσ²)
end


function create_latent_lstm(obs_dim::Int, control_dim::Int, hidden_dim::Int, latent_dim::Int, output_dim::Vector{Int}, rng::AbstractRNG, encoder_depth::Int=2, decoder_depth::Int=2)
    model = latent_lstm(obs_dim, control_dim, hidden_dim, latent_dim, output_dim, encoder_depth, decoder_depth)
    ps, st = Lux.setup(rng, model)
    return model, ps, st
end

function predict(model::latent_lstm, obs_data, u_inputs, forecast_length::Int, ps, st; mcmc_samples::Int=2, rng::AbstractRNG=Random.default_rng())
    """
    Inference function for latent lstm model with MCMC sampling.
    
    Args:
        model: The latent lstm model
        history_data: Historical data for encoding (history_dim, sequence_length, batch_size)
        u_inputs: Control inputs for decoding (control_dim, forecast_length, batch_size)
        forecast_length: Number of steps to forecast
        ps: Model parameters
        st: Model state
        mcmc_samples: Number of MCMC samples to generate per training sample
        rng: Random number generator
    
    Returns:
        predictions: Model predictions with proper structure for each output branch
        new_st: Updated model state
    """
    
    # Encode history to latent representation (only once per batch)
    (μ, logσ²), obs_encoder_st = model.obs_encoder(obs_data, ps.obs_encoder, st.obs_encoder)

    # Initialize containers for predictions
    dynamics_st = st.dynamics
    obs_decoder_st = st.obs_decoder
    all_predictions = []
    
    # Generate multiple samples
    for sample_idx in 1:mcmc_samples
        # Sample from latent distribution for this MCMC sample
        latent_init_states = reparameterize(μ, logσ², rng)
        latent_states, dynamics_st = model.dynamics(u_inputs, latent_init_states, forecast_length, ps.dynamics, st.dynamics)
        predictions, obs_decoder_st = model.obs_decoder(latent_states, ps.obs_decoder, obs_decoder_st)

        # Store predictions for this sample
        push!(all_predictions, predictions)
    end
    
    # Handle the nested tuple structure from BranchLayer
    # predictions is ((branch1_out1, branch1_out2), (branch2_out1, branch2_out2), (branch3_out1, branch3_out2))
    
    # Extract each branch and output
    branch1_out1_samples = stack([pred[1][1] for pred in all_predictions], dims=4)
    branch1_out2_samples = stack([pred[1][2] for pred in all_predictions], dims=4)
    
    branch2_out1_samples = stack([pred[2][1] for pred in all_predictions], dims=4)
    branch2_out2_samples = stack([pred[2][2] for pred in all_predictions], dims=4)
    
    branch3_out1_samples = stack([pred[3][1] for pred in all_predictions], dims=4)
    branch3_out2_samples = stack([pred[3][2] for pred in all_predictions], dims=4)
    
    stacked_predictions = (
        (branch1_out1_samples, branch1_out2_samples),
        (branch2_out1_samples, branch2_out2_samples),
        (branch3_out1_samples, branch3_out2_samples)
    )
    
    # Update state (using the last decoder and head states)
    new_st = (
        obs_encoder = obs_encoder_st,
        dynamics = dynamics_st,
        obs_decoder = obs_decoder_st
    )
    return stacked_predictions, new_st
end
