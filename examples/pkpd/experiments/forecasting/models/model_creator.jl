using Random
using Lux

# VarEncoderDecoderLSTM Encoder that outputs mean and log-variance of latent distributions
struct VarEncoderLSTM <: AbstractLuxContainerLayer{(:lstm_layers, :mean_head, :logvar_head)}
    lstm_layers::Chain
    mean_head::Dense
    logvar_head::Dense
end

function VarEncoderLSTM(history_dim::Int, hidden_dim::Int, latent_dim::Int, n_layers::Int=2)
    # Create LSTM layers
    layers = []
    
    # First layer: input_dim -> hidden_dim
    push!(layers, Recurrence(LSTMCell(history_dim => hidden_dim); return_sequence=true))
    
    # Middle layers: hidden_dim -> hidden_dim
    for i in 2:(n_layers-1)
        push!(layers, Recurrence(LSTMCell(hidden_dim => hidden_dim); return_sequence=true))
    end
    
    # Last layer: hidden_dim -> hidden_dim, return only final state
    if n_layers > 1
        push!(layers, Recurrence(LSTMCell(hidden_dim => hidden_dim); return_sequence=false))
    else
        # If only one layer, go directly from input to hidden
        layers[1] = Recurrence(LSTMCell(history_dim => hidden_dim); return_sequence=false)
    end
    
    # Create LSTM chain
    lstm_chain = Chain(layers...)
    
    mean_head = Dense(hidden_dim => latent_dim)
    logvar_head = Dense(hidden_dim => latent_dim)
    
    return VarEncoderLSTM(lstm_chain, mean_head, logvar_head)
end

function (encoder::VarEncoderLSTM)(history_data, ps, st)
    lstm_output, lstm_st = encoder.lstm_layers(history_data, ps.lstm_layers, st.lstm_layers)
    
    μ, mean_st = encoder.mean_head(lstm_output, ps.mean_head, st.mean_head)
    logσ², logvar_st = encoder.logvar_head(lstm_output, ps.logvar_head, st.logvar_head)
    
    new_st = (lstm_layers = lstm_st, mean_head = mean_st, logvar_head = logvar_st)
    return (μ, logσ²), new_st
end

function reparameterize(μ, logσ², rng::AbstractRNG)
    σ = exp.(0.5f0 .* logσ²)
    ε = randn(Float32, size(μ)...)
    return μ .+ σ .* ε
end

struct VarDecoderLSTM <: AbstractLuxContainerLayer{(:latent_projection, :model)}
    latent_projection::Dense
    model::LSTMCell
end

function VarDecoderLSTM(control_dim::Int, latent_dim::Int, hidden_dim::Int)
    latent_projection = Dense(latent_dim => hidden_dim, tanh)
    lstm_cell = LSTMCell(control_dim => hidden_dim)
    
    return VarDecoderLSTM(latent_projection, lstm_cell)
end

function (decoder::VarDecoderLSTM)(u_inputs, latent_states, forecast_length::Int, ps, st)
    projected_states, proj_st = decoder.latent_projection(latent_states, ps.latent_projection, st.latent_projection)
    h, c = projected_states, projected_states
    lstm_st = st.model  # Initialize LSTM state
    
    outputs = [begin
        u_t = u_inputs[:, t, :]
        (output, (h, c)), lstm_st = decoder.model((u_t, (h, c)), ps.model, lstm_st)
        output
    end for t in 1:forecast_length]
    
    new_st = (latent_projection = proj_st, model = lstm_st)
    return stack(outputs, dims=2), new_st  # Stack outputs along the second dimension
end

struct output_head <: AbstractLuxContainerLayer{(:model,)}
    model::Chain
end

function output_head(hidden_dim::Int, output_dim::Vector{Int}, n_layers::Int=2)

    shared_layers = []
    push!(shared_layers, Dense(hidden_dim => hidden_dim, relu))
    for i in 2:(n_layers-1)
        push!(shared_layers, Dense(hidden_dim => hidden_dim, relu))
    end
    final_layer = Lux.BranchLayer(
        Dense(hidden_dim => output_dim[1]),
        Dense(hidden_dim => output_dim[2], softplus)
    )
    if n_layers > 1
        push!(shared_layers, final_layer)
        model = Chain(shared_layers...)
    else
        model = final_layer
    end
    
    return output_head(model)
end

function (head::output_head)(x, ps, st)
    y, st= head.model(x, ps.model, st.model)
    return y, (model=st,)
end

struct VarEncoderDecoderLSTM <: AbstractLuxContainerLayer{(:encoder, :decoder, :output_head)}
    encoder:: VarEncoderLSTM
    decoder:: VarDecoderLSTM
    output_head:: output_head
end

function VarEncoderDecoderLSTM(history_dim::Int, control_dim::Int, hidden_dim::Int, latent_dim::Int, output_dim::Vector{Int}, n_layers::Int=2, n_head_layers::Int=2)
    encoder = VarEncoderLSTM(history_dim, hidden_dim, latent_dim, n_layers)
    decoder = VarDecoderLSTM(control_dim, latent_dim, hidden_dim)
    head = output_head(hidden_dim, output_dim, n_head_layers)

    return VarEncoderDecoderLSTM(encoder, decoder, head)
end

function (model::VarEncoderDecoderLSTM)(history_data, u_inputs, forecast_length::Int, ps, st; rng::AbstractRNG=Random.default_rng())

    (μ, logσ²), encoder_st = model.encoder(history_data, ps.encoder, st.encoder)
    latent_states = reparameterize(μ, logσ², rng)
    decoder_output, decoder_st = model.decoder(u_inputs, latent_states, forecast_length, ps.decoder, st.decoder)
    predictions, head_st = model.output_head(decoder_output, ps.output_head, st.output_head)
    
    new_st = (
        encoder = encoder_st,
        decoder = decoder_st,
        output_head = head_st
    )
    return predictions, new_st, (μ = μ, logσ² = logσ²)
end


function create_var_encoder_decoder_lstm(history_dim::Int, control_dim::Int, hidden_dim::Int, latent_dim::Int, output_dim::Vector{Int}, rng::AbstractRNG, n_layers::Int=2, n_head_layers::Int=2)
    model = VarEncoderDecoderLSTM(history_dim, control_dim, hidden_dim, latent_dim, output_dim, n_layers, n_head_layers)
    ps, st = Lux.setup(rng, model)
    return model, ps, st
end

# KL divergence loss for VarEncoderDecoderLSTM regularization
function kl_divergence_loss(μ, logσ²)
    return 0.5f0 * sum(1.0f0 .+ logσ² .- μ.^2 .- exp.(logσ²))
end

# # Example usage and testing
# history_dim = 2
# hidden_dim = 32
# latent_dim = 64  # Latent dimension for VAE
# sequence_length = 10
# output_dim = [6,1] # Output dimensions for each output head
# control_dim = 2
# batch_size = 8
# n_layers = 2  # Reduce from 8 to 2 for debugging

# # Create VAE model
# vae_model, vae_ps, vae_st = create_var_encoder_decoder_lstm(history_dim, control_dim, hidden_dim, latent_dim, output_dim, MersenneTwister(123), n_layers);


# history_data = rand(Float32, history_dim, sequence_length, batch_size); # Historical data
# u_input = rand(Float32, control_dim, sequence_length, batch_size); # Control inputs

# # Forward pass through the VAE model
# rng = MersenneTwister(456)
# predictions, new_vae_st, vae_params = vae_model(history_data, u_input, 5, vae_ps, vae_st; rng=rng);

# # Compute KL divergence for regularization
# kl_loss = kl_divergence_loss(vae_params.μ, vae_params.logσ²)