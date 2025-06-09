using Random
using Lux
struct EncoderLSTM<: AbstractLuxLayer
    model::Chain
end
function EncoderLSTM(history_dim::Int, hidden_dim::Int, output_dim::Int, n_layers::Int=2)
    # Create a vector to store the layers
    layers = []
    
    # First layer: input_dim -> hidden_dim
    push!(layers, Recurrence(LSTMCell(history_dim => hidden_dim); return_sequence=true))
    
    # Middle layers: hidden_dim -> hidden_dim
    for i in 2:(n_layers-1)
        push!(layers, Recurrence(LSTMCell(hidden_dim => hidden_dim); return_sequence=true))
    end
    
    # Last layer: hidden_dim -> output_dim, return only final state
    if n_layers > 1
        push!(layers, Recurrence(LSTMCell(hidden_dim => output_dim); return_sequence=false))
    else
        # If only one layer, go directly from input to output
        layers[1] = Recurrence(LSTMCell(history_dim => output_dim); return_sequence=false)
    end
    
    # Create chain with all layers
    model = Chain(layers...)
    return EncoderLSTM(model)
end

function (encoder::EncoderLSTM)(history_data, ps, st)
    # Forward pass through encoder LSTM
    output, st = encoder.model(history_data, ps, st)
    # Return final hidden and cell states
    return output, st
end
# Add required Lux interface methods
Lux.initialparameters(rng::AbstractRNG, encoder::EncoderLSTM) = Lux.initialparameters(rng, encoder.model)
Lux.initialstates(rng::AbstractRNG, encoder::EncoderLSTM) = Lux.initialstates(rng, encoder.model)

struct DecoderLSTM<: AbstractLuxLayer
    model::LSTMCell
end

function DecoderLSTM(control_dim::Int, hidden_dim::Int, output_dim::Int)

    lstm_cell= LSTMCell(control_dim => output_dim)
    return DecoderLSTM(lstm_cell)
end


function (decoder::DecoderLSTM)(u_inputs, initial_hidden, initial_cell, forecast_length::Int, ps, st)
    h, c = initial_hidden, initial_cell
    outputs = [begin
        u_t = u_inputs[:, t, :]
        (output, (h, c)), st = decoder.model((u_t, (h, c)), ps, st)
        output
    end for t in 1:forecast_length]
    return stack(outputs, dims=2), st  # Stack outputs along the second dimension
end


# Add required Lux interface methods
Lux.initialparameters(rng::AbstractRNG, decoder::DecoderLSTM) = Lux.initialparameters(rng, decoder.model)
Lux.initialstates(rng::AbstractRNG, decoder::DecoderLSTM) = Lux.initialstates(rng, decoder.model);

struct output_head<: AbstractLuxLayer
    model::Chain
end

function output_head(hidden_dim::Int, output_dim::Vector{Int}, n_layers::Int=2)
    # Create layers for the shared part
    shared_layers = []
    
    # First layer: hidden_dim -> hidden_dim
    push!(shared_layers, Dense(hidden_dim => hidden_dim, relu))
    
    # Middle layers: hidden_dim -> hidden_dim
    for i in 2:(n_layers-1)
        push!(shared_layers, Dense(hidden_dim => hidden_dim, relu))
    end
    
    # Create the final branching layer
    final_layer = BranchLayer(
        Dense(hidden_dim => output_dim[1]),
        Dense(hidden_dim => output_dim[2], softplus)
    )
    
    # Combine shared layers with final branching layer
    if n_layers > 1
        push!(shared_layers, final_layer)
        model = Chain(shared_layers...)
    else
        # If only one layer, just use the branching layer
        model = final_layer
    end
    
    return output_head(model)
end

function (head::output_head)(x, ps, st)
    return head.model(x, ps, st)
end

# Add required Lux interface methods
Lux.initialparameters(rng::AbstractRNG, head::output_head) = Lux.initialparameters(rng, head.model)
Lux.initialstates(rng::AbstractRNG, head::output_head) = Lux.initialstates(rng, head.model)

function (head::output_head)(x, ps, st)
    return head.model(x, ps, st)
end

# Add required Lux interface methods
Lux.initialparameters(rng::AbstractRNG, head::output_head) = Lux.initialparameters(rng, head.model)
Lux.initialstates(rng::AbstractRNG, head::output_head) = Lux.initialstates(rng, head.model)


struct EncoderDecoderLSTM <: AbstractLuxContainerLayer{(:encoder, :decoder, :output_head)}
    encoder:: EncoderLSTM
    decoder:: DecoderLSTM
    output_head:: output_head
end

function EncoderDecoderLSTM(history_dim::Int, control_dim::Int, hidden_dim::Int, output_dim::Vector{Int}, n_layers::Int=2, n_head_layers::Int=2)
    
    # Encoder: LSTM that processes history data and returns final state (not sequence)
    encoder = EncoderLSTM(history_dim, hidden_dim, hidden_dim, n_layers);

    # Decoder: LSTM that processes control inputs and returns full sequence
    decoder = DecoderLSTM(control_dim, hidden_dim, hidden_dim);
    head = output_head(hidden_dim, output_dim, n_head_layers)

    return EncoderDecoderLSTM(encoder, decoder, head)
end

function (model::EncoderDecoderLSTM)(history_data, u_inputs, forecast_length::Int, ps, st)
    # Step 1: Encode history data to get final hidden and cell states
    encoder_output, encoder_st = model.encoder(history_data, ps.encoder, st.encoder)
    # Step 2: Use final hidden and cell states as initial states for decoder
    final_hidden = encoder_output 
    final_cell = encoder_output  
    
    # Step 3: Decode control inputs using encoder states as initial states
    decoder_output, decoder_st = model.decoder(u_inputs, final_hidden, final_cell, forecast_length, ps.decoder, st.decoder)

    # Step 4: Apply output head to decoder output
    predictions, head_st = model.output_head(decoder_output, ps.output_head, st.output_head)
    new_st = (
        encoder = encoder_st,  # Encoder state remains unchanged
        decoder = decoder_st,  # Decoder doesn't return updated state in your implementation
        output_head = head_st
    )
    return predictions, new_st
end


function creat_encoder_decoder_lstm(history_dim::Int, control_dim::Int, hidden_dim::Int, output_dim::Vector{Int}, rng::AbstractRNG, n_layers::Int=2, n_head_layers::Int=2)
    model= EncoderDecoderLSTM(history_dim, control_dim, hidden_dim, output_dim, n_layers, n_head_layers)
    ps, st= Lux.setup(rng, model)
    return model, ps, st
end
history_dim = 2
hidden_dim = 64
sequence_length = 10
output_dim = [6,1] # Output dimensions for each output head
control_dim = 2
batch_size = 8
n_layers=8
model, ps, st = creat_encoder_decoder_lstm(history_dim, control_dim, hidden_dim, output_dim, MersenneTwister(123), n_layers);

history_data = rand(Float32, history_dim, sequence_length, batch_size); # Historical data
u_input = rand(Float32, control_dim, sequence_length, batch_size); # Control inputs
# Forward pass through the model
y, new_st = model(history_data, u_input, 5, ps, st);