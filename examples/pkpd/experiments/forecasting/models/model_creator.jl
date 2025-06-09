using Random
using Lux
struct EncoderLSTM<: AbstractLuxLayer
    model::Chain
end
function EncoderLSTM(history_dim::Int, hidden_dim::Int, output_dim::Int)
    # Encoder: LSTM that processes history data and returns final state
    model =Chain(
        Recurrence(LSTMCell(history_dim => hidden_dim); return_sequence=true),
        Recurrence(LSTMCell(hidden_dim => output_dim); return_sequence=false),
    )
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
    outputs = []
    h, c = initial_hidden, initial_cell

    for t in 1:forecast_length
        u_t = u_inputs[:, t, :]  # Control input at timestep t
        (output, (h, c)), _ = decoder.model((u_t, (h, c)), ps, st)
        push!(outputs, output)
    end
    
    return stack(outputs, dims=2)  # Shape: (output_dim, forecast_length, batch_size)
end

# Add required Lux interface methods
Lux.initialparameters(rng::AbstractRNG, decoder::DecoderLSTM) = Lux.initialparameters(rng, decoder.model)
Lux.initialstates(rng::AbstractRNG, decoder::DecoderLSTM) = Lux.initialstates(rng, decoder.model);

struct output_head<: AbstractLuxLayer
    model::Chain
end

function output_head(hidden_dim::Int, output_dim::Vector{Int})

    model = Chain(
        BranchLayer(
            Lux.Dense(hidden_dim => output_dim[1]),
            Lux.Dense(hidden_dim => output_dim[2], softplus)
        )
    )
    return output_head(model)
end

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

function EncoderDecoderLSTM(history_dim::Int, control_dim::Int, hidden_dim::Int, output_dim::Vector{Int})
    
    # Encoder: LSTM that processes history data and returns final state (not sequence)
    encoder = EncoderLSTM(history_dim, hidden_dim, hidden_dim);

    # Decoder: LSTM that processes control inputs and returns full sequence
    decoder = DecoderLSTM(control_dim, hidden_dim, hidden_dim);
    head = output_head(hidden_dim, output_dim)  # Example output dimensions for health status and cell count

    return EncoderDecoderLSTM(encoder, decoder, head)
end

function (model::EncoderDecoderLSTM)(history_data, u_inputs, forecast_length::Int, ps, st)
    # Step 1: Encode history data to get final hidden and cell states
    encoder_output, encoder_st = model.encoder(history_data, ps.encoder, st.encoder)
    # Step 2: Use final hidden and cell states as initial states for decoder
    final_hidden = encoder_output 
    final_cell = encoder_output  
    
    # Step 3: Decode control inputs using encoder states as initial states
    decoder_output = model.decoder(u_inputs, final_hidden, final_cell, forecast_length, ps.decoder, st.decoder)
    # Step 4: Apply output head to decoder output
    predictions = model.output_head(decoder_output, ps.output_head, st.output_head)
    return predictions, st
end

input_dim = 2
hidden_dim = 4
sequence_length = 10
output_dim = [2,1] # Output dimensions for each output head
control_dim = 2
batch_size = 8

model = EncoderDecoderLSTM(input_dim, control_dim, hidden_dim, output_dim);
ps, st = Lux.setup(MersenneTwister(123), model);

history_data = rand(Float32, input_dim, sequence_length, batch_size); # Historical data
u_input = rand(Float32, control_dim, sequence_length, batch_size); # Control inputs
# Forward pass through the model
y, encoder_st = model(history_data, u_input, 5, ps, st);