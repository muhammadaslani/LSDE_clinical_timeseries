using Lux, Random

input_dim = 2
hidden_dim = 4
sequence_length = 10
output_dim = 2
control_dim = 2
batch_size = 8


struct encoder_lstm
    model
    ps
    st
end
function encoder_lstm(history_dim::Int, hidden_dim::Int, output_dim::Int)
    # Encoder: LSTM that processes history data and returns final state
    model =Chain(
        Recurrence(LSTMCell(history_dim => hidden_dim); return_sequence=true),
        Recurrence(LSTMCell(hidden_dim => output_dim); return_sequence=false),
    )
    ps, st = Lux.setup(MersenneTwister(123), model)
    return encoder_lstm(model, ps, st)
end

function (encoder::encoder_lstm)(history_data)
    # Forward pass through encoder LSTM
    output, st = encoder.model(history_data, encoder.ps, encoder.st)
    # Return final hidden and cell states
    return output, st
end

# Implement parameterlength for encoder_lstm
function Lux.parameterlength(encoder::encoder_lstm)
    return Lux.parameterlength(encoder.model)
end

encoder = encoder_lstm(input_dim, hidden_dim, output_dim);
history_data = rand(Float32, input_dim, sequence_length, batch_size); # (history_dim, sequence_length, batch_size)
# Forward pass through encoder
encoder_output, st= encoder(history_data);


struct decoder_lstm
    model
    ps
    st
end

function decoder_lstm(control_dim::Int, hidden_dim::Int, output_dim::Int, forecast_length::Int=1)
    # Decoder: LSTM that processes control inputs and returns full sequence
    layers = []
    ps=[]
    st=[]
    for i in 1:forecast_length
        # Each timestep processes control inputs and returns output
        lstm_cell = LSTMCell(control_dim => output_dim)
        push!(layers, lstm_cell)
        lstm_ps, lstm_st = Lux.setup(MersenneTwister(123), lstm_cell)
        push!(ps, lstm_ps)
        push!(st, lstm_st)
    end
    return decoder_lstm(layers, ps, st)
end

function (decoder::decoder_lstm)(u_inputs, final_hidden, final_cell)
    y=[]
    for i in 1:length(decoder.model)
        # Use final hidden and cell states from encoder as initial states
        u_input = u_inputs[:, i, :]  #
        # Extract control input for this timestep   
        (output,(h_i,c_i)), decoder.st[i]= decoder.model[i]((u_input, (final_hidden, final_cell)), decoder.ps[i], decoder.st[i])
        # Update final hidden and cell states for next timestep
        final_hidden, final_cell = h_i, c_i
        # Store output for this timestep
        push!(y, output)
    end
    return stack(y, dims=2)  # Stack outputs along sequence dimension
end

# Implement parameterlength for decoder_lstm
function Lux.parameterlength(decoder::decoder_lstm)
    return sum(Lux.parameterlength(layer) for layer in decoder.model)
end


decoder= decoder_lstm(control_dim, hidden_dim, output_dim, sequence_length);
u_input= rand(Float32, control_dim, sequence_length, batch_size); # (control_dim, forecast_length, batch_size)
final_hidden = rand(Float32, output_dim, batch_size); # (hidden_dim, batch_size)
final_cell = rand(Float32, output_dim, batch_size); # (hidden_dim, batch_size)
# Forward pass through decoder
decoder_output=decoder(u_input, final_hidden, final_cell);


struct EncoderDecoderLSTM <: AbstractLuxContainerLayer{(:encoder, :decoder)}
    encoder
    decoder
end

function create_encoder_decoder_lstm(history_dim::Int, control_dim::Int, hidden_dim::Int, output_dim::Int, forecast_length::Int=1)
    
    # Encoder: LSTM that processes history data and returns final state (not sequence)
    encoder = encoder_lstm(history_dim, hidden_dim, output_dim);
    
    # Decoder: LSTM that processes control inputs and returns full sequence
    decoder = decoder_lstm(control_dim, hidden_dim, output_dim, forecast_length);

    return EncoderDecoderLSTM(encoder, decoder)
end
encoder_decoder = create_encoder_decoder_lstm(input_dim, control_dim, hidden_dim, output_dim, sequence_length);


function (model::EncoderDecoderLSTM)(history_data, u_inputs)
    # Step 1: Encode history data to get final hidden and cell states
    encoder_output, encoder_st = model.encoder(history_data)
    final_hidden = encoder_output  # Final hidden and cell states from encoder
    final_cell = encoder_output  # Assuming encoder returns both hidden and cell states as output
    
    # Step 3: Decode control inputs using encoder states as initial states
    predictions = model.decoder(u_inputs, final_hidden, final_cell)
    return predictions, encoder_st
end



y, encoder_st = encoder_decoder(history_data, u_input);