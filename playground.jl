using Lux, Random

lstm = LSTMCell(10 => 20; init_state=(rng, hidden_size, batch_size) -> h₀, init_memory=(rng, hidden_size, batch_size) -> c₀);
my_state_init = @init_fn(rng -> fill(42.0f0, 20, 4), :state)
rng = Random.default_rng();
ps_lstm, st_lstm = Lux.setup(rng, lstm);
x = randn(Float32, 10, 4);
y_lstm, st_lstm_new= lstm(x, ps_lstm, st_lstm);



# Custom hidden and cell states (20 hidden units, batch size 4)
h₀ = fill(1.0f0, 20, 4);   # hidden state filled with 1.0
c₀ = fill(-1.0f0, 20, 4);  # cell state filled with -1.0

(y, (h, c)), new_state = lstm(x, ps_lstm, st_lstm);


lstm_layer= Lux.Recurrence(lstm, return_sequence=true);
ps_layer, st_layer = Lux.setup(rng, lstm_layer);
x_layer = randn(Float32, 10, 2,  4);
y_layer, st_layer_new = lstm_layer(x_layer, ps_layer, st_layer);

lstm_layer_statefull = Lux.StatefulRecurrentCell(lstm);
ps_layer_statefull, st_layer_statefull = Lux.setup(rng, lstm_layer_statefull);
y_layer_statefull, st_layer_statefull_new = lstm_layer_statefull(x_layer, ps_layer_statefull, st_layer_statefull);


ps_lstms= Lux.initialstates(rng,lstm_layer);



input_dim = 2
hidden_dim = 4
seq_len = 10
batch_size = 3

rnn = Recurrence(LSTMCell(input_dim => hidden_dim))
x = rand(Float32, input_dim, seq_len, batch_size)
ps_rnn, st_rnn = Lux.setup(rng, rnn);
# Set arbitrary initial hidden and cell states
h0 = rand(Float32, 4, 3)
c0 = rand(Float32, 4, 3)
# Create custom initial state
custom_st = (
    rng = st_rnn.rng,  # Preserve the RNG from the original state
    hidden_state = (h0, c0)
)

y = rnn(x, ps_rnn, custom_st)



using Lux
input_dim=10;
hidden_dim=20;
sequence_length = 5
batch_size = 4
# Define your LSTM and Recurrence layer
lstm = LSTMCell(input_dim => hidden_dim)
rnn = Recurrence(lstm)

# Prepare input: x of shape (input_dim, sequence_length, batch_size)
x = rand(Float32, input_dim, sequence_length, batch_size)

# Create your custom initial hidden and cell states
h0 = rand(Float32, hidden_dim, batch_size)  # Initial hidden state
c0 = rand(Float32, hidden_dim, batch_size)  # Initial cell state

# Pass the initial state as the second argument to the Recurrence layer
output = rnn(x, (h0, c0))
