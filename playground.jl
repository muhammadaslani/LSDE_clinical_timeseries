using Lux, Random 
include("/Volumes/Mine/Academic/PhD/Codes/Packages/Rhythm.jl/examples/pkpd/experiments/forecasting/models/model_creator.jl");
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