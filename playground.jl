using Revise
using Rhythm
using Random
using Lux, DifferentialEquations, Random, SciMLSensitivity, ComponentArrays, Optimisers, OptimizationOptimisers, Statistics
using MLUtils, Printf, SciMLSensitivity, OneHotArrays, CairoMakie, Distributions
using YAML




Random.seed!(123)

# Define dimensions
obs_dim = 4
ctrl_dim = 2
latent_dim=16
hidden_size = 8
seq_len = 10
context_dim = 32
batch_size = 32

# Create simple mock components
obs_encoder = Recurrent_Encoder(obs_dim, latent_dim, context_dim; hidden_size)
init_map = NoOpLayer()
ctrl_encoder = NoOpLayer()
dynamics = LSTM(LSTMCell(latent_dim+ctrl_dim => latent_dim))
state_map = NoOpLayer()
obs_decoder = MLP_Decoder(latent_dim, obs_dim; hidden_size=hidden_size, depth=2, dist= "None")

# Create model
model = LatentLSTM(
    obs_encoder=obs_encoder,
    ctrl_encoder=ctrl_encoder,
    init_map=init_map,
    dynamics=dynamics,
    state_map=state_map,
    obs_decoder=obs_decoder
)

# Create test data
# Create batch test data
# Shape: (obs_dim, seq_len, batch_size)
y = randn(Float32, obs_dim, seq_len, batch_size);
u = randn(Float32, ctrl_dim, seq_len, batch_size);
ts = collect(1:seq_len);

# Initialize parameters and states
rng = Random.default_rng()
ps, st = Lux.setup(rng, model);
ps = ComponentArray(ps);

# Test forward pass


config= YAML.load_file("examples/pkpd/configs/PkPD_config_latent_lstm.yml")["model"]
model, ps, st = create_latent_lstm(config, Dict("input_dim" => ctrl_dim, "obs_dim" => obs_dim, "output_dim" => [obs_dim, obs_dim]), rng);

ŷ, px₀, kl_path = model(y, u, ts, ps, st);

xx,yy= predict(model, y, u, ts, ps, st, 5, CPUDevice());