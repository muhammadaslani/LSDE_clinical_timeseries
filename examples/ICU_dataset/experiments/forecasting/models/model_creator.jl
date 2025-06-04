# Function to create RNN model
function create_rnn_model(config, dims, rng, n_timepoints_for=25)
    hidden_dim = config["obs_encoder"]["hidden_size"]
    latent_dim = config["latent_dim"]
    n_features = dims["obs_dim"]+ dims["input_dim"]
    
    model = Chain(
        encoder=Chain(
            Recurrence(LSTMCell(n_features => hidden_dim); return_sequence=true),
            Recurrence(LSTMCell(hidden_dim => latent_dim); return_sequence=false)
        ),
        decoder=Chain(
            Dense(latent_dim, hidden_dim),
            BranchLayer(
                BranchLayer(Dense(hidden_dim, n_timepoints_for), Dense(hidden_dim, n_timepoints_for, softplus)),
                BranchLayer(Dense(hidden_dim, n_timepoints_for), Dense(hidden_dim, n_timepoints_for, softplus)),
                BranchLayer(Dense(hidden_dim, n_timepoints_for), Dense(hidden_dim, n_timepoints_for, softplus))
            )
        )
    )
    
    θ, st = Lux.setup(rng, model)
    return model, θ, st
end