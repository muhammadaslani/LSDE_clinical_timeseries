# Function to create RNN model for PKPD dataset
function create_rnn_model(config, dims, rng, n_timepoints_for=25)
    hidden_dim = config["obs_encoder"]["hidden_size"]
    latent_dim = config["latent_dim"]
    
    # Calculate input features: obs features (covariates + health + count) + control inputs
    n_obs_features = dims["obs_dim"]  # This should be sum of covariates (5) + health (6) + count (1) = 12
    n_input_features = dims["input_dim"]  # This should be control inputs (2 for chemo/radio)
    n_features = n_obs_features + n_input_features
    
    # Health status classes (0-5, so 6 classes total)
    n_health_classes = 6
    
    model = Chain(
        # Encoder: Process sequential input data
        encoder=Chain(
            Recurrence(LSTMCell(n_features => hidden_dim); return_sequence=true),
            Recurrence(LSTMCell(hidden_dim => latent_dim); return_sequence=false)
        ),
        
        # Decoder: Generate predictions for forecast horizon
        decoder=Chain(
            Dense(latent_dim, hidden_dim, relu),
            BranchLayer(
                # Branch 1: Health status classification (6 classes for each time point)
                Chain(
                    #Dense(hidden_dim, hidden_dim, relu),
                    Dense(hidden_dim, n_health_classes * n_timepoints_for),
                    x -> reshape(x, n_health_classes, n_timepoints_for, :)  # Reshape to (classes, time, batch)
                ),
                # Branch 2: Cell count prediction (Poisson rate parameter)
                Chain(
                    #Dense(hidden_dim, hidden_dim, relu),
                    Dense(hidden_dim, n_timepoints_for, softplus),  # softplus ensures positive rates
                    x -> reshape(x, 1, n_timepoints_for, :)  # Reshape to (1, time, batch)
                )
            )
        )
    )
    
    θ, st = Lux.setup(rng, model)
    return model, θ, st
end
