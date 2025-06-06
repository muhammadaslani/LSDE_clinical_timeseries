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
            # x -> (@show "Input shape: ", size(x);x),
            Recurrence(LSTMCell(n_features => hidden_dim); return_sequence=true),
            # x -> (@show "Encoder output shape: ", size(stack(x));x),
            Recurrence(LSTMCell(hidden_dim => latent_dim); return_sequence=false),
            # x -> (@show "Latent state shape: ", size(x);x)
        ),
        
        # Decoder: Generate predictions for forecast horizon
        decoder=Chain(
            x -> repeat(x, 1, 1, n_timepoints_for),  # Repeat latent state for each time point
            #x -> (@show "Repeated latent state shape: ", size(x);x),
            x -> reshape(x,  latent_dim,  n_timepoints_for, :),  # Reshape to (1, time, latent_dim)
            # x -> (@show "Reshaped latent state shape: ", size(x);x),
            BranchLayer(
                # Branch 1: Health status classification (6 classes for each time point)
                Chain(
                    Recurrence(LSTMCell(latent_dim => hidden_dim); return_sequence=true),
                    Recurrence(LSTMCell(hidden_dim => n_health_classes); return_sequence=true),
                    x -> stack(x, dims=2)  # Stack to get shape (n_health_classes, time, batch_size)
                 ),
                # Branch 2: Cell count prediction (Poisson rate parameter)
                Chain(
                    Recurrence(LSTMCell(latent_dim => hidden_dim); return_sequence=true),
                    Recurrence(LSTMCell(hidden_dim => 1); return_sequence=true),
                    x -> softplus.(x),  # Apply softplus to ensure positive rates
                    x-> stack(x, dims=2)  # Stack to get shape (1, time, batch_size)
                )
            )
        )
    )
    
    θ, st = Lux.setup(rng, model)
    return model, θ, st
end


rng = Random.default_rng()
Random.seed!(rng, 42)  # For reproducible results
    
    # 2. Define configuration dictionary
    # This mimics what would typically come from a YAML config file
config = Dict(
        "obs_encoder" => Dict("hidden_size" => 64),  # Hidden dimension for LSTM layers
        "latent_dim" => 32                           # Latent space dimension
    )
    
    # 3. Define data dimensions
    # These represent the structure of your PKPD dataset
dims = Dict(
        "obs_dim" => 12,      # Observation features: covariates(5) + health(6) + count(1)
        "input_dim" => 2      # Control inputs: chemo + radio treatment
    )
    
    # 4. Set forecast horizon
n_timepoints_for = 5  # Number of future time points to predict
    
    # 5. Create the model
    println("Creating RNN model...")
    model, θ, st = create_rnn_model(config, dims, rng, n_timepoints_for);


    batch_size = 32
    sequence_length = 50  # Length of input sequences
    n_features = dims["obs_dim"] + dims["input_dim"]  # Total features = 14
    sample_input = randn(Float32, n_features, sequence_length, batch_size);
    output, st_new = model(sample_input, θ, st); 