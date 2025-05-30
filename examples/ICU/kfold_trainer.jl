
"""
    kfold_train(
        data, n_folds, rng, 
        config_path, model, 
        timepoints,
        loss_fn, eval_fn, viz_fn
    )

Performs k-fold cross-validation training on a model.

# Arguments
- `data`: The dataset, typically from load_data function.
- `n_folds`: Number of folds for cross-validation.
- `rng`: Random number generator for reproducibility.
- `config_path`: Path to the YAML configuration file.
- `model`: Type of model to train, e.g., "lsde", "lode".
- `timepoints`: Array of timepoints for the model.
- `loss_fn`: Loss function for training.
- `eval_fn`: Evaluation function.
- `viz_fn`: Visualization function.

# Returns
- A tuple containing (models, parameters, states, performances)
"""
function kfold_train(data, n_folds, rng, config_path, model_type, timepoints, loss_fn, eval_fn, viz_fn)
    # Load configuration
    config = YAML.load_file(config_path)
    exp_path = joinpath(config["experiment"]["path"], config["experiment"]["name"])
    
    # Extract the complete dataset
    inputs_data_obs, obs_data_obs, output_data_obs, masks_obs, 
    inputs_data_for, obs_data_for, output_data_for, masks_for = data
    
    # Prepare for k-fold splitting
    n_samples = size(inputs_data_obs, 3)
    sample_indices = collect(1:n_samples)
    fold_size = div(n_samples, n_folds)
    
    # Initialize storage for models, parameters, states, and performance metrics
    models = []
    trained_params = []
    states = []
    performances = []
    
    # Shuffle sample indices for randomization
    shuffle!(rng, sample_indices)
    
    # Create folds
    folds = []
    for i in 1:n_folds
        start_idx = (i-1) * fold_size + 1
        end_idx = i == n_folds ? n_samples : i * fold_size
        test_indices = sample_indices[start_idx:end_idx]
        train_indices = setdiff(sample_indices, test_indices)
        
        # Split train indices further into train and validation
        n_train = length(train_indices)
        val_size = round(Int, n_train * 0.2) # 20% for validation
        val_indices = train_indices[1:val_size]
        train_indices = train_indices[val_size+1:end]
        
        push!(folds, (train_indices, val_indices, test_indices))
    end
    
    # Define model dimensions
    dims = Dict(
        "input_dim" => size(inputs_data_obs, 1),
        "obs_dim" => size(obs_data_obs, 1),
        "output_dim" => ones(Int, size(output_data_for, 1)),
    )
    
    # Split timepoints for observation and forecasting
    timepoints_obs = timepoints[1:size(obs_data_obs, 2)]
    timepoints_for = timepoints[size(obs_data_obs, 2)+1:end]
    
    # Perform k-fold training
    for fold_idx in 1:n_folds
        @info "Training fold $fold_idx/$n_folds"
        train_indices, val_indices, test_indices = folds[fold_idx]
        # Create data loaders for this fold
         train_data = (
            inputs_data_obs[:,:,train_indices], obs_data_obs[:,:,train_indices], output_data_obs[:,:,train_indices], masks_obs[:,:,train_indices],
            inputs_data_for[:,:,train_indices], obs_data_for[:,:,train_indices], output_data_for[:,:,train_indices], masks_for[:,:,train_indices]
        )
        
        val_data = (
            inputs_data_obs[:,:,val_indices], obs_data_obs[:,:,val_indices], output_data_obs[:,:,val_indices], masks_obs[:,:,val_indices],
            inputs_data_for[:,:,val_indices], obs_data_for[:,:,val_indices], output_data_for[:,:,val_indices], masks_for[:,:,val_indices]
        )
        
        test_data = (
            inputs_data_obs[:,:,test_indices], obs_data_obs[:,:,test_indices], output_data_obs[:,:,test_indices], masks_obs[:,:,test_indices],
            inputs_data_for[:,:,test_indices], obs_data_for[:,:,test_indices], output_data_for[:,:,test_indices], masks_for[:,:,test_indices]
        )
        
        # Create data loaders
        batch_size = 32
        train_loader = DataLoader(train_data, batchsize=batch_size, shuffle=true)
        val_loader = DataLoader(val_data, batchsize=batch_size, shuffle=false)
        test_loader = DataLoader(test_data, batchsize=batch_size, shuffle=false)
        
        # Initialize model
        if model_type == "lsde"
            model, θ, st = create_latentsde(config["model"], dims, rng)
        elseif model_type == "lode"
            # We need to set the appropriate parameters for LODE
            lode_config = deepcopy(config["model"])
            #lode_config["noise_type"] = "none"
            model, θ, st = create_latentsde(lode_config, dims, rng)
        else
            error("Unsupported model type: $model_type")
        end
        
        # Train the model
        θ_trained = train(model, θ, st, timepoints_for, loss_fn, eval_fn, viz_fn, 
                          train_loader, val_loader, config["training"], exp_path)
        
        # Evaluate on test data
        u_obs, x_obs, y_obs, masks_obs_test, u_for, x_for, y_for, masks_for_test = test_loader.data
        data_obs = (u_obs, x_obs, y_obs, masks_obs_test)
        future_true_data = (u_for, x_for, y_for, masks_for_test)

        # Make predictions
        μ, σ = forecast(model, θ_trained, st, data_obs, u_for, timepoints_for, config["training"]["validation"])
        forecasted_data = (μ, σ)
        
        # Evaluate model performance without plotting
        rmse, crps = viz_fn_forecast(timepoints_obs, timepoints_for, data_obs, future_true_data, forecasted_data, plot=false)
        
        # Store model, parameters, state, and performance
        push!(models, model)
        push!(trained_params, θ_trained)
        push!(states, st)
        push!(performances, (rmse, crps))
        
        @info "Fold $fold_idx completed"
    end
    
    # Compute average performance across folds
    avg_rmse = mean([perf[1] for perf in performances])
    avg_crps = mean([perf[2] for perf in performances])
    
    @info "K-Fold Cross-Validation Results:"
    @info "Average RMSE across $n_folds folds: $avg_rmse"
    @info "Average CRPS across $n_folds folds: $avg_crps"
    
    return models, trained_params, states, performances
end

"""
    kfold_forecast(model, θ, st, data, timepoints, config, viz_fn; fold_idx=1, sample_n=1, plot=true)

Generate forecasts using a trained model from k-fold cross-validation.

# Arguments
- `model`: The trained model.
- `θ`: Trained parameters.
- `st`: Model state.
- `data`: Test data for forecasting.
- `timepoints`: Array of timepoints.
- `config`: Model configuration.
- `viz_fn`: Visualization function.
- `fold_idx`: Index of the fold (for logging purposes).
- `sample_n`: Sample number to visualize.
- `plot`: Whether to generate plots.

# Returns
- Performance metrics (and visualization if plot=true).
"""
function kfold_forecast(model, θ, st, data, timepoints, config, viz_fn; fold_idx=1, sample_n=1, plot=true)
    # Prepare data for forecasting
    inputs_data_obs, obs_data_obs, output_data_obs, masks_obs, 
    inputs_data_for, obs_data_for, output_data_for, masks_for = data
    
    # Split timepoints
    timepoints_obs = timepoints[1:size(obs_data_obs, 2)]
    timepoints_for = timepoints[size(obs_data_obs, 2)+1:end]
    
    # Prepare data for forecast function
    data_obs = (inputs_data_obs, obs_data_obs, output_data_obs, masks_obs)
    future_true_data = (inputs_data_for, obs_data_for, output_data_for, masks_for)
    
    # Generate forecast
    μ, σ = forecast(model, θ, st, data_obs, inputs_data_for, timepoints_for, config["training"]["validation"])
    forecasted_data = (μ, σ)
    
    # Evaluate and optionally visualize
    if plot
        fig, rmse, crps = viz_fn(timepoints_obs, timepoints_for, data_obs, future_true_data, forecasted_data, sample_n=sample_n, plot=true)
        return fig, rmse, crps
    else
        rmse, crps = viz_fn(timepoints_obs, timepoints_for, data_obs, future_true_data, forecasted_data, sample_n=sample_n, plot=false)
        return rmse, crps
    end
end
