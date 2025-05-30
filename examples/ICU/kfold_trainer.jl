"""
    kfold_train(
        data, n_folds, rng, 
        config_path, model, 
        timepoints,
        loss_fn, eval_fn, viz_fn
    )

Performs k-fold cross-validation training on a model with warm start initialization.

The warm start strategy works as follows:
- Each fold independently trains for 10% of the total epochs first, then continues 
  for the remaining 90% using the same fold's 10% trained parameters as initialization

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
    # Start timing the entire k-fold training process
    start_time = time()
    
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
    
    # Perform k-fold training with independent warm start for each fold
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
        
        batch_size = 32
        train_loader = DataLoader(train_data, batchsize=batch_size, shuffle=true)
        val_loader = DataLoader(val_data, batchsize=batch_size, shuffle=false)
        test_loader = DataLoader(test_data, batchsize=batch_size, shuffle=false)
        
        # Initialize model for this fold
        if model_type == "lsde"
            model, θ, st = create_latentsde(config["model"], dims, rng)
        elseif model_type == "lode"
            lode_config = deepcopy(config["model"])
            model, θ, st = create_latentsde(lode_config, dims, rng)
        else
            error("Unsupported model type: $model_type")
        end
        
        # Prepare training configuration for warm start
        training_config = deepcopy(config["training"])
        
        # Step 1: Train for 10% of epochs (warm start phase)
        warm_start_epochs = round(Int, training_config["epochs"] * 0.1)
        training_config["epochs"] = warm_start_epochs
        @info "Fold $fold_idx: warm start training for $warm_start_epochs epochs (10% of total)"
        
        # Train the model for warm start
        θ_warm_start = train(model, θ, st, timepoints_for, loss_fn, eval_fn, viz_fn, 
                           train_loader, val_loader, training_config, exp_path)
        
        # Step 2: Continue training for remaining 90% of epochs using warm start parameters
        remaining_epochs = round(Int, config["training"]["epochs"] * 0.9)
        training_config["epochs"] = remaining_epochs
        @info "Fold $fold_idx: continuing training for $remaining_epochs epochs (90% of total) using warm start parameters"
        
        # Train for the remaining epochs using warm start parameters as initialization
        θ_trained = train(model, θ_warm_start, st, timepoints_for, loss_fn, eval_fn, viz_fn, 
                         train_loader, val_loader, training_config, exp_path)
        
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
    
    # Calculate total training time
    end_time = time()
    total_training_time = end_time - start_time
    
    @info "K-Fold Cross-Validation Results:"
    @info "Average RMSE across $n_folds folds: $avg_rmse"
    @info "Average CRPS across $n_folds folds: $avg_crps"
    @info "Total training time: $(round(total_training_time, digits=2)) seconds ($(round(total_training_time/60, digits=2)) minutes)"
    
    return models, trained_params, states, performances
end
