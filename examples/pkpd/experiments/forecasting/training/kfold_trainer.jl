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
- `model`: Type of model to train, e.g., "lsde", "lode", or "rnn".
- `timepoints`: Array of timepoints for the model.
- `loss_fn`: Loss function for training.
- `eval_fn`: Evaluation function.
- `forecast_fn`: Forecast function.
- `viz_fn`: Visualization function.

# Returns
- A tuple containing (models, parameters, states, performances)
"""
function kfold_train_pkpd(data, dims, n_folds, rng, config_path, model_type, timepoints_for, loss_fn, eval_fn, forecast_fn, viz_fn)
    # Start timing the entire k-fold training process
    start_time = time()
    
    # Load configuration
    config = YAML.load_file(config_path)
    exp_path = joinpath(config["experiment"]["path"], config["experiment"]["name"])
    
    # Extract the complete dataset
    inputs_data_obs, covars_data_obs, x_data_obs, y₁_data_obs, y₂_data_obs, mask₁_data_obs, mask₂_data_obs,
        inputs_data_for, covars_data_for, x_data_for, y₁_data_for, y₂_data_for, mask₁_data_for, mask₂_data_for = data

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

    
    
    # Perform k-fold training with independent warm start for each fold
    for fold_idx in 1:n_folds
        @info "Training fold $fold_idx/$n_folds"
        train_indices, val_indices, test_indices = folds[fold_idx]
        
        # Create data loaders for this fold
        train_data = (
            inputs_data_obs[:,:,train_indices], covars_data_obs[:,:,train_indices], x_data_obs[:,:,train_indices], y₁_data_obs[:,:,train_indices], y₂_data_obs[:,:,train_indices], mask₁_data_obs[:,:,train_indices], mask₂_data_obs[:,:,train_indices],
            inputs_data_for[:,:,train_indices], covars_data_for[:,:,train_indices], x_data_for[:,:,train_indices], y₁_data_for[:,:,train_indices], y₂_data_for[:,:,train_indices], mask₁_data_for[:,:,train_indices], mask₂_data_for[:,:,train_indices]
        )
        
        val_data = (
            inputs_data_obs[:,:,val_indices], covars_data_obs[:,:,val_indices], x_data_obs[:,:,val_indices], y₁_data_obs[:,:,val_indices], y₂_data_obs[:,:,val_indices], mask₁_data_obs[:,:,val_indices], mask₂_data_obs[:,:,val_indices],
            inputs_data_for[:,:,val_indices], covars_data_for[:,:,val_indices], x_data_for[:,:,val_indices], y₁_data_for[:,:,val_indices], y₂_data_for[:,:,val_indices], mask₁_data_for[:,:,val_indices], mask₂_data_for[:,:,val_indices]
        )

        test_data = (
            inputs_data_obs[:,:,test_indices], covars_data_obs[:,:,test_indices], x_data_obs[:,:,test_indices], y₁_data_obs[:,:,test_indices], y₂_data_obs[:,:,test_indices], mask₁_data_obs[:,:,test_indices], mask₂_data_obs[:,:,test_indices],
            inputs_data_for[:,:,test_indices], covars_data_for[:,:,test_indices], x_data_for[:,:,test_indices], y₁_data_for[:,:,test_indices], y₂_data_for[:,:,test_indices], mask₁_data_for[:,:,test_indices], mask₂_data_for[:,:,test_indices]
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
            model, θ, st = create_latentode(lode_config, dims, rng)
        elseif model_type == "rnn"
            model, θ, st = create_var_encoder_decoder_lstm(dims["obs_dim"]+dims["input_dim"], dims["input_dim"], config["model"]["encoder"]["hidden_size"],config["model"]["encoder"]["latent_dim"], dims["output_dim"], rng, config["model"]["encoder"]["num_layers"])
        else
            error("Unsupported model type: $model_type")
        end
        training_config = deepcopy(config["training"])
        warm_start_epochs = round(Int, training_config["epochs"] * 0.1)
        training_config["epochs"] = warm_start_epochs
        @info "Fold $fold_idx: warm start training for $warm_start_epochs epochs (10% of total)"
        θ_warm_start = train(model, θ, st, timepoints_for, loss_fn, eval_fn, viz_fn, 
                           train_loader, val_loader, training_config, exp_path)
        
        # Step 2: Continue training for remaining 90% of epochs using warm start parameters
        remaining_epochs = round(Int, config["training"]["epochs"] * 0.9)
        training_config["epochs"] = remaining_epochs
        @info "Fold $fold_idx: continuing training for $remaining_epochs epochs (90% of total) using warm start parameters"
                θ_trained = train(model, θ_warm_start, st, timepoints_for, loss_fn, eval_fn, viz_fn, 
                         train_loader, val_loader, training_config, exp_path)
        
        # Evaluate on test data
        input_data_obs_test, covars_data_obs_test, x_data_obs_test, y₁_data_obs_test, y₂_data_obs_test, masks₁_data_obs_test, masks₂_data_obs_test,
         input_data_for_test, covars_data_forecast_test, x_data_for_test, y₁_data_for_test, y₂_data_for_test, masks₁_for_test, masks₂_for_test = test_loader.data

        data_obs = (input_data_obs_test, covars_data_obs_test, x_data_obs_test, y₁_data_obs_test, y₂_data_obs_test, masks₁_data_obs_test, masks₂_data_obs_test)
        future_true_data = (input_data_for_test, covars_data_forecast_test, x_data_for_test, y₁_data_for_test, y₂_data_for_test, masks₁_for_test, masks₂_for_test)

        # Make predictions
        Ex, Ey_pred = forecast_fn(model, θ_trained, st, data_obs, input_data_for_test, timepoints_for, config["training"]["validation"])
        forecasted_data = (Ex, Ey_pred)
        crossentropy_health, rmse_tumor, nll_count = viz_fn(timepoints_obs, timepoints_for, data_obs, future_true_data, forecasted_data, plot=false)
        
        # Store model, parameters, state, and performance
        push!(models, model)
        push!(trained_params, θ_trained)
        push!(states, st)
        push!(performances, (crossentropy_health, rmse_tumor, nll_count))
        
        @info "Fold $fold_idx completed: Health cross-entropy=$crossentropy_health, Tumor RMSE=$rmse_tumor, Count NLL=$nll_count"
    end
    
    # Compute average performance across folds
    if model_type == "rnn"
        avg_crossentropy_health = mean([perf[1] for perf in performances])
        avg_rmse_tumor = mean([perf[2] for perf in performances])
        avg_nll_count = mean([perf[3] for perf in performances])
    else
        # Extract the three specific metrics for LSDE/LODE models
        crossentropy_health_values = [perf[1] for perf in performances]
        rmse_tumor_values = [perf[2] for perf in performances]
        nll_count_values = [perf[3] for perf in performances]
        
        # Calculate averages
        avg_crossentropy_health = mean(crossentropy_health_values)
        avg_rmse_tumor = mean(rmse_tumor_values)
        avg_nll_count = mean(nll_count_values)
     end
    
    # Calculate total training time
    end_time = time()
    total_training_time = end_time - start_time
    
    @info "K-Fold Cross-Validation Results:"
    @info "Average Health Cross-entropy across $n_folds folds: $avg_crossentropy_health"
    @info "Average Tumor RMSE across $n_folds folds: $avg_rmse_tumor"
    @info "Average Cell Count NLL across $n_folds folds: $avg_nll_count"
    @info "Total training time: $(round(total_training_time, digits=2)) seconds ($(round(total_training_time/60, digits=2)) minutes)"
    
    return models, trained_params, states, performances
end
