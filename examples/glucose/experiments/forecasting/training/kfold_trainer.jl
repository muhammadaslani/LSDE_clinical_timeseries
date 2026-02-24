function kfold_train(data, dims, n_folds, rng, config_path, model_type, timepoints, loss_fn, eval_fn, forecast_fn, viz_fn)
    start_time = time()

    config = YAML.load_file(config_path)
    exp_path = joinpath(config["experiment"]["path"], config["experiment"]["name"])

    # Extract dataset (10-element tuple)
    u_data_obs, covars_data_obs, x_data_obs, y_data_obs, mask_data_obs,
    u_data_for, covars_data_for, x_data_for, y_data_for, mask_data_for = data

    n_samples = size(u_data_obs, 3)
    sample_indices = collect(1:n_samples)
    fold_size = div(n_samples, n_folds)

    models = []
    trained_params = []
    states = []
    performances = []

    shuffle!(rng, sample_indices)

    # Create folds
    folds = []
    for i in 1:n_folds
        start_idx = (i - 1) * fold_size + 1
        end_idx = i == n_folds ? n_samples : i * fold_size
        test_indices = sample_indices[start_idx:end_idx]
        train_indices = setdiff(sample_indices, test_indices)

        n_train = length(train_indices)
        val_size = round(Int, n_train * 0.2)
        val_indices = train_indices[1:val_size]
        train_indices = train_indices[val_size+1:end]

        push!(folds, (train_indices, val_indices, test_indices))
    end

    for fold_idx in 1:n_folds
        @info "Training fold $fold_idx/$n_folds"
        train_idx, val_idx, test_idx = folds[fold_idx]

        train_data = (
            u_data_obs[:, :, train_idx], covars_data_obs[:, :, train_idx], x_data_obs[:, :, train_idx], y_data_obs[:, :, train_idx], mask_data_obs[:, :, train_idx],
            u_data_for[:, :, train_idx], covars_data_for[:, :, train_idx], x_data_for[:, :, train_idx], y_data_for[:, :, train_idx], mask_data_for[:, :, train_idx]
        )
        val_data = (
            u_data_obs[:, :, val_idx], covars_data_obs[:, :, val_idx], x_data_obs[:, :, val_idx], y_data_obs[:, :, val_idx], mask_data_obs[:, :, val_idx],
            u_data_for[:, :, val_idx], covars_data_for[:, :, val_idx], x_data_for[:, :, val_idx], y_data_for[:, :, val_idx], mask_data_for[:, :, val_idx]
        )
        test_data = (
            u_data_obs[:, :, test_idx], covars_data_obs[:, :, test_idx], x_data_obs[:, :, test_idx], y_data_obs[:, :, test_idx], mask_data_obs[:, :, test_idx],
            u_data_for[:, :, test_idx], covars_data_for[:, :, test_idx], x_data_for[:, :, test_idx], y_data_for[:, :, test_idx], mask_data_for[:, :, test_idx]
        )

        batch_size = 16
        train_loader = DataLoader(train_data, batchsize=batch_size, shuffle=true)
        val_loader = DataLoader(val_data, batchsize=batch_size, shuffle=false)
        test_loader = DataLoader(test_data, batchsize=batch_size, shuffle=false)

        # Initialize model
        if model_type == "lsde"
            model, θ, st = create_latentsde(config["model"], dims, rng)
        elseif model_type == "lode"
            model, θ, st = create_latentode(config["model"], dims, rng)
        elseif model_type == "latent_lstm"
            model, θ, st = create_latent_lstm(config["model"], dims, rng)
        elseif model_type == "latent_cde"
            model, θ, st = create_latent_cde(config["model"], dims, rng)
        else
            error("Unsupported model type: $model_type")
        end

        # Warm start: 10% epochs
        training_config = deepcopy(config["training"])
        warm_start_epochs = round(Int, training_config["epochs"] * 0.1)
        training_config["epochs"] = warm_start_epochs

        @info "Fold $fold_idx: warm start for $warm_start_epochs epochs"
        θ_warm = train(model, θ, st, timepoints, loss_fn, eval_fn, viz_fn,
            train_loader, val_loader, training_config, exp_path)

        # Continue: remaining 90% epochs
        remaining_epochs = round(Int, config["training"]["epochs"] * 0.9)
        training_config["epochs"] = remaining_epochs
        @info "Fold $fold_idx: continuing for $remaining_epochs epochs"
        θ_trained = train(model, θ_warm, st, timepoints, loss_fn, eval_fn, viz_fn,
            train_loader, val_loader, training_config, exp_path)

        # Evaluate on test set
        u_obs_test, covars_obs_test, x_obs_test, y_obs_test, mask_obs_test,
        u_for_test, covars_for_test, x_for_test, y_for_test, mask_for_test = test_loader.data

        data_obs = (u_obs_test, covars_obs_test, x_obs_test, y_obs_test, mask_obs_test)
        future_true = (u_for_test, covars_for_test, x_for_test, y_for_test, mask_for_test)

        Ex, Ey_pred = forecast_fn(model, θ_trained, st, data_obs, u_for_test, timepoints, config["training"]["validation"])
        forecasted = (Ex, Ey_pred)
        perf = eval_forecast(future_true, forecasted)

        push!(models, model)
        push!(trained_params, θ_trained)
        push!(states, st)
        push!(performances, perf)

        ŷ_rmse, ŷ_crps = perf
        @info @sprintf("Fold %d completed: RMSE=%.4e, CRPS=%.4e", fold_idx, ŷ_rmse, ŷ_crps)
    end

    avg_rmse = mean([p[1] for p in performances])
    avg_crps = mean([p[2] for p in performances])

    elapsed = time() - start_time
    @info @sprintf("K-Fold Results: avg RMSE=%.4e, avg CRPS=%.4e", avg_rmse, avg_crps)
    @info "Total training time: $(round(elapsed, digits=2))s ($(round(elapsed/60, digits=2)) min)"

    return models, trained_params, states, performances
end
