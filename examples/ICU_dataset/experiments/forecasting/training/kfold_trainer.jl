function load_config(config_path)
    shared_path = joinpath(dirname(config_path), "shared.yml")
    shared = isfile(shared_path) ? get(YAML.load_file(shared_path), "shared", Dict()) : Dict()
    model_cfg = YAML.load_file(config_path)
    return deep_merge(shared, model_cfg)
end

function deep_merge(base::Dict, override::Dict)
    result = copy(base)
    for (k, v) in override
        if haskey(result, k) && isa(result[k], Dict) && isa(v, Dict)
            result[k] = deep_merge(result[k], v)
        else
            result[k] = v
        end
    end
    return result
end

const MODEL_FACTORIES = Dict(
    "lsde"        => create_latentsde,
    "lode"        => create_latentode,
    "latent_lstm" => create_latent_lstm,
    "latent_cde"  => create_latent_cde,
)

function slice_data(data::Tuple, indices)
    return Tuple(x[:, :, indices] for x in data)
end

function kfold_train(data, dims, n_folds, rng, config_path, model_type, timepoints, loss_fn, eval_fn, forecast_fn, viz_fn)
    start_time = time()

    config     = load_config(config_path)
    exp_path   = joinpath(config["experiment"]["path"], config["experiment"]["name"])
    train_cfg  = config["training"]
    model_cfg  = config["model"]
    val_config = merge(get(model_cfg, "validation", Dict()), train_cfg["validation"])
    batch_size = get(train_cfg, "batch_size", 32)

    factory = get(MODEL_FACTORIES, model_type) do
        error("Unsupported model type: $model_type")
    end

    n_samples      = size(data[1], 3)
    sample_indices = shuffle!(rng, collect(1:n_samples))
    fold_size      = div(n_samples, n_folds)

    folds = map(1:n_folds) do i
        start_idx = (i - 1) * fold_size + 1
        end_idx   = i == n_folds ? n_samples : i * fold_size
        test_idx  = sample_indices[start_idx:end_idx]
        remaining = setdiff(sample_indices, test_idx)
        val_size  = round(Int, length(remaining) * 0.2)
        (remaining[val_size+1:end], remaining[1:val_size], test_idx)
    end

    models         = Vector{Any}(undef, n_folds)
    trained_params = Vector{Any}(undef, n_folds)
    states         = Vector{Any}(undef, n_folds)
    performances   = Vector{Any}(undef, n_folds)

    for fold_idx in 1:n_folds
        @info "Training fold $fold_idx/$n_folds"
        train_idx, val_idx, test_idx = folds[fold_idx]

        train_loader = DataLoader(slice_data(data, train_idx), batchsize=batch_size, shuffle=true)
        val_loader   = DataLoader(slice_data(data, val_idx),   batchsize=batch_size, shuffle=false)
        test_data    = slice_data(data, test_idx)

        model, θ, st = factory(model_cfg, dims, rng)

        θ_trained = train(model, θ, st, timepoints, loss_fn, eval_fn, viz_fn,
            train_loader, val_loader, train_cfg, exp_path)

        # Evaluate on test set
        data_obs    = test_data[1:4]
        future_true = test_data[5:8]
        u_for_test  = test_data[5]
        Ex, Ey      = forecast_fn(model, θ_trained, st, data_obs, u_for_test, timepoints, val_config)
        perf        = eval_forecast(future_true, (Ex, Ey))

        models[fold_idx]         = model
        trained_params[fold_idx] = θ_trained
        states[fold_idx]         = st
        performances[fold_idx]   = perf

        rmse, crps = perf
        @info @sprintf("Fold %d completed: RMSE=%s, CRPS=%s",
            fold_idx, string(round.(rmse, digits=4)), string(round.(crps, digits=4)))
    end

    avg_rmse = mean(mean(p[1]) for p in performances)
    avg_crps = mean(mean(p[2]) for p in performances)
    elapsed  = time() - start_time

    @info @sprintf("K-Fold Results: Avg RMSE=%.4e, Avg CRPS=%.4e", avg_rmse, avg_crps)
    @info "Total training time: $(round(elapsed, digits=2))s ($(round(elapsed/60, digits=2)) min)"

    return models, trained_params, states, performances
end
