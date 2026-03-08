function generate_dataloader(; n_samples=512, batchsize=32, split=(0.5, 0.3), obs_fraction=0.5,
    normalization=true, seed::Union{Int,Nothing}=1234)
    if n_samples <= 512
        U, X, Y₁, Y₂, T, covariates = generate_dataset(n_samples=n_samples, seed=seed)
        Y₁_padded, Masks₁, timepoints = pad_matrices(Y₁, T)
        Y₂_padded, Masks₂, _ = pad_matrices(Y₂, T)
        X_padded, _ = pad_matrices(X, T; return_timepoints=false, pad_method=:last)
        Y₁_irreg, Y₂_irreg, Masks₁, Masks₂ = irregularize(Y₁_padded, Y₂_padded, Masks₁, Masks₂, irreg_rate=0.8)

        normalization_stats = Dict()

        # Normalize timepoints to [0,1]
        t_min, t_max = minimum(timepoints), maximum(timepoints)
        normalization_stats["T_stats"] = (min_val=t_min, max_val=t_max)
        timepoints = (timepoints .- t_min) ./ (t_max - t_min)

        if normalization
            @info "Data normalized: cell counts scaled to [0,1]"
            Y₂_max = maximum(Y₂_irreg)
            Y₂_irreg = Y₂_irreg ./ Y₂_max
            normalization_stats["Y₂_stats"] = (max_val=Y₂_max,)
            normalization_stats["normalized"] = true
        else
            normalization_stats["Y₂_stats"] = (max_val=maximum(Y₂_irreg),)
            normalization_stats["normalized"] = false
            @info "Data not normalized: cell counts kept as raw integers"
        end

        # Normalize controls (U) per-channel to [0,1]
        U = cat(U..., dims=3)
        if normalization
            U_max = maximum(abs.(U), dims=(2, 3))  # (n_inputs, 1, 1) — per-channel max
            U_max = max.(U_max, 1f-8)              # avoid division by zero
            U = U ./ U_max
            normalization_stats["U_stats"] = (max_vals=dropdims(U_max, dims=(2, 3)),)
            @info "Controls normalized per-channel"
        end

        # Normalize covariates to [0,1]
        covars_min = minimum(covariates, dims=2)
        covars_max = maximum(covariates, dims=2)
        covariates_norm = (covariates .- covars_min) ./ (covars_max .- covars_min .+ 1e-8)
        if normalization
            normalization_stats["Covars_stats"] = (min_vals=covars_min, max_vals=covars_max)
            @info "Covariates normalized to [0,1]"
        end
        covars = repeat(reshape(Float32.(covariates_norm), 5, 1, size(covariates, 2)), 1, size(Y₁_padded, 2), 1)

        U_obs, U_forecast = split_matrix(U, obs_fraction)
        X_obs, X_forecast = split_matrix(X_padded, obs_fraction)
        Covars_obs, Covars_forecast = split_matrix(covars, obs_fraction)
        Y₁_obs, Y₁_forecast = split_matrix(Y₁_irreg, obs_fraction)
        Y₂_obs, Y₂_forecast = split_matrix(Y₂_irreg, obs_fraction)
        Masks₁_obs, Masks₁_forecast = split_matrix(Masks₁, obs_fraction)
        Masks₂_obs, Masks₂_forecast = split_matrix(Masks₂, obs_fraction)
        ts_obs, ts_for = split_matrix(timepoints, obs_fraction)

        data = (U_obs, Covars_obs, X_obs, Y₁_obs, Y₂_obs, Masks₁_obs, Masks₂_obs,
            U_forecast, Covars_forecast, X_forecast, Y₁_forecast, Y₂_forecast, Masks₁_forecast, Masks₂_forecast)
    else
        @warn "n_samples is too large, using chunked data generation"
        data, train_loader, val_loader, test_loader, dims, ts_obs, ts_for, normalization_stats =
            generate_dataloader_in_chunks(; n_samples=n_samples, batchsize=batchsize, split=split,
                obs_fraction=obs_fraction, normalization=normalization, seed=seed)
    end

    (train_data, val_data, test_data) = splitobs((data), at=split)
    train_loader = DataLoader(train_data, batchsize=batchsize, shuffle=true)
    val_loader = DataLoader(val_data, batchsize=batchsize, shuffle=true)
    test_loader = DataLoader(test_data, batchsize=batchsize, shuffle=false)

    dims = Dict(
        "obs_dim" => size(data[2], 1) + size(data[4], 1) + size(data[5], 1),
        "input_dim" => size(data[1], 1),
        "state_dim" => size(data[3], 1),
        "output_dim" => [size(data[4], 1), size(data[5], 1)]
    )

    return map(x -> Float32.(x), data), train_loader, val_loader, test_loader, dims, ts_obs, ts_for, normalization_stats
end

function generate_dataloader_in_chunks(; n_samples=512, batchsize=32, split=(0.5, 0.3),
    obs_fraction=0.5, chunk_size=256, normalization=true,
    seed::Union{Int,Nothing}=1234)
    n_chunks = ceil(Int, n_samples / chunk_size)
    all_data = []
    ts_obs, ts_for = nothing, nothing
    normalization_stats = nothing
    for i in 1:n_chunks
        @info "Generating and processing chunk $i/$n_chunks"
        current_samples = min(chunk_size, n_samples - (i - 1) * chunk_size)
        current_samples <= 0 && break
        chunk_seed = isnothing(seed) ? nothing : seed + (i - 1)
        data, train_loader, val_loader, test_loader, dims, ts_obs, ts_for, normalization_stats =
            generate_dataloader(n_samples=current_samples, batchsize=batchsize, split=split,
                obs_fraction=obs_fraction, normalization=normalization, seed=chunk_seed)
        push!(all_data, data)
        GC.gc()
    end
    n_components = length(all_data[1])
    data = Tuple(
        cat([chunk[i] for chunk in all_data]..., dims=3) for i in 1:n_components
    )
    (train_data, val_data, test_data) = splitobs((data), at=split)
    train_loader = DataLoader(train_data, batchsize=batchsize, shuffle=true)
    val_loader = DataLoader(val_data, batchsize=batchsize, shuffle=true)
    test_loader = DataLoader(test_data, batchsize=batchsize, shuffle=false)
    dims = Dict(
        "obs_dim" => size(data[2], 1) + size(data[4], 1) + size(data[5], 1),
        "input_dim" => size(data[1], 1),
        "state_dim" => size(data[3], 1),
        "output_dim" => [size(data[4], 1), size(data[5], 1)]
    )

    return data, train_loader, val_loader, test_loader, dims, ts_obs, ts_for, normalization_stats
end
