function generate_dataloader(; n_samples=512, batchsize=64, split=(0.5, 0.3), obs_fraction=0.5, normalization=true)
    if n_samples <= 512
        U, X, Y₁, Y₂, T, covariates = generate_dataset(n_samples=n_samples)
        Y₁_padded, Masks₁, timepoints = pad_matrices(Y₁, T)
        Y₂_padded, Masks₂, _ = pad_matrices(Y₂, T)
        X_padded, _ = pad_matrices(X, T; return_timepoints=false)
        Y₁_irreg, Y₂_irreg, Masks₁, Masks₂ = irregularize(Y₁_padded, Y₂_padded, Masks₁, Masks₂)
        # Store normalization statistics
        normalization_stats = Dict()

        if normalization
            @info "data has been normalized: only cell counts are normalized by scaling to [0,1]"
            Y₂_max = maximum(Y₂_irreg)
            Y₂_irreg = Y₂_irreg ./ Y₂_max
            normalization_stats["Y₂_stats"] = (max_val=Y₂_max,)
        else
            @info "data has not been normalized"
            normalization_stats = nothing
        end

        timepoints = timepoints / 7.0f0 # Normalize timepoints to days
        covars = repeat(reshape(covariates, 5, 1, size(covariates, 2)), 1, size(Y₁_padded)[2], 1)
        U = cat(U..., dims=3)
        U_obs, U_forecast = split_matrix(U, obs_fraction)
        X_obs, X_forecast = split_matrix(X_padded, obs_fraction)
        Covars_obs, Covars_forcast = split_matrix(covars, obs_fraction)
        Y₁_obs, Y₁_forecast = split_matrix(Y₁_irreg, obs_fraction)
        Y₂_obs, Y₂_forecast = split_matrix(Y₂_irreg, obs_fraction)
        Masks₁_obs, Masks₁_forecast = split_matrix(Masks₁, obs_fraction)
        Masks₂_obs, Masks₂_forecast = split_matrix(Masks₂, obs_fraction)
        ts_obs, ts_for = split_matrix(timepoints, obs_fraction)
        data = (U_obs, Covars_obs, X_obs, Y₁_obs, Y₂_obs, Masks₁_obs, Masks₂_obs,
            U_forecast, Covars_forcast, X_forecast, Y₁_forecast, Y₂_forecast, Masks₁_forecast, Masks₂_forecast)

        (train_data, val_data, test_data,) = splitobs((data), at=split)
        train_loader = DataLoader(train_data, batchsize=batchsize, shuffle=true)
        val_loader = DataLoader(val_data, batchsize=batchsize, shuffle=true)
        test_loader = DataLoader(test_data, batchsize=batchsize, shuffle=false)

        dims = Dict(
            "obs_dim" => size(Covars_obs, 1) + size(Y₁_irreg, 1) + size(Y₂_irreg, 1),
            "input_dim" => size(U, 1),
            "state_dim" => size(X_padded, 1),
            "output_dim" => [size(Y₁_irreg, 1), size(Y₂_irreg, 1)]
        )

    else
        @warn "n_samples is too large, using chunked data generation"
        data, train_loader, val_loader, test_loader, dims, ts_obs, ts_for, normalization_stats = generate_dataloader_in_chunks(; n_samples=n_samples, batchsize=batchsize, split=split, obs_fraction=obs_fraction, normalization=normalization)
    end

    return data, train_loader, val_loader, test_loader, dims, ts_obs, ts_for, normalization_stats
end




# Function to generate a data loader in chunks: helpful for large datasets
function generate_dataloader_in_chunks(; n_samples=512, batchsize=64, split=(0.5, 0.3), obs_fraction=0.5, chunk_size=256, normalization=true)
    # Calculate number of chunks needed
    n_chunks = ceil(Int, n_samples / chunk_size)
    all_data = []
    ts_obs, ts_for = nothing, nothing

    # Process each chunk independently
    for i in 1:n_chunks
        @info "Generating and processing chunk $i/$n_chunks"
        # Calculate samples for this chunk
        current_samples = min(chunk_size, n_samples - (i - 1) * chunk_size)
        # Skip if no samples left
        if current_samples <= 0
            break
        end

        # Generate current chunk
        data, train_loader, val_loader, test_loader, dims, ts_obs, ts_for, normalization_stats = generate_dataloader(n_samples=current_samples, batchsize=batchsize, split=split, obs_fraction=obs_fraction, normalization=normalization)
        # Store chunk data
        push!(all_data, data)
        # Force garbage collection to free memory
        GC.gc()
    end
    n_components = length(all_data[1])
    combined = Tuple(
        cat([chunk[i] for chunk in all_data]..., dims=3) for i in 1:n_components
    )

    return combined, train_loader, val_loader, test_loader, dims, ts_obs, ts_for, normalization_stats
end

