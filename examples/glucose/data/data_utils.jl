function generate_dataloader(; n_samples=512, batchsize=16, split=(0.5, 0.3), obs_fraction=0.5,
    normalization=true, seed::Union{Int,Nothing}=1234)
    if n_samples <= 512
        U, X, Y, T, covariates = generate_dataset(n_samples=n_samples, seed=seed)

        # Reshape Y from Vector{Vector} to Vector{Matrix} (1×T) for pad_matrices
        Y_mat = [reshape(y, 1, :) for y in Y]

        Y_padded, Masks, timepoints = pad_matrices(Y_mat, T)
        X_padded, _ = pad_matrices(X, T; return_timepoints=false)

        # Irregularize: randomly zero out observations to simulate missing data
        Masks = copy(Masks)
        for i in axes(Y_padded, 3)
            for j in axes(Y_padded, 2)
                if rand() > 0.8  # ~20% irregularity rate
                    Y_padded[:, j, i] .= 0
                    Masks[:, j, i] .= false
                end
            end
        end

        normalization_stats = Dict()
        if normalization
            @info "Data normalized: glucose observations scaled to [0,1]"
            Y_max = maximum(Y_padded)
            Y_padded = Y_padded ./ Y_max
            normalization_stats["Y_stats"] = (max_val=Y_max,)
        else
            @info "Data has not been normalized"
        end

        t_min, t_max = minimum(timepoints), maximum(timepoints)
        normalization_stats["T_stats"] = (min_val=t_min, max_val=t_max)
        timepoints = (timepoints .- t_min) ./ (t_max - t_min)

        # Normalize controls (U) per-channel to [0,1]
        U = cat(U..., dims=3)
        if normalization
            U_max = maximum(abs.(U), dims=(2, 3))  # (2, 1, 1) — per-channel max
            U_max = max.(U_max, 1f-8)              # avoid division by zero
            U = U ./ U_max
            normalization_stats["U_stats"] = (max_vals=dropdims(U_max, dims=(2, 3)),)
            @info "Controls normalized per-channel: meal max=$(U_max[1]), insulin max=$(U_max[2])"
        end

        # Normalize covariates to [0,1]
        covars_min = minimum(covariates, dims=2)
        covars_max = maximum(covariates, dims=2)
        covariates_norm = (covariates .- covars_min) ./ (covars_max .- covars_min .+ 1e-8)
        if normalization
            normalization_stats["Covars_stats"] = (min_vals=covars_min, max_vals=covars_max)
            @info "Covariates normalized to [0,1]"
        end
        covars = repeat(reshape(Float32.(covariates_norm), 6, 1, size(covariates, 2)), 1, size(Y_padded, 2), 1)

        U_obs, U_forecast = split_matrix(U, obs_fraction)
        X_obs, X_forecast = split_matrix(X_padded, obs_fraction)
        Covars_obs, Covars_forecast = split_matrix(covars, obs_fraction)
        Y_obs, Y_forecast = split_matrix(Y_padded, obs_fraction)
        Masks_obs, Masks_forecast = split_matrix(Masks, obs_fraction)
        ts_obs, ts_for = split_matrix(timepoints, obs_fraction)

        data = (U_obs, Covars_obs, X_obs, Y_obs, Masks_obs,
            U_forecast, Covars_forecast, X_forecast, Y_forecast, Masks_forecast)
    else
        @warn "n_samples is too large, using chunked data generation"
        data, train_loader, val_loader, test_loader, dims, ts_obs, ts_for, normalization_stats =
            generate_dataloader_in_chunks(; n_samples=n_samples, batchsize=batchsize, split=split,
                obs_fraction=obs_fraction, normalization=normalization, seed=seed)
    end

    (train_data, val_data, test_data,) = splitobs((data), at=split)
    train_loader = DataLoader(train_data, batchsize=batchsize, shuffle=true)
    val_loader = DataLoader(val_data, batchsize=batchsize, shuffle=true)
    test_loader = DataLoader(test_data, batchsize=batchsize, shuffle=false)

    dims = Dict(
        "obs_dim" => size(data[2], 1) + size(data[4], 1) + size(data[5], 1),
        "input_dim" => size(data[1], 1),
        "state_dim" => size(data[3], 1),
        "output_dim" => size(data[4], 1)
    )

    return map(x -> Float32.(x), data), train_loader, val_loader, test_loader, dims, ts_obs, ts_for, normalization_stats
end

function generate_dataloader_in_chunks(; n_samples=512, batchsize=64, split=(0.5, 0.3),
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
    (train_data, val_data, test_data,) = splitobs((data), at=split)
    train_loader = DataLoader(train_data, batchsize=batchsize, shuffle=true)
    val_loader = DataLoader(val_data, batchsize=batchsize, shuffle=true)
    test_loader = DataLoader(test_data, batchsize=batchsize, shuffle=false)
    dims = Dict(
        "obs_dim" => size(data[2], 1) + size(data[4], 1) + size(data[5], 1),
        "input_dim" => size(data[1], 1),
        "state_dim" => size(data[3], 1),
        "output_dim" => size(data[4], 1)
    )

    return data, train_loader, val_loader, test_loader, dims, ts_obs, ts_for, normalization_stats
end
