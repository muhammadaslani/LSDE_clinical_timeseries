function load_dataset(; n_samples=512, obs_fraction=0.5, normalization=true, seed::Union{Int,Nothing}=1234)
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
        Y₂_max = maximum(Y₂_irreg)
        Y₂_irreg = Y₂_irreg ./ Y₂_max
        normalization_stats["Y₂_stats"] = (max_val=Y₂_max,)
        normalization_stats["normalized"] = true
        @info "Cell counts scaled to [0,1]"
    else
        normalization_stats["Y₂_stats"] = (max_val=maximum(Y₂_irreg),)
        normalization_stats["normalized"] = false
    end

    # Normalize controls (U) per-channel to [0,1]
    U = cat(U..., dims=3)
    if normalization
        U_max = max.(maximum(abs.(U), dims=(2, 3)), 1f-8)
        U = U ./ U_max
        normalization_stats["U_stats"] = (max_vals=dropdims(U_max, dims=(2, 3)),)
    end

    # Normalize covariates to [0,1]
    covars_min = minimum(covariates, dims=2)
    covars_max = maximum(covariates, dims=2)
    covariates_norm = (covariates .- covars_min) ./ (covars_max .- covars_min .+ 1e-8)
    if normalization
        normalization_stats["Covars_stats"] = (min_vals=covars_min, max_vals=covars_max)
    end
    covars = repeat(reshape(Float32.(covariates_norm), 5, 1, size(covariates, 2)), 1, size(Y₁_padded, 2), 1)

    # Split into observation and forecast windows
    U_obs, U_forecast = split_matrix(U, obs_fraction)
    X_obs, X_forecast = split_matrix(X_padded, obs_fraction)
    Covars_obs, Covars_forecast = split_matrix(covars, obs_fraction)
    Y₁_obs, Y₁_forecast = split_matrix(Y₁_irreg, obs_fraction)
    Y₂_obs, Y₂_forecast = split_matrix(Y₂_irreg, obs_fraction)
    Masks₁_obs, Masks₁_forecast = split_matrix(Masks₁, obs_fraction)
    Masks₂_obs, Masks₂_forecast = split_matrix(Masks₂, obs_fraction)
    timepoints = Tuple(split_matrix(timepoints, obs_fraction))

    data = (U_obs, Covars_obs, X_obs, Y₁_obs, Y₂_obs, Masks₁_obs, Masks₂_obs,
        U_forecast, Covars_forecast, X_forecast, Y₁_forecast, Y₂_forecast, Masks₁_forecast, Masks₂_forecast)
    data = map(x -> Float32.(x), data)

    dims = Dict(
        "obs_dim" => size(data[2], 1) + size(data[4], 1) + size(data[5], 1),
        "input_dim" => size(data[1], 1),
        "state_dim" => size(data[3], 1),
        "output_dim" => [size(data[4], 1), size(data[5], 1)]
    )

    return data, dims, timepoints, normalization_stats
end
