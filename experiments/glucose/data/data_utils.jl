function load_dataset(; n_samples=512, obs_fraction=0.5, normalization=true, seed::Union{Int,Nothing}=1234)
    U, X, Y, T, covariates = generate_dataset(n_samples=n_samples, seed=seed)

    # Reshape Y from Vector{Vector} to Vector{Matrix} (1×T) for pad_matrices
    Y_mat = [reshape(y, 1, :) for y in Y]
    Y_padded, Masks, timepoints = pad_matrices(Y_mat, T)
    X_padded, _ = pad_matrices(X, T; return_timepoints=false)

    # Irregularize: randomly zero out ~20% of observations to simulate missing data
    Masks = copy(Masks)
    for i in axes(Y_padded, 3), j in axes(Y_padded, 2)
        if rand() > 0.8
            Y_padded[:, j, i] .= 0
            Masks[:, j, i] .= false
        end
    end

    # Normalize
    normalization_stats = Dict()
    if normalization
        Y_max = maximum(Y_padded)
        Y_padded = Y_padded ./ Y_max
        normalization_stats["Y_stats"] = (max_val=Y_max,)
        @info "Glucose observations scaled to [0,1]"
    end

    # Normalize timepoints to [0,1]
    t_min, t_max = minimum(timepoints), maximum(timepoints)
    normalization_stats["T_stats"] = (min_val=t_min, max_val=t_max)
    timepoints = (timepoints .- t_min) ./ (t_max - t_min)

    # Concatenate and normalize controls (U) per-channel to [0,1]
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
    covars = repeat(reshape(Float32.(covariates_norm), 6, 1, size(covariates, 2)), 1, size(Y_padded, 2), 1)

    # Split into observation and forecast windows
    U_obs, U_forecast = split_matrix(U, obs_fraction)
    X_obs, X_forecast = split_matrix(X_padded, obs_fraction)
    Covars_obs, Covars_forecast = split_matrix(covars, obs_fraction)
    Y_obs, Y_forecast = split_matrix(Y_padded, obs_fraction)
    Masks_obs, Masks_forecast = split_matrix(Masks, obs_fraction)
    timepoints = Tuple(split_matrix(timepoints, obs_fraction))

    data = (U_obs, Covars_obs, X_obs, Y_obs, Masks_obs,
        U_forecast, Covars_forecast, X_forecast, Y_forecast, Masks_forecast)
    data = map(x -> Float32.(x), data)

    dims = Dict(
        "obs_dim" => size(data[2], 1) + size(data[4], 1) + size(data[5], 1),
        "input_dim" => size(data[1], 1),
        "state_dim" => size(data[3], 1),
        "output_dim" => size(data[4], 1)
    )

    return data, dims, timepoints, normalization_stats
end
