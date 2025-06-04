function generate_dataloader(; n_samples=512, batchsize=64, split=(0.5,0.3), obs_fraction=0.5)
    U, X, Y₁, Y₂, T, covariates = generate_dataset(n_samples=n_samples);
    Y₁_padded, Masks₁, timepoints = pad_matrices(Y₁, T)
    Y₂_padded, Masks₂, _ = pad_matrices(Y₂, T)
    X_padded, _ = pad_matrices(X, T; return_timepoints=false)
    Y₁_irreg, Y₂_irreg, Masks₁, Masks₂ = irregularize(Y₁_padded,Y₂_padded, Masks₁, Masks₂)
    timepoints = timepoints/7.0f0

    covars=repeat(reshape(covariates,5,1,size(covariates,2)),1,size(Y₁_padded)[2],1)
    U = cat(U..., dims=3)
    U_obs, U_forecast=split_matrix(U, obs_fraction)
    X_obs, X_forecast=split_matrix(X_padded, obs_fraction)
    Covars_obs, Covars_forcast=split_matrix(covars, obs_fraction)
    Y₁_obs, Y₁_forecast=split_matrix(Y₁_irreg, obs_fraction)
    Y₂_obs, Y₂_forecast=split_matrix(Y₂_irreg, obs_fraction)
    Masks₁_obs, Masks₁_forecast=split_matrix(Masks₁, obs_fraction)
    Masks₂_obs, Masks₂_forecast=split_matrix(Masks₂, obs_fraction)
    timepoints_obs, timepoints_forecast= split_matrix(timepoints, obs_fraction)
    data_obs= (U_obs, Covars_obs, X_obs, Y₁_obs, Y₂_obs, Masks₁_obs, Masks₂_obs)
    data_forecast= (U_forecast, Covars_forcast, X_forecast, Y₁_forecast, Y₂_forecast, Masks₁_forecast, Masks₂_forecast)
    data= (U_obs, Covars_obs, X_obs, Y₁_obs, Y₂_obs, Masks₁_obs, Masks₂_obs, 
           U_forecast, Covars_forcast, X_forecast, Y₁_forecast, Y₂_forecast, Masks₁_forecast, Masks₂_forecast)

    (train_data, val_data, test_data,) = splitobs((data), at=split)
    train_loader = DataLoader(train_data, batchsize=batchsize, shuffle=true)
    val_loader = DataLoader(val_data, batchsize=batchsize, shuffle=true)
    test_loader = DataLoader(test_data, batchsize=batchsize, shuffle=false)

    dims = Dict(
        "obs_dim" => size(Covars_obs,1)+ size(Y₁_irreg, 1)+ size(Y₂_irreg, 1),
        "input_dim" => size(U, 1),
        "state_dim" => size(X_padded, 1),
        "output_dim" => [size(Y₁_irreg, 1), size(Y₂_irreg, 1)]
    )
    return data, train_loader, val_loader, test_loader, dims, timepoints_obs, timepoints_forecast
end

function generate_dataloader_in_chunks(; n_samples=512, batchsize=64, split=(0.5,0.3), obs_fraction=0.5, chunk_size=500)
    # Calculate number of chunks needed
    n_chunks = ceil(Int, n_samples / chunk_size)
    
    # Initialize arrays to store processed data from each chunk
    all_U_padded = []
    all_X_padded = []
    all_Y₁_padded = []
    all_Y₂_padded = []
    all_Y₁_irreg = []
    all_Y₂_irreg = []
    all_Masks₁ = []
    all_Masks₂ = []
    tpoints = nothing
    all_covariates = []
    
    @info "Generating dataset in $n_chunks chunks of size $chunk_size"
    
    # Process each chunk independently
    for i in 1:n_chunks
        @info "Generating and processing chunk $i/$n_chunks"
        # Calculate samples for this chunk
        current_samples = min(chunk_size, n_samples - (i-1)*chunk_size)
        
        # Skip if no samples left
        if current_samples <= 0
            break
        end
        
        # Generate current chunk
        U_chunk, X_chunk, Y₁_chunk, Y₂_chunk, T_chunk, covariates_chunk = generate_dataset(n_samples=current_samples)
        
        # Process this chunk fully
        Y₁_padded, Masks₁, timepoints = pad_matrices(Y₁_chunk, T_chunk)
        Y₂_padded, Masks₂ = pad_matrices(Y₂_chunk, T_chunk; return_timepoints=false)
        X_padded, _ = pad_matrices(X_chunk, T_chunk; return_timepoints=false)
        Y₁_irreg, Y₂_irreg, Masks₁_irreg, Masks₂_irreg = irregularize(Y₁_padded, Y₂_padded, Masks₁, Masks₂)
        
        # Normalize timepoints for this chunk
        tpoints = timepoints ./ (7.0f0 * 52.0f0)
        
        # Prepare U for this chunk (assuming U is a list of tensors)
        U_padded = cat(U_chunk..., dims=3)
        
        # Store processed chunk data
        push!(all_U_padded, U_padded)
        push!(all_X_padded, X_padded)
        push!(all_Y₁_irreg, Y₁_irreg)
        push!(all_Y₂_irreg, Y₂_irreg)
        push!(all_Masks₁, Masks₁_irreg)
        push!(all_Masks₂, Masks₂_irreg)
        push!(all_covariates, covariates_chunk)
        timepoints = timepoints
        # Force garbage collection to free memory
        GC.gc()
    end
    
    @info "Combining processed chunks"
    
    # Create covariates matrices for each chunk before combining
    all_covars = []
    for i in 1:length(all_covariates)
        covars = repeat(reshape(all_covariates[i], 5, 1, size(all_covariates[i], 2)), 
                       1, size(all_Y₁_irreg[i])[2], 1)
        push!(all_covars, covars)
    end
    
    # Combine all processed chunks
    # Note: We need to make sure the dimensions match before concatenation
    # This assumes all chunks have the same time dimension after padding
    U = cat(all_U_padded..., dims=3)
    X_padded = cat(all_X_padded..., dims=3)
    Y₁_irreg = cat(all_Y₁_irreg..., dims=3)
    Y₂_irreg = cat(all_Y₂_irreg..., dims=3)
    Masks₁ = cat(all_Masks₁..., dims=3)
    Masks₂ = cat(all_Masks₂..., dims=3)
    covars = cat(all_covars..., dims=3)
    
    @info "Splitting into observation and forecast portions"
    
    # Split into observation and forecast portions
    U_obs, U_forcast = split_matrix(U, obs_fraction)
    X_obs, X_forcast = split_matrix(X_padded, obs_fraction)
    Covars_obs, Covars_forcast = split_matrix(covars, obs_fraction)
    Y₁_obs, Y₁_forcast = split_matrix(Y₁_irreg, obs_fraction)
    Y₂_obs, Y₂_forcast = split_matrix(Y₂_irreg, obs_fraction)
    Masks₁_obs, Masks₁_forcast = split_matrix(Masks₁, obs_fraction)
    Masks₂_obs, Masks₂_forcast = split_matrix(Masks₂, obs_fraction)
    timepoints_obs, timepoints_forecast = split_matrix(tpoints, obs_fraction)

    @info "Creating data loaders"
    
    # Package data
    data_obs = (U_obs, X_obs, Covars_obs, Y₁_obs, Y₂_obs, Masks₁_obs, Masks₂_obs)
    data_forecast = (U_forcast, X_forcast, Covars_forcast, Y₁_forcast, Y₂_forcast, Masks₁_forcast, Masks₂_forcast)
    
    # Split into train/val/test
    (train_data, val_data, test_data) = splitobs((data_obs, data_forecast), at=split)
    
    # Create data loaders
    train_loader = DataLoader(train_data, batchsize=batchsize, shuffle=true)
    val_loader = DataLoader(val_data, batchsize=batchsize, shuffle=true)
    test_loader = DataLoader(test_data, batchsize=batchsize, shuffle=false)

    # Store dimensions
    dims = Dict(
        "obs_dim" => [size(covars, 1), size(Y₁_irreg, 1), size(Y₂_irreg, 1)],
        "input_dim" => size(U, 1),
        "state_dim" => size(X_padded, 1),
        "output_dim" => [size(Y₁_irreg, 1), size(Y₂_irreg, 1)]
    )
    
    @info "Data generation complete"
    
    return train_loader, val_loader, test_loader, dims, timepoints_obs, timepoints_forecast
end