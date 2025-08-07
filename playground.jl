function npe_timepoint(y_pred::AbstractArray{T,4}, mask::AbstractArray{Bool,3}) where T <: Number
    n_features, n_timepoints, n_samples, n_mc_samples = size(y_pred)
    npe_per_t_s = zeros(T, n_timepoints, n_samples)

        for t in 1:n_timepoints
            for s in 1:n_samples
                if mask[1, t, s] == true
                # Get MC samples for this prediction point
                pred_dist = y_pred[:, t, s, :]
                # Calculate negative entropy over MC samples
                pred_softmax = softmax(pred_dist, dims=1)
                npe_val = -sum(pred_softmax .* log.(pred_softmax .+ 1e-10))
                
                # Store NPE for this feature, time point, and sample
                npe_per_t_s[t, s] = npe_val / n_mc_samples
                end
            end
        end

    return npe_per_t_s
end



x= rand(5, 10, 3, 100); # Example input
mask = rand(Bool, 5, 10, 3); # Example mask
np = npe_timepoint(x, mask) # Call the function to test it
1 .- np./log(5) # Example usage of the output