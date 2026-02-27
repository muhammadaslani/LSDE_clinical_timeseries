
"""
    smooth_data(input_tensor; window_size=5)

Smooths the input tensor along the second dimension using a moving average filter.

# Arguments
- `input_tensor`: A 3-dimensional array to be smoothed.
- `window_size`: An optional integer specifying the size of the moving window. Default is 5.

# Returns
- A 3-dimensional array of the same size as `input_tensor`, where each element is the mean of the elements within the moving window along the second dimension.
"""

function smooth_data(input_tensor; window_size=5)
    smoothed_tensor = similar(input_tensor)

    for i in axes(input_tensor)[1]
        for k in axes(input_tensor)[3]
            for j in axes(input_tensor)[2]
                start_idx = max(1, j - window_size + 1)
                end_idx = min(size(input_tensor, 2), j + window_size - 1)
                smoothed_tensor[i, j, k] = mean(input_tensor[i, start_idx:end_idx, k])
            end
        end
    end
    return smoothed_tensor
end


"""
    z_normalize(X::AbstractArray{<:Real})

Z-normalizes the input array `X` along the last dimension.

For each element `x` in `X`, the z-normalized value `z` is computed as:
`z = (x - μ) / σ`, where `μ` is the mean and `σ` is the standard deviation
of the elements along the last dimension of `X`.

# Arguments
- `X::AbstractArray{<:Real}`: Input array of real numbers.

# Returns
- `AbstractArray{Float64}`: Z-normalized array with the same dimensions as `X`.

"""

function z_normalize(input_tensor)
    n_features = size(input_tensor, 1)
    normalized_tensor = similar(input_tensor)
    μ_vec = Vector{Float64}(undef, n_features)
    σ_vec = Vector{Float64}(undef, n_features)
    for i in 1:n_features
        μ = mean(input_tensor[i,:,:])
        σ = std(input_tensor[i,:,:])
        σ = σ ≈ 0 ? 1.0 : σ   # avoid divide-by-zero for constant features
        normalized_tensor[i,:,:] = (input_tensor[i,:,:] .- μ) ./ σ
        μ_vec[i] = μ
        σ_vec[i] = σ
    end
    return normalized_tensor, μ_vec, σ_vec
end

"""
    min_max_normalize(input_tensor, target_min=0.0, target_max=1.0)
Min-max normalizes the input tensor along the first dimension.
# Arguments
- `input_tensor`: A 3-dimensional array to be normalized.
- `target_min`: The minimum value of the target range. Default is 0.0.
- `target_max`: The maximum value of the target range. Default is 1.0.     
# Returns
- A 3-dimensional array of the same size as `input_tensor`, where each slice along the first dimension is normalized to the range [target_min, target_max].
"""

function min_max_normalize(input_tensor, target_min=0.0, target_max=1.0)
    normalized_tensor = similar(input_tensor)
    for i in axes(input_tensor)[1]
        # Find the minimum and maximum values for the current slice
        min_val = minimum(input_tensor[i,:,:])
        max_val = maximum(input_tensor[i,:,:])
        
        # Avoid division by zero if all values are the same
        if max_val ≈ min_val
            normalized_tensor[i,:,:] .= target_min
        else
            # Scale to the target range
            range = max_val - min_val
            normalized_tensor[i,:,:] = target_min .+ (target_max - target_min) .* 
                                      (input_tensor[i,:,:] .- min_val) ./ range
        end
    end
    return normalized_tensor
end