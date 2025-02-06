
"""
    poisson_loglikelihood(λ::AbstractArray, y::AbstractArray)

Calculate the Poisson log-likelihood of observed counts `y` given rates `λ`.

# Arguments
- `λ::AbstractArray`: Predicted rates (λ > 0)
- `y::AbstractArray`: Observed counts (non-negative integers)

# Returns
- `ll::Float32`: The calculated log-likelihood

# Notes
- A small constant (1e-4) is added to λ to prevent log(0)
- NaN or negative values in λ will raise an error
"""
function poisson_loglikelihood(λ::AbstractArray, y::AbstractArray)
    @assert size(λ) == size(y) "poisson_loglikelihood: Rates and spikes should be of the same shape"
    @assert !any(isnan.(λ)) "poisson_loglikelihood: NaN rate predictions found"
    @assert all(λ .>= 0) "poisson_loglikelihood: Negative rate predictions found"

    λ = λ .+ 1.0f-4  # Add small constant to prevent log(0)
    ll = sum(y .* log.(λ) .- λ .- loggamma.(y .+ 1))

    return ll
end

"""
    poisson_loglikelihood(λ::AbstractArray, y::AbstractArray, mask::AbstractArray{Bool})

Calculate the masked Poisson log-likelihood of observed counts `y` given rates `λ`.

# Arguments
- `λ::AbstractArray`: Predicted rates (λ > 0)
- `y::AbstractArray`: Observed counts (non-negative integers)
- `mask::AbstractArray{Bool}`: Boolean mask to specify which elements to include in the calculation

# Returns
- `ll::Float32`: The calculated log-likelihood

# Notes
- Only the elements where mask is true are included in the calculation
- A small constant (1e-4) is added to λ to prevent log(0)
- NaN or negative values in λ will raise an error
"""
function poisson_loglikelihood(λ::AbstractArray, y::AbstractArray, mask::AbstractArray{Bool})
    @assert size(λ) == size(y) "poisson_loglikelihood: Rates, spikes, and mask should be of the same shape"
    @assert !any(isnan.(λ)) "poisson_loglikelihood: NaN rate predictions found"
    @assert all(λ .>= 0) "poisson_loglikelihood: Negative rate predictions found"

    λ = λ .+ 1.0f-4  # Add small constant to prevent log(0)
    ll = sum(@. mask * (y * log(λ) - λ - loggamma(y + 1)))

    return ll
end


"""
    weighted_bce_loss(y, ŷ; α=0.9, agg=sum)

Calculate the weighted binary cross-entropy loss.

# Arguments
- `y`: Ground truth binary labels (0 or 1)
- `ŷ`: Predicted probabilities (0 < ŷ < 1)
- `α`: Weighting factor for the positive class (default: 0.9)
- `agg`: Aggregation function to apply to the loss (default: sum)

# Returns
- `loss`: The calculated weighted binary cross-entropy loss

# Notes
- A small constant (1e-15) is added to ŷ to prevent log(0)
"""
function weighted_bce_loss(y, ŷ; α=0.9, agg=sum)
    ε = 1e-15  # small constant to avoid log(0)
    ŷ = clamp.(ŷ, ε, 1 - ε)
    loss = -agg(α * y .* log.(ŷ) .+ (1 - α) * (1 .- y) .* log.(1 .- ŷ))
    return loss
end


bce_loss = BinaryCrossEntropyLoss(; agg=sum, label_smoothing=0.3, logits=Val(false));
focal_loss = FocalLoss(; gamma=1.0, dims=1, agg=sum, epsilon=1e-8);

"""
    binary_focal_loss(y_true::Array{T,1}, y_pred::Array{T,1}; gamma::Float64=2.0, alpha::Float64=0.25) -> Float64

Compute the binary focal loss between true labels `y_true` and predicted labels `y_pred`.

# Arguments
- `y_true::Array{T,1}`: Array of true binary labels (0 or 1).
- `y_pred::Array{T,1}`: Array of predicted probabilities for the positive class.
- `gamma::Float64=2.0`: Focusing parameter to adjust the rate at which easy examples are down-weighted.
- `alpha::Float64=0.25`: Balancing parameter to balance the importance of positive/negative examples.

# Returns
- `Float64`: The computed binary focal loss.

"""

function binary_focal_loss(y, ŷ; γ=3.0, agg=sum, ϵ=1e-10, α=0.9)
    ŷϵ = clamp.(ŷ, ϵ, 1 - ϵ)
    loss = -α .* y .* ((1 .- ŷϵ) .^ γ) .* log.(ŷϵ) .- (1 .- y) .* ŷϵ .^ γ .* log.(1 .- ŷϵ)
    return agg(loss)
end


"""
    normalize_to_range(arr::AbstractArray, target_min::Number, target_max::Number) -> AbstractArray

Normalize the elements of the input array `arr` to a specified range [`target_min`, `target_max`].

# Arguments
- `tensor::AbstractArray`: The input array containing numerical values to be normalized.
- `target_min::Number`: The minimum value of the desired output range.
- `target_max::Number`: The maximum value of the desired output range.

# Returns
- normalized_tensor: A new array with the elements of `arr` normalized to the range [`new_min`, `new_max`].

"""

function normalize_to_range(tensor; target_min=0.0f0, target_max=1.0f0)
    normalized_tensor = similar(input_tensor)
    for i in axes(input_tensor)[1]
                mi=min(input_tensor[i,:,:])
                mx=max(input_tensor[i,:,:])
                normalized_tensor[i, :,:] = (input_tensor[i,:,:] .- mi) ./ (mx.-mi) .* (target_max - target_min) .+ target_min
    end
    return normalized_tensor
end

function z_normalize(input_tensor)
    normalized_tensor = similar(input_tensor)
    for i in axes(input_tensor)[1]
                μ=mean(input_tensor[i,:,:])
                σ=std(input_tensor[i,:,:])
                normalized_tensor[i, :,:] = (input_tensor[i,:,:] .- μ) ./ σ
    end
    return normalized_tensor
end


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
    visualize_results(; obs, u, y, masks, model, θ, st, ts, sample_n)

Visualizes the results of the model predictions against the actual values.

# Arguments
- `obs`: Observations used for prediction.
- `u`: Control inputs.
- `y`: Actual values to compare against.
- `masks`: Masks indicating valid data points.
- `model`: The model used for prediction.
- `θ`: Model parameters.
- `st`: Initial state.
- `ts`: Time steps.
- `sample_n`: Sample index to visualize.

# Returns
- A tuple containing the actual values and predicted values for the selected sample.
"""


function visualize_results_2output(; obs, u, y,masks, model, θ, st, ts, sample_n)
    ŷ, _, x₀ = predict(model, obs, u, ts, θ, st, 20)
    ŷ_std = dropdims(std(ŷ, dims=4), dims=4)
    ŷ_mean = dropdims(mean(ŷ, dims=4), dims=4)
    val_indx₁ = findall(masks[1, :, sample_n] .== 1)
    val_indx₂ = findall(masks[2, :, sample_n] .== 1)
    y₁_val = y[1, val_indx₁, :]
    y₂_val = y[2, val_indx₂, :]
    ŷ₁_val = ŷ_mean[1, val_indx₁, :]
    ŷ₂_val = ŷ_mean[2, val_indx₂, :]
    ŷ₁_val_std = ŷ_std[1, val_indx₁, :]
    ŷ₂_val_std = ŷ_std[2, val_indx₂, :]

    ## Upper bound of 95% confidence interval
    ci_upper₁ = ŷ₁_val .+ 1.96 * ŷ₁_val_std  
    ci_lower₁ = ŷ₁_val .- 1.96 * ŷ₁_val_std 
    ci_upper₂ = ŷ₂_val .+ 1.96 * ŷ₂_val_std  
    ci_lower₂ = ŷ₂_val .- 1.96 * ŷ₂_val_std  

    # Plotting
    fig = Figure(size=(1000, 800), fontsize=12)
    ax1 = CairoMakie.Axis(fig[1, 1], xlabel="Time (hours)",ylabel="HR (heart rate)" )
    ax2= CairoMakie.Axis(fig[2, 1], xlabel="Time (hours)", ylabel="MAP (mean arterial blood pressure)")

    lines!(ax1, y₁_val[:,sample_n], label="Actual Value", color=:blue, linewidth=2, linestyle=:dot)
    CairoMakie.scatter!(ax1, y₁_val[:,sample_n], color=:blue)
    lines!(ax1, ŷ₁_val[:,sample_n], label="Predicted Value", color=:red, linewidth=2, linestyle=:dot)
    CairoMakie.scatter!(ax1, ŷ₁_val[:,sample_n], color=:red)
    band!(ax1, 1:length(ŷ₁_val[:,sample_n]), ci_lower₁[:,sample_n], ci_upper₁[:,sample_n], color=(:red, 0.2), label="95% CI")

    lines!(ax2, y₂_val[:,sample_n], label="Actual Value", color=:blue, linewidth=2, linestyle=:dot)
    CairoMakie.scatter!(ax2, y₂_val[:,sample_n], color=:blue)
    lines!(ax2, ŷ₂_val[:,sample_n], label="Predicted Value", color=:red, linewidth=2, linestyle=:dot)
    CairoMakie.scatter!(ax2, ŷ₂_val[:,sample_n], color=:red)
    band!(ax2, 1:length(ŷ₂_val[:,sample_n]), ci_lower₂[:,sample_n], ci_upper₂[:,sample_n], color=(:red, 0.2), label="95% CI")


    axislegend(ax1,ax2,position=:lb, backgroundcolor=:transparent)
    display(fig)
    y_mse₁= mse(y₁_val, ŷ₁_val)/length(y₁_val)
    y_mse₂= mse(y₂_val, ŷ₂_val)/length(y₂_val)
    println("MSE for HR: ", y_mse₁)
    println("MSE for MAP: ", y_mse₂)
    return (y₁_val,y₂_val), (ŷ₁_val, ŷ₂_val)
end

function visualize_results_1output(; obs, u, y,masks, model, θ, st, ts, sample_n)
    ŷ, _, x₀ = predict(model, obs, u, ts, θ, st, 5)
    ŷ_std = dropdims(std(ŷ, dims=4), dims=4)
    ŷ_mean = dropdims(mean(ŷ, dims=4), dims=4)
    val_indx₁ = findall(masks[1, :, sample_n] .== 1)
    y₁_val = y[1, val_indx₁, :]
    ŷ₁_val = ŷ_mean[1, val_indx₁, :]
    ŷ₁_val_std = ŷ_std[1, val_indx₁, :]

    ## Upper bound of 95% confidence interval
    ci_upper₁ = ŷ₁_val .+ 1.96 * ŷ₁_val_std  
    ci_lower₁ = ŷ₁_val .- 1.96 * ŷ₁_val_std 

    # Plotting
    fig = Figure(size=(600, 400), fontsize=12)
    ax1 = CairoMakie.Axis(fig[1, 1], xlabel="Time (hours)", ylabel="Observations")

    lines!(ax1, y₁_val[:,sample_n], label="Observed Value", color=:blue, linewidth=2, linestyle=:dot)
    CairoMakie.scatter!(ax1, y₁_val[:,sample_n], color=:blue)
    lines!(ax1, ŷ₁_val[:,sample_n], label="Predicted Value", color=:red, linewidth=2, linestyle=:dot)
    CairoMakie.scatter!(ax1, ŷ₁_val[:,sample_n], color=:red)
    #band!(ax1, 1:length(ŷ₁_val[:,sample_n]), ci_lower₁[:,sample_n], ci_upper₁[:,sample_n], color=(:red, 0.2), label="95% CI")

    axislegend(ax1,position=:lb, backgroundcolor=:transparent)
    display(fig)
    y_mse₁= mse(y₁_val, ŷ₁_val)/length(y₁_val)
    println("Prediction MSE: ", y_mse₁)
    return (y₁_val), (ŷ₁_val)
end


function visualize_results_1o(; obs, u, y,masks, model, θ, st, ts, sample_n)
    ŷ, _, x₀ = predict(model, obs, u, ts, θ, st, 50)
    μ, σ=ŷ[1], ŷ[2]
    val_indx₁ = findall(masks[1, :, sample_n] .== 1)
    y_val = y[1, val_indx₁, :]
    μ_val = μ[1, val_indx₁, :,:]
    σ_val = σ[1, val_indx₁, :,:]

    ŷ_val= μ_val .+ σ_val .* randn(Float32, size(μ_val))
    ŷ_val_mean = dropdims(mean(ŷ_val, dims=3), dims=3)
    ŷ_val_std = dropdims(std(ŷ_val, dims=3), dims=3)
    ci_lower = ŷ_val_mean .- 1.96 * ŷ_val_std
    ci_upper = ŷ_val_mean .+ 1.96 * ŷ_val_std

    # Plotting
    fig = Figure(size=(600, 400), fontsize=12)
    ax1 = CairoMakie.Axis(fig[1, 1], xlabel="Time (hours)", ylabel="Observations", title="Heart rate prediction")

    lines!(ax1, y_val[:,sample_n], label="Observed Value", color=:blue, linewidth=2, linestyle=:dot)
    CairoMakie.scatter!(ax1, y_val[:,sample_n], color=:blue)
    lines!(ax1, ŷ_val_mean[:,sample_n], label="Predicted Value", color=:red, linewidth=2, linestyle=:dot)
    CairoMakie.scatter!(ax1, ŷ_val_mean[:,sample_n], color=:red)
    #band!(ax1, 1:length(ŷ_val_mean[:,sample_n]), ci_lower[:,sample_n], ci_upper[:,sample_n], color=(:red, 0.2), label="95% CI")

    axislegend(ax1,position=:lb, backgroundcolor=:transparent)
    display(fig)
    y_mse= mse(y_val, ŷ_val_mean)/length(y_val)
    println("Prediction MSE: ", y_mse)
    return (y_val), (ŷ_val_mean)
end


function visualize_results_1oo(; obs, u, y,masks, model, θ, st, ts, sample_n)
    ŷ, _, x₀ = predict(model, obs, u, ts, θ, st, 50)
    μ, σ=dropdims(mean(ŷ[1],dims=4),dims=4),dropdims( mean(ŷ[2], dims=4),dims=4)
    val_indx₁ = findall(masks[1, :, sample_n] .== 1)
    y_val = y[1, val_indx₁, :]
    μ_val = μ[1, val_indx₁, :]
    σ_val = σ[1, val_indx₁, :]

    ŷ_val= μ_val
    ci_lower = ŷ_val .- 1.96 * σ_val
    ci_upper = ŷ_val .+ 1.96 * σ_val

    # Plotting
    fig = Figure(size=(600, 400), fontsize=12)
    ax1 = CairoMakie.Axis(fig[1, 1], xlabel="Time (hours)", ylabel="Observations")

    lines!(ax1, y_val[:,sample_n], label="Observed Value", color=:blue, linewidth=2, linestyle=:dot)
    CairoMakie.scatter!(ax1, y_val[:,sample_n], color=:blue)
    lines!(ax1, ŷ_val[:,sample_n], label="Predicted Value", color=:red, linewidth=2, linestyle=:dot)
    CairoMakie.scatter!(ax1, ŷ_val[:,sample_n], color=:red)
    #band!(ax1, 1:length(ŷ_val[:,sample_n]), ci_lower[:,sample_n], ci_upper[:,sample_n], color=(:red, 0.2), label="95% CI")

    axislegend(ax1,position=:lb, backgroundcolor=:transparent)
    display(fig)
    y_mse= mse(y_val, ŷ_val)/length(y_val)
    println("Prediction MSE: ", y_mse)
    return (y_val), (ŷ_val)
end



function visualize_results_1ooo(; obs, u, y,masks, model, θ, st, ts, sample_n)
    ŷ, _, x₀ = predict(model, obs, u, ts, θ, st, 50)
    μ, σ=ŷ[1], ŷ[2]
    val_indx₁ = findall(masks[1, :, sample_n] .== 1)
    y_val = y[1, val_indx₁, :]
    μ_val = μ[1, val_indx₁, :,:]
    σ_val = σ[1, val_indx₁, :,:]

    dists=Normal.(μ_val,σ_val);
    ŷ_val=zeros(Float32, size(μ_val))
    for i in 1:size(μ_val)[1]
        for j in 1:size(μ_val)[2]
            for k in 1:size(μ_val)[3]
            ŷ_val[i,j,k]=rand(dists[i,j,k])
            end
        end
    end
    ŷ_val_mean = dropdims(mean(ŷ_val, dims=3), dims=3)


    # Plotting
    fig = Figure(size=(600, 400), fontsize=12)
    ax1 = CairoMakie.Axis(fig[1, 1], xlabel="Time (hours)", ylabel="Observations", title="Heart rate prediction")

    lines!(ax1, y_val[:,sample_n], label="Observed Value", color=:blue, linewidth=2, linestyle=:dot)
    CairoMakie.scatter!(ax1, y_val[:,sample_n], color=:blue)
    lines!(ax1, ŷ_val_mean[:,sample_n], label="Predicted Value", color=:red, linewidth=2, linestyle=:dot)
    CairoMakie.scatter!(ax1, ŷ_val_mean[:,sample_n], color=:red)
    #band!(ax1, 1:length(ŷ_val_mean[:,sample_n]), ci_lower[:,sample_n], ci_upper[:,sample_n], color=(:red, 0.2), label="95% CI")

    axislegend(ax1,position=:lb, backgroundcolor=:transparent)
    display(fig)
    y_mse= mse(y_val, ŷ_val_mean)/length(y_val)
    println("Prediction MSE: ", y_mse)
    return (y_val), (ŷ_val_mean)
end