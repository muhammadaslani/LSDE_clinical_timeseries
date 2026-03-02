"""
    kl_normal(μ, log_σ²)

Compute the KL divergence between a normal distribution and a standard normal distribution.

Arguments:

  - `μ`: Mean of the normal distribution.
  - `log_σ²`: Log variance of the normal distribution.

returns: 

    - The KL divergence.

"""
function kl_normal(μ, log_σ²)
    kl = 0.5f0 * mean(exp.(log_σ²) .+ μ .^ 2 .- 1 .- log_σ²)
    return kl
end


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

    λ = λ .+ 1f-4  # Add small constant to prevent log(0)
    ll = sum((y * log(λ) - λ - loggamma(y + 1)))

    return ll
end
"""
    poisson_loglikelihood(λ::AbstractArray, y::AbstractArray, mask::AbstractArray{Bool})

Calculate the Poisson log-likelihood of observed counts `y` given rates `λ`.

# Arguments
- `λ::AbstractArray`: Predicted rates (λ > 0)
- `y::AbstractArray`: Observed counts (non-negative integers)
- `mask::AbstractArray{Bool}`: Boolean mask to specify which elements to include in the calculation

# Returns
- `ll::Float32`: The calculated log-likelihood

# Notes
- A small constant (1e-4) is added to λ to prevent log(0)
- NaN or negative values in λ will raise an error
"""
function poisson_loglikelihood(λ::AbstractArray, y::AbstractArray, mask::AbstractArray)
    @assert size(λ) == size(y) "poisson_loglikelihood: Rates and spikes should be of the same shape"
    @assert !any(isnan.(λ)) "poisson_loglikelihood: NaN rate predictions found"
    @assert all(λ .>= 0) "poisson_loglikelihood: Negative rate predictions found"

    λ = λ .+ 1f-4  # Add small constant to prevent log(0)
    ll = sum(@. mask * (y * log(λ) - λ - loggamma(y + 1)))

    return ll
end


"""
    poisson_loglikelihood_multiple_samples(λ::AbstractArray, y::AbstractArray, mask::AbstractArray{Bool}; agg=mean)

Calculate the Poisson log-likelihood across multiple Monte Carlo samples with masking.

# Arguments
- `λ::AbstractArray`: Predicted rates with shape (n_features, n_timepoints, n_samples, n_mc_samples)
- `y::AbstractArray`: Observed counts with shape (n_features, n_timepoints, n_samples)
- `mask::AbstractArray{Bool}`: Boolean mask indicating valid entries
- `agg`: Aggregation function (default: `mean`). Can be `mean` or `sum`

# Returns
- `Float32`: Aggregated log-likelihood across all Monte Carlo samples

# Description
Computes the Poisson log-likelihood for each Monte Carlo sample in the 4th dimension
and returns the aggregated result based on the `agg` parameter.
"""
function poisson_loglikelihood_multiple_samples(λ::AbstractArray, y::AbstractArray, mask::AbstractArray; agg=mean)
    ll = 0.0f0
    for i in eachindex(size(λ, 4))
        ll += poisson_loglikelihood(λ[:, :, :, i], y[:, :, :], mask)
    end

    if agg == mean
        num_valid = sum(mask)
        return ll / num_valid / size(λ, 4)
    elseif agg == sum
        return ll
    else
        error("Unsupported aggregation function. Use `mean` or `sum`.")
    end
end

"""
    poisson_loglikelihood_multiple_samples(λ::AbstractArray, y::AbstractArray; agg=mean)

Calculate the Poisson log-likelihood across multiple Monte Carlo samples without masking.

# Arguments
- `λ::AbstractArray`: Predicted rates with shape (n_features, n_timepoints, n_samples, n_mc_samples)
- `y::AbstractArray`: Observed counts with shape (n_features, n_timepoints, n_samples)
- `agg`: Aggregation function (default: `mean`). Can be `mean` or `sum`

# Returns
- `Float32`: Aggregated log-likelihood across all Monte Carlo samples

# Description
Computes the Poisson log-likelihood for each Monte Carlo sample in the 4th dimension
and returns the aggregated result based on the `agg` parameter.
"""
function poisson_loglikelihood_multiple_samples(λ::AbstractArray, y::AbstractArray; agg=mean)
    ll = 0.0f0
    for i in eachindex(size(λ, 4))
        ll += poisson_loglikelihood(λ[:, :, :, i], y[:, :, :])
    end

    if agg == mean
        num_elements = prod(size(y))
        return ll / num_elements / size(λ, 4)
    elseif agg == sum
        return ll
    else
        error("Unsupported aggregation function. Use `mean` or `sum`.")
    end
end




"""
Negative log‑likelihood for a Poisson with **log‑rates**.

logλ  – model output (same shape as y)
y     – non‑negative integer counts
Returns a scalar loss to **minimise**.
"""
function poisson_nll_lograte(logλ::AbstractArray, y::AbstractArray)
    @assert size(logλ) == size(y)
    @assert all(y .>= 0)
    ll = y .* logλ .- exp.(logλ) .- loggamma.(y .+ 1)   # log‑likelihood
    return -mean(ll)   # negative mean log‑likelihood
end




"""
    normal_loglikelihood(μ, log_σ², y)

Compute the log-likelihood of a normal distribution.

Arguments:

  - `μ`: Mean of the normal distribution.
  - `log_σ²`: Log of variance of the normal distribution.
  - `y`: The observed values.

returns: 

    - The log-likelihood.

"""
function normal_loglikelihood(μ, log_σ², y; ϵ=1e-8)
    # Clamp log_σ² to prevent extreme values
    log_σ² = clamp.(log_σ², -3.0f0, 3.0f0)
    # Compute log-likelihood in a numerically stable way
    # Using mean instead of sum to stabilize loss magnitude across sequences
    ll = -0.5 * mean(log_σ² .+ log(2π) .+ ((y .- μ) .^ 2 ./ exp.(log_σ²) .+ ϵ))
    return -ll
end

"""
    normal_loglikelihood(μ, log_σ², y, mask; ϵ=1e-8)

Masked negative log-likelihood for a Gaussian distribution.
Only the entries where `mask == true` are included in the computation.

The result is normalised by the number of valid entries so its magnitude
is comparable to the unmasked version and stays scale-invariant w.r.t.
the density of observations.

# Arguments
- `μ`      : Predicted means (same shape as `y`)
- `log_σ²` : Predicted log-variances (same shape as `y`)
- `y`      : Observations (same shape as `μ`)
- `mask`   : Boolean mask — `true` where the observation is real, `false` where it is missing/imputed
- `ϵ`      : Small constant for numerical stability (default `1e-8`)

# Returns
- Negative masked Gaussian log-likelihood, as a scalar
"""
function normal_loglikelihood(μ, log_σ², y, mask::AbstractArray{Bool}; ϵ=1e-8)
    log_σ² = clamp.(log_σ², -3.0f0, 3.0f0)
    # Zero out invalid positions before computing NLL terms
    μ_m = μ .* mask
    log_σ²_m = log_σ² .* mask
    y_m = y .* mask
    # Pointwise NLL (invalid positions contribute 0 since all inputs are 0 there)
    nll_terms = 0.5f0 .* (log_σ²_m .+ log(2π) .+ ((y_m .- μ_m) .^ 2 ./ (exp.(log_σ²_m) .+ ϵ)))
    # Average over observed entries only
    n_valid = max(sum(mask), 1)
    return sum(nll_terms .* mask) / n_valid
end



"""
    mse(ŷ, y)

Compute the mean squared error.

Arguments:

  - `ŷ`: Predicted values.
  - `y`: Observed values.

returns: 

    - The mean squared error.

"""
function mse(ŷ, y)
    return mean((ŷ .- y) .^ 2)
end

function mse(ŷ, y, mask::AbstractArray)
    @assert size(ŷ) == size(y) "MSE: Predictions and targets must have the same shape"
    @assert size(ŷ) == size(mask) "MSE: Predictions and mask must have the same shape"
    num_valid = sum(mask)
    return sum((ŷ .* mask .- y .* mask) .^ 2) / num_valid
end



"""
    CrossEntropy_loss(y, ŷ, mask; agg=mean, logits=true, label_smoothing=0.2)

Compute the cross-entropy loss between ground truth labels `y` and predicted labels `ŷ`, applying a mask.

# Arguments
- `y`: Ground truth labels.
- `ŷ`: Predicted labels.
- `mask`: Boolean mask specifying which elements to include in the calculation.
- `agg`: Aggregation function (default: `mean`).
- `logits`: Indicates if `ŷ` contains logits (normalized) (default: `true`).
- `label_smoothing`: Label smoothing factor (default: `0.2`).

# Returns
- The computed cross-entropy loss.

"""
CrossEntropy_Loss(ŷ, y, mask; agg=mean, logits=true, label_smoothing=0.1, epsilon=1e-10) =
    CrossEntropyLoss(; agg=agg, logits=logits, label_smoothing=label_smoothing, epsilon=epsilon)(mask .* ŷ, mask .* y)




#######################################
"""
    bits_per_spike(rates, spikes)

Compute the bits per spike by comparing the Poisson log-likelihood of the rates with the Poisson log-likelihood of the mean spikes. 

Arguments:

  - `rates`: The predicted rates.
  - `spikes`: The observed spikes.

returns: 

    - The bits per spike.

"""
function bits_per_spike(rates, spikes)
    @assert size(rates) == size(spikes) "Rates and spikes must have the same shape"
    rates_ll = poisson_loglikelihood(rates, spikes)
    mean_spikes = mean(spikes, dims=(2, 3))
    null_rates = repeat(mean_spikes, 1, size(spikes, 2), size(spikes, 3))
    null_ll = poisson_loglikelihood(null_rates, spikes)
    spike_sum = sum(spikes)
    bps = (rates_ll / log(2) - null_ll / log(2)) / spike_sum
    return bps
end

"""
    frange_cycle_linear(n_iter, start, stop, n_cycle, ratio)

Generate a linear schedule with cycles.

Arguments:

  - `n_iter`: Number of iterations.
  - `start`: Start value.
  - `stop`: Stop value.
  - `n_cycle`: Number of cycles.
  - `ratio`: Ratio of the linear schedule.

returns: 

    - The linear schedule.

"""
function frange_cycle_linear(n_iter, start::T=0.0f0, stop::T=1.0f0, n_cycle=4, ratio=0.5) where T
    L = ones(n_iter) * stop
    period = n_iter / n_cycle
    step = T((stop - start) / (period * ratio)) # linear schedule

    for c in 0:n_cycle-1
        v, i = start, 1
        while (v ≤ stop) & (Int(round(i + c * period)) < n_iter)
            L[Int(round(i + c * period))] = v
            v += step
            i += 1
        end
    end
    return T.(L)
end
