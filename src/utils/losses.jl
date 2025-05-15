"""
    kl_normal(μ, σ²)

Compute the KL divergence between a normal distribution and a standard normal distribution.

Arguments:

  - `μ`: Mean of the normal distribution.
  - `σ²`: Variance of the normal distribution.

returns: 

    - The KL divergence.

"""
function kl_normal(μ, σ²)
    kl = 0.5f0 * mean(σ² .+ μ .^ 2 .- 1 .- log.(σ²))
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
function poisson_loglikelihood(λ::AbstractArray, y::AbstractArray,  mask::AbstractArray{Bool})
    @assert size(λ) == size(y) "poisson_loglikelihood: Rates, spikes, and mask should be of the same shape"
    @assert !any(isnan.(λ)) "poisson_loglikelihood: NaN rate predictions found"
    @assert all(λ .>= 0) "poisson_loglikelihood: Negative rate predictions found"
    
    λ = λ .+ 1f-4  # Add small constant to prevent log(0)
    ll = sum(@. mask * (y * log(λ) - λ - loggamma(y + 1)))
    
    return ll
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
function normal_loglikelihood(μ, log_σ², y)
    σ² = exp.(log_σ²)
    ll = -0.5f0 * sum(log.(2π * σ²) + ((y - μ).^2 ./ σ²))
    return -ll
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
    return sum(abs, ŷ .- y)
end

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
    bps = (rates_ll/log(2) - null_ll/log(2)) / spike_sum
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
function frange_cycle_linear(n_iter, start::T=0.0f0, stop::T=1.0f0,  n_cycle=4, ratio=0.5) where T
    L = ones(n_iter) * stop
    period = n_iter/n_cycle
    step = T((stop-start)/(period*ratio)) # linear schedule

    for c in 0:n_cycle-1
        v, i = start, 1
        while (v ≤ stop) & (Int(round(i+c*period)) < n_iter)
            L[Int(round(i+c*period))] = v
            v += step
            i += 1
        end
    end
    return T.(L)
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
CrossEntropy_Loss( ŷ, y, mask; agg=mean, logits=true, label_smoothing=0.1, epsilon=1e-10) =
    CrossEntropyLoss(; agg=agg, logits=logits, label_smoothing=label_smoothing, epsilon=epsilon)(mask .* ŷ, mask .* y)