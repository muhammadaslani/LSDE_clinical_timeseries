# PKPD Model Simulation
"""
    ModelParameters
"""
Base.@kwdef struct ModelParameters
    rng::Random.AbstractRNG = Random.GLOBAL_RNG

    # Static covariates
    gender::Int = rand(rng, 0:1)        # 0 for male, 1 for female
    age::Float64 = rand(rng, 18:95)     # Age in years
    weight::Float64 = rand(rng, 50:120) # Weight in kg
    height::Float64 = rand(rng, 150:190) # Height in cm
    tumor_type::String = rand(rng, ["NSCLC", "SCLC"]) # Tumor type

    # Derived values
    BSA::Float64 = sqrt((height * weight) / 3600) # Body surface area
    BMI::Float64 = weight / ((height / 100)^2)    # Body mass index

    # Base parameters — LogNormal distributions (standard in population PK/PD)
    # LogNormal(log(median), σ_log): σ_log ≈ 0.08 gives ~8% CV

    # Tumor growth rate: affected by age, gender, tumor type
    ρ::Float64 = rand(rng, LogNormal(log(0.015), 0.08)) *
                 (1 + 0.001 * (age - 50) / 30) *
                 (gender == 0 ? 1.01 : 0.99) *
                 (tumor_type == "SCLC" ? 1.02 : 0.98)

    # Tumor carrying capacity: affected by gender and tumor type
    K::Float64 = rand(rng, LogNormal(log(100.0), 0.08)) *
                 (gender == 0 ? 1.01 : 0.99) *
                 (tumor_type == "SCLC" ? 0.98 : 1.02)

    # Chemotherapy efficacy: affected by age, BSA, tumor type
    β_c::Float64 = rand(rng, LogNormal(log(0.4), 0.08)) *
                   (1 - 0.0003 * (age - 50)) *
                   (1 / (BSA / 1.7)) *
                   (tumor_type == "SCLC" ? 1.02 : 0.98)

    ω_c::Float64 = 1.0   # Chemotherapy sessions frequency (every X weeks)

    λ_c::Float64 = rand(rng, LogNormal(log(0.7), 0.05))  # Chemo decay rate

    # Radiotherapy linear effect: affected by age and tumor type
    α_r::Float64 = rand(rng, LogNormal(log(1.2), 0.08)) *
                   (1 - 0.0003 * (age - 50)) *
                   (tumor_type == "SCLC" ? 1.02 : 0.98)

    # Radiotherapy quadratic effect: affected by tumor type
    β_r::Float64 = rand(rng, LogNormal(log(0.5), 0.08)) *
                   (tumor_type == "SCLC" ? 1.01 : 0.99)

    ω_r::Float64 = 1.0   # Radiotherapy sessions frequency (every X days)

    λ_r::Float64 = rand(rng, LogNormal(log(4.0), 0.05))  # Radio decay rate (DNA repair)

    # Immune growth rate: affected by age and BMI
    δ::Float64 = rand(rng, LogNormal(log(0.05), 0.08)) *
                 (1 - 0.0005 * (age - 50) / 30) *
                 (BMI > 30 ? 0.99 : 1.01)

    # Chemo-induced immune suppression: affected by age and BMI
    β_I::Float64 = rand(rng, LogNormal(log(0.05), 0.08)) *
                   (1 + 0.0003 * (age - 50)) *
                   (BMI > 30 ? 1.01 : 0.99)

    # Radio-induced immune suppression: affected by BMI
    α_I::Float64 = rand(rng, LogNormal(log(0.03), 0.08)) *
                   (BMI > 30 ? 1.01 : 0.99)

    # Immune stimulation by tumor: affected by tumor type
    θ_I::Float64 = rand(rng, LogNormal(log(0.1), 0.08)) *
                   (tumor_type == "SCLC" ? 0.98 : 1.02)

    # Immune suppression by large tumors: affected by age
    λ_I::Float64 = rand(rng, LogNormal(log(0.01), 0.08)) *
                   (1 + 0.0002 * (age - 50))

    # Immune decay rate: affected by age
    ω_I::Float64 = rand(rng, LogNormal(log(0.1), 0.08)) *
                   (1 + 0.0005 * (age - 50) / 30)

    # Maximum immune response: affected by BMI and age
    I_max::Float64 = rand(rng, LogNormal(log(1.0), 0.05)) *
                     (BMI > 30 ? 0.99 : 1.01) *
                     (1 - 0.0003 * (age - 50) / 30)

    # Immune effect on health: affected by age
    γ_S::Float64 = rand(rng, LogNormal(log(0.02), 0.08)) *
                   (1 - 0.0005 * (age - 50) / 30)

    # Health recovery rate: affected by age and BMI
    θ_S::Float64 = rand(rng, LogNormal(log(0.5), 0.08)) *
                   (1 - 0.0005 * (age - 50) / 30) *
                   (BMI > 30 ? 0.99 : 1.01)

    # Health impact of tumor: affected by gender
    λ_S::Float64 = rand(rng, LogNormal(log(0.5), 0.08)) *
                   (gender == 0 ? 1.01 : 0.99)

    # Tumor burden effect on health
    η_x::Float64 = rand(rng, LogNormal(log(0.5), 0.08))

    # Chemo toxicity on health
    η_c::Float64 = rand(rng, LogNormal(log(0.5), 0.08))

    # Radio toxicity on health
    η_r::Float64 = rand(rng, LogNormal(log(0.3), 0.08))

    σ_tumor::Float64 = 0.02    # Tumor growth noise (low)
    σ_immune::Float64 = 0.02   # Immune response noise (low)
    σ_health::Float64 = 0.02   # Health status noise (low)
end

"""
    health_to_score(S::Float64)::Int

Convert health value to a discrete score between 0 (best) and 5 (worst).
"""
function health_to_score(S::Float64)::Int
    S = clamp(S, 0.0, 1.0)
    if S <= 0.05
        return 5  # terminal (very rare)
    elseif S <= 0.25
        return 4  # very severe (rare)
    elseif S <= 0.45
        return 3  # severe (less common)
    elseif S <= 0.65
        return 2  # moderate
    elseif S <= 0.85
        return 1  # mildly impaired
    else
        return 0  # good health
    end
end

"""
    add_health_noise(H_true; rng, misclass_prob)

Add categorical observation noise to health scores.
"""
function add_health_noise(H_true::Vector{Int}; rng::Random.AbstractRNG, misclass_prob::Float64)::Vector{Int}
    H_obs = copy(H_true)
    for i in eachindex(H_obs)
        if rand(rng) < misclass_prob
            shift = rand(rng, (-1, 1))
            H_obs[i] = clamp(H_obs[i] + shift, 0, 5)
        end
    end
    return H_obs
end

"""
    generate_observations(sol, sample_rate; rng, misclass_prob)

Generate noisy observations from a single solution.
"""
function generate_observations(
    sol::RODESolution,
    sample_rate::Int;
    rng::Random.AbstractRNG=Random.GLOBAL_RNG,
    misclass_prob::Float64=0.05
)::Tuple{Vector{Int},Vector{Int},Vector{Float64}}
    tumor_counts = max.(sol[1, :], 0.0)[1:sample_rate:end]
    y_obs = [rand(rng, Poisson(λ)) for λ in tumor_counts]
    H_true = health_to_score.(sol[5, :])[1:sample_rate:end]
    H_obs = add_health_noise(H_true; rng=rng, misclass_prob=misclass_prob)
    t_obs = sol.t[1:sample_rate:end]
    return H_obs, y_obs, t_obs
end

"""
    generate_observations(ensemble_sol, sample_rate; rng)

Generate observations for every trajectory in an ensemble.
"""
function generate_observations(
    ensemble_sol::EnsembleSolution,
    sample_rate::Int;
    rng::Random.AbstractRNG=Random.GLOBAL_RNG
)::Tuple{Vector{Matrix{Int}},Vector{Vector{Float64}}}
    Y = Vector{Matrix{Int}}()
    T = Vector{Vector{Float64}}()
    for sol in ensemble_sol
        misclass_prob = 0.0  # disabled for now
        H_obs, y_obs, t_obs = generate_observations(sol, sample_rate; rng=rng, misclass_prob=misclass_prob)
        push!(Y, hcat(H_obs, y_obs)')
        push!(T, t_obs)
    end
    return Y, T
end

"""
Chemotherapy input function.
"""
u_c(t::Float64, ω_c::Union{Int64,Float64})::Float64 = (t % (ω_c * 7) < 0.1) && t > 1 ? 1.0 : 0.0

"""
Radiotherapy input function.
"""
u_r(t::Float64, ω_r::Union{Int64,Float64})::Float64 = (t % ω_r < 0.1) && t > 1 ? 1.0 : 0.0

"""
    generate_inputs(ω_c, ω_r, tspan, sample_rate)

Generate input matrix for chemotherapy and radiotherapy.
Treatment inputs are aggregated over each sampling window so no treatment session is missed.
"""
function generate_inputs(ω_c::Int64, ω_r::Int64, tspan::Tuple{Float64,Float64}, sample_rate::Int)::Matrix{Float64}
    t = tspan[1]:1.0:tspan[2]
    # Compute daily treatment signals
    uc_daily = [u_c(tt, ω_c) for tt in t]
    ur_daily = [u_r(tt, ω_r) for tt in t]
    # Aggregate: 1 if any treatment occurred in each sampling window
    n_obs = length(1:sample_rate:length(t))
    uc = [maximum(uc_daily[max(1, (i - 1) * sample_rate + 1):min(end, i * sample_rate)]) for i in 1:n_obs]
    ur = [maximum(ur_daily[max(1, (i - 1) * sample_rate + 1):min(end, i * sample_rate)]) for i in 1:n_obs]
    return hcat(uc, ur)'
end

"""
    model!(dX, X, p, t)

PKPD model differential equations.
"""
function model!(dX::Vector{Float64}, X::Vector{Float64}, p::ModelParameters, t::Float64)
    x, c, d, I, S = X
    x_safe = max(x, 0.0)
    dX[1] = (p.ρ * (1 - x_safe / p.K) - p.β_c * c - (p.α_r * d + p.β_r * d^2)) * x_safe
    dX[2] = -p.λ_c * c + u_c(t, p.ω_c)
    dX[3] = -p.λ_r * d + u_r(t, p.ω_r)
    dX[4] = p.δ * (1 - I / p.I_max) * I - p.β_I * c - p.α_I * d +
            p.θ_I * (p.I_max - I) / (1 + p.λ_I * x) - p.ω_I * I
    health_tumor_effect = p.θ_S * (1 - S) / (1 + p.λ_S * x)
    health_tumor_burden = -p.η_x * (x_safe / p.K) * S           # large tumors directly degrade health
    health_immune_effect = -p.γ_S * ((I / p.I_max) - 1)^2
    health_chemo_toxicity = -p.η_c * p.β_c * c * S               # chemo degrades health
    health_radio_toxicity = -p.η_r * p.α_r * d * S               # radio degrades health
    dX[5] = health_tumor_effect + health_tumor_burden + health_immune_effect + health_chemo_toxicity + health_radio_toxicity
end

"""
    diffusion(dX, X, p, t)

Diffusion term for the stochastic differential equation.
"""
function diffusion(dX::Vector{Float64}, X::Vector{Float64}, p::ModelParameters, t::Float64)
    x, c, d, I, S = X
    dX[1] = p.σ_tumor * sqrt(max(x, 0.0) + 1e-5)   # tumor: moderate uncertainty
    dX[2] = 0.0                                       # chemo PK: deterministic
    dX[3] = 0.0                                       # radio PK: deterministic
    dX[4] = p.σ_immune * sqrt(abs(I) + 1e-5)          # immune: moderate uncertainty
    dX[5] = p.σ_health * sqrt(abs(S) + 1e-5)          # health: low uncertainty
end

"""
Termination condition: health drops below threshold (death).
"""
condition_death(X::Vector{Float64}, t::Float64, integrator) = X[5] - 0.1

"""
Termination condition: tumor drops below threshold (remission/cure).
"""
condition_remission(X::Vector{Float64}, t::Float64, integrator) = X[1] - 0.5

"""
Callback effect for termination.
"""
function affect_death!(integrator)
    terminate!(integrator)
    integrator.u[5] = 0.1
end

function affect_remission!(integrator)
    terminate!(integrator)
    integrator.u[1] = 0.5
end

"""
    generate_dataset(; kwargs...)

Generate a dataset of PKPD model simulations.
"""
function generate_dataset(;
    n_samples::Int,
    X₀_mean::Vector{Float64}=[20.0, 0.0, 0.0, 0.8, 0.9],
    X₀_std::Vector{Float64}=[2.0, 0.0, 0.0, 0.2, 0.2],
    tspan::Tuple{Float64,Float64}=(0.0, 90.0),
    sample_rate::Int=3,
    params::ModelParameters=ModelParameters(),
    seed::Union{Int,Nothing}=1234
)
    rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed)

    ω_cs = rand(rng, 2:4, n_samples)
    ω_rs = rand(rng, 2:4, n_samples)
    covariates = zeros(Float64, 5, n_samples)

    @info "Generating inputs"
    U = [generate_inputs(ω_cs[i], ω_rs[i], tspan, sample_rate) for i in 1:n_samples]

    @info "Generating states"
    prob = SDEProblem(model!, diffusion, X₀_mean, tspan, params)
    cb_death = ContinuousCallback(condition_death, affect_death!, save_positions=(false, false))
    cb_remission = ContinuousCallback(condition_remission, affect_remission!, save_positions=(false, false))
    cb = CallbackSet(cb_death, cb_remission)

    function prob_func(prob, i, repeat)
        patient_rng = Random.MersenneTwister(rand(rng, UInt))
        new_params = ModelParameters(; ω_c=ω_cs[i], ω_r=ω_rs[i], rng=patient_rng)
        new_X₀ = [
            max(rand(patient_rng, Normal(X₀_mean[1], X₀_std[1])), 12.0),  # tumor: min size 12
            0.0,                                                            # chemo: no drug at start
            0.0,                                                            # radio: no drug at start
            rand(patient_rng, Beta(20, 5)),                                 # immune: mean ≈ 0.8, very tight
            rand(patient_rng, Beta(15, 3))                                  # health: mean ≈ 0.83, very tight
        ]

        covariates[:, i] .= [
            new_params.gender,
            new_params.age,
            new_params.weight,
            new_params.height,
            new_params.tumor_type == "SCLC" ? 1.0 : 0.0
        ]
        remake(prob, u0=new_X₀, p=new_params)
    end

    ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)
    ensemble_sol = solve(
        ensemble_prob,
        SOSRI(),
        EnsembleThreads();
        callback=cb,
        saveat=1.0,
        trajectories=n_samples,
        seed=seed
    )
    X = [Array(sol) for sol in ensemble_sol]

    @info "Generating observations"
    Y, T = generate_observations(ensemble_sol, sample_rate; rng=rng)

    # One-hot encode health status
    Y₁ = [Array(onehotbatch(y[1, :], 0:5)) for y in Y]
    Y₂ = [reshape(y[2, :], 1, :) for y in Y]

    @info "Dataset generation complete"
    return Array{Float32}.(U), Array{Float32}.(X), Array{Int}.(Y₁), Array{Int}.(Y₂), Array{Float32}.(T), covariates
end

# U, X, Y₁, Y₂, T, covariates = generate_dataset(; n_samples = 512)