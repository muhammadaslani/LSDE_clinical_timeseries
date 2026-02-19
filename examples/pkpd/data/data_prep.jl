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

    # Base parameters with covariate effects
    # Tumor growth rate: affected by age, gender, and tumor type
    ρ::Float64 = abs(rand(rng, Normal(8e-2, 2e-2))) *
                 (1 + 0.002 * (age - 50) / 30) *
                 (gender == 0 ? rand(rng, Normal(1.03, 0.05)) : rand(rng, Normal(0.97, 0.09))) *
                 (tumor_type == "SCLC" ? 1.05 : 0.95)

    # Tumor carrying capacity: affected by gender and tumor type
    K::Float64 = abs(rand(rng, Normal(100.0, 30))) *
                 (gender == 0 ? rand(rng, Normal(1.05, 0.05)) : rand(rng, Normal(0.95, 0.09))) *
                 (tumor_type == "SCLC" ? 0.92 : 1.08)

    # Linear effect of chemotherapy: affected by age, BSA, and tumor type
    β_c::Float64 = abs(rand(rng, Normal(0.1, 0.05))) *
                   (1 - 0.001 * (age - 50)) *
                   (1 / (BSA / 1.7)) *
                   (tumor_type == "SCLC" ? 1.08 : 0.92)

    ω_c::Float64 = 1.0   # Chemotherapy sessions frequency (every X weeks)

    t_half_c::Float64 = abs(rand(rng, Normal(0.1, 0.05))) # Half-life of chemotherapy

    # Linear effect of radiotherapy: affected by age and tumor type
    α_r::Float64 = abs(rand(rng, Normal(0.1, 0.05))) *
                   (1 - 0.001 * (age - 50)) *
                   (tumor_type == "SCLC" ? 1.08 : 0.92)

    # Quadratic effect of radiotherapy: affected by tumor type
    β_r::Float64 = abs(rand(rng, Normal(0.1, 0.05))) *
                   (tumor_type == "SCLC" ? 1.04 : 0.96)

    ω_r::Float64 = 3.0   # Radiotherapy sessions frequency (every X weeks)

    t_half_r::Float64 = abs(rand(rng, Normal(0.1, 0.05))) # Half-life of radiotherapy

    # Reduced immune growth rate: affected by age and BMI
    δ::Float64 = abs(rand(rng, Normal(0.013, 0.005))) *
                 (1 - 0.005 * (age - 50) / 30) *
                 (BMI > 30 ? 0.95 : 1.05)

    # Increased chemotherapy-induced immune suppression: affected by age and BMI
    β_I::Float64 = abs(rand(rng, Normal(0.1, 0.05))) *
                   (1 + 0.001 * (age - 50)) *
                   (BMI > 30 ? 1.05 : 0.95)

    # Increased radiotherapy-induced immune suppression: affected by BMI
    α_I::Float64 = abs(rand(rng, Normal(0.1, 0.05))) *
                   (BMI > 30 ? 1.04 : 0.96)

    # Immune stimulation by tumor: affected by tumor type
    θ_I::Float64 = abs(rand(rng, Normal(0.08, 0.04))) *
                   (tumor_type == "SCLC" ? 0.94 : 1.06)

    # Immune suppression by large tumors: affected by age
    λ_I::Float64 = abs(rand(rng, Normal(0.005, 0.002))) *
                   (1 + 0.0005 * (age - 50))

    # Immune decay rate: affected by age
    ω_I::Float64 = abs(rand(rng, Normal(0.15, 0.05))) *
                   (1 + 0.002 * (age - 50) / 30)

    # Maximum immune response: affected by BMI and age
    I_max::Float64 = abs(rand(rng, Normal(0.95, 0.4))) *
                     (BMI > 30 ? 0.96 : 1.04) *
                     (1 - 0.001 * (age - 50) / 30)

    # Immune effect on health: affected by age
    γ_S::Float64 = abs(rand(rng, Normal(8e-3, 5e-3))) *
                   (1 - 0.002 * (age - 50) / 30)

    # Health recovery rate: affected by age and BMI
    θ_S::Float64 = abs(rand(rng, Normal(100.0, 10))) *
                   (1 - 0.003 * (age - 50) / 30) *
                   (BMI > 30 ? 0.96 : 1.04)

    # Health impact of tumor: affected by gender
    λ_S::Float64 = abs(rand(rng, Normal(200.0, 20))) *
                   (gender == 0 ? 1.04 : 0.96)

    σ_process::Float64 = 1e-1 # Patient-specific noise level
end

"""
    health_to_score(S::Float64)::Int

Convert health value to a discrete score between 0 (best) and 5 (worst).
"""
function health_to_score(S::Float64)::Int
    S = clamp(S, 0.0, 1.0)
    if S <= 0.01
        return 5
    elseif S <= 0.2
        return 4
    elseif S <= 0.4
        return 3
    elseif S <= 0.6
        return 2
    elseif S <= 0.8
        return 1
    else
        return 0
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
        misclass_prob = rand(rng, Uniform(0.02, 0.1))
        H_obs, y_obs, t_obs = generate_observations(sol, sample_rate; rng=rng, misclass_prob=misclass_prob)
        push!(Y, hcat(H_obs, y_obs)')
        push!(T, t_obs)
    end
    return Y, T
end

"""
Chemotherapy input function.
"""
u_c(t::Float64, ω_c::Union{Int64,Float64})::Float64 = (t % (ω_c * 7) < 0.5) && t > 1 ? 1.0 : 0.0

"""
Radiotherapy input function.
"""
u_r(t::Float64, ω_r::Union{Int64,Float64})::Float64 = (t % (ω_r * 7) < 0.5) && t > 1 ? 1.0 : 0.0

"""
    generate_inputs(ω_c, ω_r, tspan, sample_rate)

Generate input matrix for chemotherapy and radiotherapy.
"""
function generate_inputs(ω_c::Int64, ω_r::Int64, tspan::Tuple{Float64,Float64}, sample_rate::Int)::Matrix{Float64}
    t = tspan[1]:1.0:tspan[2]
    uc = [u_c(tt, ω_c) for tt in t][1:sample_rate:end]
    ur = [u_r(tt, ω_r) for tt in t][1:sample_rate:end]
    return hcat(uc, ur)'
end

"""
    model!(dX, X, p, t)

PKPD model differential equations.
"""
function model!(dX::Vector{Float64}, X::Vector{Float64}, p::ModelParameters, t::Float64)
    x, c, d, I, S = X
    dX[1] = (p.ρ * log(p.K / max(x, 1e-5)) - p.β_c * c - (p.α_r * d + p.β_r * d^2)) * x
    dX[2] = -p.t_half_c * c + u_c(t, p.ω_c)
    dX[3] = -p.t_half_r * d + u_r(t, p.ω_r)
    dX[4] = p.δ * (1 - I / p.I_max) * I - p.β_I * c - p.α_I * d +
            p.θ_I * (p.I_max - I) / (1 + p.λ_I * x) - p.ω_I * I
    health_tumor_effect = p.θ_S * (1 - S) / (1 + p.λ_S * x)
    health_immune_effect = -p.γ_S * ((I / p.I_max) - 1)^2
    dX[5] = health_tumor_effect + health_immune_effect
end

"""
    diffusion(dX, X, p, t)

Diffusion term for the stochastic differential equation.
"""
function diffusion(dX::Vector{Float64}, X::Vector{Float64}, p::ModelParameters, t::Float64)
    dX .= p.σ_process .* sqrt.(abs.(X) .+ 1e-5)
end

"""
Termination condition callback.
"""
condition(X::Vector{Float64}, t::Float64, integrator) = X[5] - 0.01

"""
Callback effect for termination.
"""
function affect!(integrator)
    terminate!(integrator)
    integrator.u[5] = 0.01
end

"""
    generate_dataset(; kwargs...)

Generate a dataset of PKPD model simulations.
"""
function generate_dataset(;
    n_samples::Int,
    X₀_mean::Vector{Float64}=[50.0, 0.0, 0.0, 0.8, 0.9],
    X₀_std::Vector{Float64}=[25.0, 0.0, 0.0, 0.3, 0.3],
    tspan::Tuple{Float64,Float64}=(0.0, 365.0),
    sample_rate::Int=7,
    params::ModelParameters=ModelParameters(),
    seed::Union{Int,Nothing}=1234
)
    rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed)

    ω_cs = rand(rng, 1:10, n_samples)
    ω_rs = rand(rng, 1:10, n_samples)
    covariates = zeros(Float64, 5, n_samples)

    @info "Generating inputs"
    U = [generate_inputs(ω_cs[i], ω_rs[i], tspan, sample_rate) for i in 1:n_samples]

    @info "Generating states"
    prob = SDEProblem(model!, diffusion, X₀_mean, tspan, params)
    cb = ContinuousCallback(condition, affect!, save_positions=(false, false))

    function prob_func(prob, i, repeat)
        patient_rng = Random.MersenneTwister(rand(rng, UInt))
        new_params = ModelParameters(; ω_c=ω_cs[i], ω_r=ω_rs[i], rng=patient_rng)
        new_X₀ = [max(rand(patient_rng, Normal(X₀_mean[j], X₀_std[j])), 0.0) for j in eachindex(X₀_mean)]
        new_X₀[5] = min(new_X₀[5], 1.0)
        new_X₀[4] = min(new_X₀[4], 1.0)

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