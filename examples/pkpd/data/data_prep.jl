# PKPD Model Simulation
"""
    ModelParameters
"""
Base.@kwdef struct ModelParameters
    # Static covariates
    gender::Int = rand(0:1)       # 0 for male, 1 for female
    age::Float64 = rand(20:80)    # Age in years
    weight::Float64 = rand(50:120) # Weight in kg
    height::Float64 = rand(150:190) # Height in cm
    tumor_type::String = rand(["NSCLC", "SCLC"]) # Tumor type

    # Derived values
    BSA::Float64 = sqrt((height * weight) / 3600) # Body surface area
    BMI::Float64 = weight / ((height / 100)^2)    # Body mass index

    # Base parameters with covariate effects
    # Tumor growth rate: affected by age, gender, and tumor type
    ρ::Float64 = abs(rand(Normal(8e-2, 1e-3))) *
                 (1 + 0.002 * (age - 50) / 30) *
                 (gender == 0 ? 1.03 : 0.97) *
                 (tumor_type == "SCLC" ? 1.05 : 0.95)

    # Tumor carrying capacity: affected by gender and tumor type
    K::Float64 = abs(rand(Normal(100.0, 10))) *
                 (gender == 0 ? 1.05 : 0.95) *
                 (tumor_type == "SCLC" ? 0.92 : 1.08)

    # Linear effect of chemotherapy: affected by age, BSA, and tumor type
    β_c::Float64 = abs(rand(Normal(0.1, 0.05))) *
                   (1 - 0.001 * (age - 50)) *
                   (1 / (BSA / 1.7)) *
                   (tumor_type == "SCLC" ? 1.08 : 0.92)

    ω_c::Float64 = 1.0   # Chemotherapy sessions frequency (every X weeks)

    t_half_c::Float64 = abs(rand(Normal(0.1, 0.05))) # Half-life of chemotherapy

    # Linear effect of radiotherapy: affected by age and tumor type
    α_r::Float64 = abs(rand(Normal(0.1, 0.05))) *
                   (1 - 0.001 * (age - 50)) *
                   (tumor_type == "SCLC" ? 1.08 : 0.92)

    # Quadratic effect of radiotherapy: affected by tumor type
    β_r::Float64 = abs(rand(Normal(0.1, 0.05))) *
                   (tumor_type == "SCLC" ? 1.04 : 0.96)

    ω_r::Float64 = 3.0   # Radiotherapy sessions frequency (every X weeks)

    t_half_r::Float64 = abs(rand(Normal(0.1, 0.05))) # Half-life of radiotherapy


    # Reduced immune growth rate: affected by age and BMI
    δ::Float64 = abs(rand(Normal(0.013, 0.005))) *
                 (1 - 0.005 * (age - 50) / 30) *
                 (BMI > 30 ? 0.95 : 1.05)

    # Increased drug-induced immune suppression: affected by age and BMI
    β_I::Float64 = abs(rand(Normal(0.1, 0.05))) *
                   (1 + 0.001 * (age - 50)) *
                   (BMI > 30 ? 1.05 : 0.95)

    # Increased radiotherapy-induced immune suppression: affected by BMI
    α_I::Float64 = abs(rand(Normal(0.1, 0.05))) *
                   (BMI > 30 ? 1.04 : 0.96)

    # Immune stimulation by tumor: affected by tumor type
    θ_I::Float64 = abs(rand(Normal(0.08, 0.04))) *
                   (tumor_type == "SCLC" ? 0.94 : 1.06)

    # Immune suppression by large tumors: affected by age
    λ_I::Float64 = abs(rand(Normal(0.005, 0.002))) *
                   (1 + 0.0005 * (age - 50))

    # Immune decay rate: affected by age
    ω_I::Float64 = abs(rand(Normal(0.15, 0.05))) *
                   (1 + 0.002 * (age - 50) / 30)

    # Maximum immune response: affected by BMI and age
    I_max::Float64 = abs(rand(Normal(0.95, 0.4))) *
                     (BMI > 30 ? 0.96 : 1.04) *
                     (1 - 0.001 * (age - 50) / 30)

    # Immune effect on health: affected by age
    γ_S::Float64 = abs(rand(Normal(8e-3, 5e-3))) *
                   (1 - 0.002 * (age - 50) / 30)

    # Health recovery rate: affected by age and BMI
    θ_S::Float64 = abs(rand(Normal(100.0, 10))) *
                   (1 - 0.003 * (age - 50) / 30) *
                   (BMI > 30 ? 0.96 : 1.04)

    # Health impact of tumor: affected by gender
    λ_S::Float64 = abs(rand(Normal(200.0, 20))) *
                   (gender == 0 ? 1.04 : 0.96)
end


"""
    health_to_score(S::Float64)::Int

Convert health value to a discrete score between 0 (best) and 5 (worst).

# Arguments
- `S::Float64`: Health value in [0.0, 1.0]

# Returns
- `Int`: Discrete score between 0 and 5
"""

function health_to_score(S::Float64)::Int
    S = clamp(S, 0.0, 1.0)
    if S <= 0.01
        return 5
    elseif S <= 0.2 && S > 0.01
        return 4
    elseif S <= 0.4 && S > 0.2
        return 3
    elseif S <= 0.6 && S > 0.4
        return 2
    elseif S <= 0.8 && S > 0.6
        return 1
    elseif S <= 1.0 && S > 0.8
        return 0
    end
end

"""
    generate_observations(sol::RODESolution, sample_rate::Int)::Tuple{Vector{Int}, Vector{Int}, Vector{Float64}}

Generate observations from a solution.

# Arguments
- `sol::RODESolution`: Solution of the differential equation
- `sample_rate::Int`: Rate at which to sample the solution

# Returns
- `Tuple{Vector{Int}, Vector{Int}, Vector{Float64}}`: Health observations, tumor observations, and time points
"""

function generate_observations(sol::RODESolution, sample_rate::Int; noise_std::Float64=0.0)::Tuple{Vector{Int},Vector{Int},Vector{Float64}}
    # Poisson sampling
    y_obs = rand.(Poisson.((sol[1, :])))[1:sample_rate:end]
    noise_std = noise_std .* y_obs
    # Add observation noise
    noisy_counts = y_obs .+ rand.(Normal.(0, noise_std))
    # Round to integer and clip to non-negative values
    y_obs = [max(round(Int, n), 0) for n in noisy_counts]
    H_obs = health_to_score.(sol[5, :])[1:sample_rate:end]
    t_obs = sol.t[1:sample_rate:end]
    return H_obs, y_obs, t_obs
end

"""
    generate_observations(ensemble_sol::EnsembleSolution, sample_rate::Int)::Tuple{Vector{Matrix{Int}}, Vector{Vector{Float64}}}

Generate observations from an ensemble solution.

# Arguments
- `ensemble_sol::EnsembleSolution`: Ensemble solution of the differential equation
- `sample_rate::Int`: Rate at which to sample the solution

# Returns
- `Tuple{Vector{Matrix{Int}}, Vector{Vector{Float64}}}`: Observations and time points for each trajectory
"""
function generate_observations(ensemble_sol::EnsembleSolution, sample_rate::Int)::Tuple{Vector{Matrix{Int}},Vector{Vector{Float64}}}
    Y = Vector{Matrix{Int}}()
    T = Vector{Vector{Float64}}()

    for sol in ensemble_sol
        H_obs, y_obs, t_obs = generate_observations(sol, sample_rate)
        push!(Y, hcat(H_obs, y_obs)')
        push!(T, t_obs)
    end
    return Y, T
end


"""
    u_c(t::Float64, ω_c::Float64)::Float64

Chemotherapy input function.

# Arguments
- `t::Float64`: Time
- `ω_c::Float64`: Chemotherapy frequency

# Returns
- `Float64`: Chemotherapy input (0 or 1)
"""
u_c(t::Float64, ω_c::Union{Int64,Float64})::Float64 = (t % (ω_c * 7) < 0.5) && t > 1 ? 1.0 : 0.0

"""
    u_r(t::Float64, ω_r::Float64)::Float64

Radiotherapy input function.

# Arguments
- `t::Float64`: Time
- `ω_r::Float64`: Radiotherapy frequency

# Returns
- `Float64`: Radiotherapy input (0 or 1)
"""
u_r(t::Float64, ω_r::Union{Int64,Float64})::Float64 = (t % (ω_r * 7) < 0.5) && t > 1 ? 1.0 : 0.0

"""
    generate_inputs(ω_c::Float64, ω_r::Float64, tspan::Tuple{Float64,Float64}, sample_rate::Int)::Matrix{Float64}

Generate input matrix for chemotherapy and radiotherapy.

# Arguments
- `ω_c::Float64`: Chemotherapy frequency
- `ω_r::Float64`: Radiotherapy frequency
- `tspan::Tuple{Float64,Float64}`: Time span
- `sample_rate::Int`: Sampling rate

# Returns
- `Matrix{Float64}`: Input matrix
"""
function generate_inputs(ω_c::Int64, ω_r::Int64, tspan::Tuple{Float64,Float64}, sample_rate::Int)::Matrix{Float64}
    t = tspan[1]:1.0:tspan[2]
    uc = [u_c(t, ω_c) for t in t][1:sample_rate:end]
    ur = [u_r(t, ω_r) for t in t][1:sample_rate:end]
    return hcat(uc, ur)'
end

"""
    model!(dX::Vector{Float64}, X::Vector{Float64}, p::ModelParameters, t::Float64)

PKPD model differential equations.

# Arguments
- `dX::Vector{Float64}`: Derivative vector
- `X::Vector{Float64}`: State vector
- `p::ModelParameters`: Model parameters
- `t::Float64`: Time
"""
function model!(dX::Vector{Float64}, X::Vector{Float64}, p::ModelParameters, t::Float64)
    x, c, d, I, S = X
    dX[1] = (p.ρ * log(p.K / (max(x, 1e-5))) - p.β_c * c - (p.α_r * d + p.β_r * d^2)) * x
    dX[2] = -p.t_half_c * c + u_c(t, p.ω_c)
    dX[3] = -p.t_half_r * d + u_r(t, p.ω_r)
    dX[4] = p.δ * (1 - I / p.I_max) * I - p.β_I * c - p.α_I * d + p.θ_I * (p.I_max - I) / (1 + p.λ_I * x) - p.ω_I * I
    health_tumor_effect = p.θ_S * (1 - S) / (1 + p.λ_S * x)
    health_immune_effect = -p.γ_S * ((I / p.I_max) - 1)^2
    dX[5] = health_tumor_effect + health_immune_effect
end

"""
    diffusion(dX::Vector{Float64}, X::Vector{Float64}, p::ModelParameters, t::Float64)

Diffusion term for the stochastic differential equation.

# Arguments
- `dX::Vector{Float64}`: Noise vector
- `X::Vector{Float64}`: State vector
- `p::ModelParameters`: Model parameters
- `t::Float64`: Time
"""
function diffusion(dX::Vector{Float64}, X::Vector{Float64}, p::ModelParameters, t::Float64)
    dX[1] = 10e-2 * sqrt(X[1]^2)
end

"""
    condition(X::Vector{Float64}, t::Float64, integrator)::Bool

Termination condition for the simulation.

# Arguments
- `X::Vector{Float64}`: State vector
- `t::Float64`: Time
- `integrator`: DifferentialEquations integrator

# Returns
- `Bool`: True if the simulation should terminate
"""
condition(X::Vector{Float64}, t::Float64, integrator) = X[5] - 0.01

"""
    affect!(integrator)

Callback effect for termination.

# Arguments
- `integrator`: DifferentialEquations integrator
"""
function affect!(integrator)
    terminate!(integrator)
    integrator.u[5] = 0.01
end
"""
    generate_dataset(
        n_samples::Int;
        X₀::Vector{Float64} = [30.0, 0.0, 0.0, 0.8, 0.9],
        tspan::Tuple{Float64,Float64} = (0.0, 365.0),
        sample_rate::Int = 7,
        params::ModelParameters = ModelParameters()
    )::Tuple{Vector{Matrix{Float64}}, Vector{Vector{Float64}}, Vector{Matrix{Int}}, Vector{Vector{Float64}}}

Generate a dataset of PKPD model simulations.

# Arguments
- `n_samples::Int`: Number of samples to generate
- `X₀::Vector{Float64}`: Initial conditions (default: [30.0, 0.0, 0.0, 0.8, 0.9])
- `tspan::Tuple{Float64,Float64}`: Time span (default: (0.0, 365.0))
- `sample_rate::Int`: Sampling rate (default: 7), i.e. observations are made every 7 days
- `params::ModelParameters`: Model parameters (default: ModelParameters())

# Returns
- `U::Vector{Matrix{Float64}`: Interventions (chemotherapy and radiotherapy)
- `X::Vector{Vector{Float64}}`: States (tumorsize, chemotherapy, radiotherapy, immune response, health)
- `Y::Vector{Matrix{Int}}`: Observations (health status, cancer cell count)
- `T::Vector{Vector{Float64}}`: Time points for each trajectory in days.
"""
function generate_dataset(;
    n_samples::Int,
    X₀_mean::Vector{Float64}=[50.0, 0.0, 0.0, 0.8, 0.9],
    X₀_std::Vector{Float64}=[10.0, 0.0, 0.0, 0.0, 0.0],
    tspan::Tuple{Float64,Float64}=(0.0, 210.0),
    sample_rate::Int=7,
    params::ModelParameters=ModelParameters()
)

    Random.seed!(1234)

    ω_cs = rand([2, 5, 6, 7], n_samples)
    ω_rs = rand([2, 5, 6, 7], n_samples)
    covariates = zeros(5, n_samples)

    @info "Generating inputs"
    U = [generate_inputs(ω_cs[i], ω_rs[i], tspan, sample_rate) for i in 1:n_samples]

    @info "Generating states"
    prob = SDEProblem(model!, diffusion, X₀_mean, tspan, params)
    cb = ContinuousCallback(condition, affect!, save_positions=(false, false))
    function prob_func(prob, i, repeat)
        new_params = ModelParameters(ω_c=ω_cs[i], ω_r=ω_rs[i])
        new_X₀ = rand.(Normal.(X₀_mean, X₀_std))
        new_X₀ = max.(new_X₀, 0.0) # Ensure initial conditions are non-negative
        new_X₀[5] = min(new_X₀[5], 1.0) # Ensure max health is 1.0
        new_X₀[4] = min(new_X₀[4], 1.0) # Ensure max immune response is 1.0

        covariates[:, i] .= [new_params.gender, new_params.age, new_params.weight, new_params.height, new_params.tumor_type == "SCLC" ? 1 : 0]
        remake(prob, u0=new_X₀, p=new_params)
    end

    ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)
    ensemble_sol = solve(ensemble_prob, SOSRI(), EnsembleThreads(); callback=cb, saveat=1.0, trajectories=n_samples)
    X = [Array(sol) for sol in ensemble_sol]

    @info "Generating observations"
    Y, T = generate_observations(ensemble_sol, sample_rate)

    # One-hot encode health status
    Y₁ = [Array(onehotbatch(y[1, :], Array(0:5))) for y in Y]
    Y₂ = [reshape(y[2, :], 1, :) for y in Y]

    @info "Done"
    # Return dataset but in Float32 
    return Array{Float32}.(U), Array{Float32}.(X), Array{Int}.(Y₁), Array{Int}.(Y₂), Array{Float32}.(T), covariates
end


#U, X, Y, T = generate_dataset(;n_samples=512);
