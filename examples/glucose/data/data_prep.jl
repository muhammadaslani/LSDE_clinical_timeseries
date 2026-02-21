# Bergman Minimal Glucose Model (extended with gut absorption compartment)
#
# States (all absolute, not deviation variables):
#   D(t) : gut glucose amount      [mg/dL]  — carbohydrate absorption buffer
#   G(t) : plasma glucose          [mg/dL]  — absolute (basal Gb when fasting)
#   X(t) : remote (interstitial) insulin  [1/min]  — absolute (≈0 at basal)
#   I(t) : plasma insulin          [μU/mL]  — absolute (basal Ib when fasting)
#
# Inputs (external, known):
#   meal_input(t)  : rate of carbohydrate ingestion into gut  [mg/dL/min]
#                    rectangular pulse of 30 min per meal
#   u_I(t)         : exogenous insulin injection rate          [μU/mL/min]
#                    rectangular pulse of 1 min per injection
#
# Dynamics (absolute formulation):
#   dD/dt = -kabs*D + meal_input(t)
#   dG/dt = -(p1 + X)*(G - Gb) + kabs*D      self-suppression acts on deviation from basal
#   dX/dt = -p2*X + p3*(I - Ib)              remote insulin driven by deviation from basal
#   dI/dt =  γ*max(G - Gb, 0) - n*(I - Ib) + u_I(t)/Vi
#
# Observed output:
#   G_obs ~ Normal(G(t), σ_obs)   [mg/dL]   continuous CGM-like measurement

"""
    ModelParameters

Patient-specific parameters for the Bergman Minimal Glucose Model
(extended with gut absorption compartment).
"""
Base.@kwdef struct ModelParameters
    rng::Random.AbstractRNG = Random.GLOBAL_RNG

    # Gut glucose absorption rate constant [1/min] (~1/kabs gives mean absorption delay ~20-40 min)
    kabs::Float64 = abs(rand(rng, Normal(0.035, 0.008)))

    # Glucose effectiveness (self-suppression of glucose) [1/min]
    p1::Float64 = abs(rand(rng, Normal(0.028, 0.005)))

    # Rate constant of remote insulin disappearance [1/min]
    p2::Float64 = abs(rand(rng, Normal(0.028, 0.005)))

    # Insulin-dependent glucose uptake rate constant [1/(min·μU/mL)]
    p3::Float64 = abs(rand(rng, Normal(5e-5, 1e-5)))

    # Plasma insulin first-order decay rate [1/min]
    n::Float64 = abs(rand(rng, Normal(0.090, 0.015)))

    # Pancreatic insulin secretion rate above threshold [μU/mL/min per mg/dL]
    γ::Float64 = abs(rand(rng, Normal(0.006, 0.001)))

    # Insulin distribution volume [dL]
    Vi::Float64 = rand(rng, Uniform(120.0, 160.0))

    # Basal plasma glucose [mg/dL]
    Gb::Float64 = rand(rng, Uniform(80.0, 100.0))

    # Basal plasma insulin [μU/mL]
    Ib::Float64 = rand(rng, Uniform(8.0, 15.0))

    # Meal schedule
    meal_times::Vector{Float64} = Float64[]
    meal_doses::Vector{Float64} = Float64[]     # total carbohydrate dose per meal [mg/dL]

    # Insulin injection schedule
    insulin_times::Vector{Float64} = Float64[]
    insulin_doses::Vector{Float64} = Float64[]  # total insulin per injection [μU/mL·min]

    # Process noise level (SDE diffusion coefficient)
    σ_process::Float64 = 5e-10
end

# ── Input functions ──────────────────────────────────────────────────────────

"""
    meal_input(t, meal_times, meal_doses)

Rate of carbohydrate ingestion into the gut [mg/dL/min].
Each meal is a rectangular pulse of 30 min duration.
"""
function meal_input(t::Float64, meal_times::Vector{Float64}, meal_doses::Vector{Float64})::Float64
    duration = 30.0
    Ra = 0.0
    for (tm, dose) in zip(meal_times, meal_doses)
        if tm <= t < tm + duration
            Ra += dose / duration
        end
    end
    return Ra
end

"""
    insulin_input(t, insulin_times, insulin_doses)

Exogenous insulin injection rate u_I(t) [μU/mL/min].
Each injection is a rectangular pulse of 1 min duration.
"""
function insulin_input(t::Float64, insulin_times::Vector{Float64}, insulin_doses::Vector{Float64})::Float64
    duration = 1.0
    uI = 0.0
    for (ti, dose) in zip(insulin_times, insulin_doses)
        if ti <= t < ti + duration
            uI += dose / duration
        end
    end
    return uI
end

"""
    make_schedule(n_meals, first_meal, meal_interval, bolus_offset, basal_time, tspan)

Derive meal and insulin times for one patient from their sampled schedule parameters.

- `first_meal`    : time of first meal [min]
- `meal_interval` : time between consecutive meals [min]
- `bolus_offset`  : minutes before each meal the bolus is injected [min]
- `basal_time`    : time of the nightly basal injection [min]
"""
function make_schedule(
    n_meals::Int,
    first_meal::Float64,
    meal_interval::Float64,
    bolus_offset::Float64,
    basal_time::Float64,
    tspan::Tuple{Float64,Float64}
)::Tuple{Vector{Float64},Vector{Float64}}
    meal_times  = [first_meal + (k - 1) * meal_interval for k in 1:n_meals]
    meal_times  = Base.filter(t -> t + 30.0 <= tspan[2], meal_times)

    bolus_times   = max.(meal_times .- bolus_offset, 0.0)
    insulin_times = tspan[2] >= basal_time ? vcat(bolus_times, basal_time) : bolus_times

    return meal_times, insulin_times
end

"""
    generate_inputs(meal_times, meal_doses, insulin_times, insulin_doses, tspan, sample_rate)

Build the sampled input matrix (2 × T_obs): row 1 = meal_input, row 2 = u_I.
Insulin pulses (1 min wide) are max-pooled over each sample window to avoid aliasing.
"""
function generate_inputs(
    meal_times::Vector{Float64},
    meal_doses::Vector{Float64},
    insulin_times::Vector{Float64},
    insulin_doses::Vector{Float64},
    tspan::Tuple{Float64,Float64},
    sample_rate::Int
)::Matrix{Float64}
    t_grid  = collect(tspan[1]:1.0:tspan[2])
    Ra      = [meal_input(Float64(tt), meal_times, meal_doses) for tt in t_grid][1:sample_rate:end]
    uI_full = [insulin_input(Float64(tt), insulin_times, insulin_doses) for tt in t_grid]
    n_obs   = length(Ra)
    uI      = [maximum(uI_full[max(1, (k-1)*sample_rate+1):min(end, k*sample_rate)]) for k in 1:n_obs]
    return hcat(Ra, uI)'
end

# ── Model dynamics ───────────────────────────────────────────────────────────

"""
    model!(dX, X, p, t)

Bergman Minimal Model ODEs (absolute state formulation).

  dD/dt = -kabs*D + meal_input(t)
  dG/dt = -(p1 + X)*(G - Gb) + kabs*D
  dX/dt = -p2*X + p3*(I - Ib)
  dI/dt =  γ*max(G - Gb, 0) - n*(I - Ib) + u_I(t)/Vi
"""
function model!(dX::Vector{Float64}, X::Vector{Float64}, p::ModelParameters, t::Float64)
    D, G, Xi, I = X
    uM = meal_input(t, p.meal_times, p.meal_doses)
    uI = insulin_input(t, p.insulin_times, p.insulin_doses)
    dX[1] = -p.kabs * D + uM
    dX[2] = -(p.p1 + Xi) * (G - p.Gb) + p.kabs * D
    dX[3] = -p.p2 * Xi + p.p3 * (I - p.Ib)
    dX[4] =  p.γ * max(G - p.Gb, 0.0) - p.n * (I - p.Ib) + uI / p.Vi
end

"""
    diffusion(dX, X, p, t)

Diffusion term for the stochastic differential equation.
"""
function diffusion(dX::Vector{Float64}, X::Vector{Float64}, p::ModelParameters, t::Float64)
    dX .= p.σ_process .* sqrt.(abs.(X) .+ 1e-5)
end

# ── Observations ─────────────────────────────────────────────────────────────

"""
    generate_observations(sol, sample_rate; rng, σ_obs)

Sample absolute G(t) (state index 2) at `sample_rate` and add Gaussian CGM noise.
Returns (G_obs [mg/dL], t_obs).
"""
function generate_observations(
    sol,
    sample_rate::Int;
    rng::Random.AbstractRNG = Random.GLOBAL_RNG,
    σ_obs::Float64 = 5.0
)::Tuple{Vector{Float64},Vector{Float64}}
    G_true = sol[2, :][1:sample_rate:end]
    G_obs  = G_true .+ rand(rng, Normal(0.0, σ_obs), length(G_true))
    G_obs  = max.(G_obs, 0.0)
    t_obs  = sol.t[1:sample_rate:end]
    return G_obs, t_obs
end

# ── Dataset generation ───────────────────────────────────────────────────────

"""
    generate_dataset(; kwargs...)

Generate a dataset of Bergman Minimal Model simulations with gut absorption,
per-patient variable meal frequency, insulin timing, doses, and kinetic parameters.

Returns
-------
- `U`          : Vector of input matrices  (2 × T_obs)  [meal_input; u_I]
- `X`          : Vector of state arrays    (4 × T_full) [D, G, X, I]
- `Y`          : Vector of observation vectors (T_obs,) absolute G [mg/dL]
- `T`          : Vector of observation time vectors [min]
- `covariates` : Matrix (6 × n_samples) [Gb; Ib; first_meal; meal_interval; bolus_offset; basal_time]
"""
function generate_dataset(;
    n_samples::Int,
    tspan::Tuple{Float64,Float64} = (0.0, 1440.0),  # 720 min = 12 hours
    sample_rate::Int = 10,                             # observe every 5 min (CGM-like)
    n_meals::Int = 6,
    seed::Union{Int,Nothing} = 1234
)
    rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed)

    covariates = zeros(Float64, 6, n_samples)

    # Per-patient schedule parameters
    # first_meal:    60–120 min (1–2 hours after start)
    # meal_interval: 120–180 min (2–3 hours between meals)
    # bolus_offset:  0–15 min (pre-meal bolus injection timing)
    # basal_time:    1200–1380 min (8–11 PM nightly basal injection)
    first_meals    = [rand(rng, Uniform(60.0, 120.0))   for _ in 1:n_samples]
    meal_intervals = [rand(rng, Uniform(120.0, 180.0))  for _ in 1:n_samples]
    bolus_offsets  = [rand(rng, Uniform(0.0,   15.0))   for _ in 1:n_samples]
    basal_times    = [rand(rng, Uniform(1200.0, 1380.0)) for _ in 1:n_samples]

    @info "Generating states"

    # Placeholder SDE problem structure
    mt0, it0 = make_schedule(n_meals, first_meals[1], meal_intervals[1], bolus_offsets[1], basal_times[1], tspan)
    placeholder_params = ModelParameters(;
        meal_times    = mt0,
        meal_doses    = fill(1.0, length(mt0)),
        insulin_times = it0,
        insulin_doses = fill(200.0, length(it0))
    )
    X₀_placeholder = [0.0, placeholder_params.Gb, 0.0, placeholder_params.Ib]
    prob = SDEProblem(model!, diffusion, X₀_placeholder, tspan, placeholder_params)

    # Clamp all states to non-negative after every solver step
    clamp_cb = DiscreteCallback(
        (u, t, integrator) -> true,
        integrator -> integrator.u .= max.(integrator.u, 0.0);
        save_positions = (false, false)
    )

    base_seed = seed === nothing ? rand(UInt) : UInt(seed)

    function prob_func(prob, i, repeat)
        patient_rng = Random.MersenneTwister(base_seed + UInt(i))

        meal_times, insulin_times = make_schedule(
            n_meals, first_meals[i], meal_intervals[i], bolus_offsets[i], basal_times[i], tspan
        )
        n_meals_actual   = length(meal_times)
        n_insulin_actual = length(insulin_times)

        # Meal doses: total gut glucose load [mg/dL]
        meal_doses  = [rand(patient_rng, Uniform(20.0, 60.0)) for _ in 1:n_meals_actual]
        # Bolus insulin [μU/mL·min] — 1-min pulse
        bolus_doses = [rand(patient_rng, Uniform(4000.0, 8000.0)) for _ in 1:n_meals_actual]
        # Basal insulin [μU/mL·min] — larger nightly dose
        basal_dose  = n_insulin_actual > n_meals_actual ?
            [rand(patient_rng, Uniform(8000.0, 16000.0))] : Float64[]
        insulin_doses = vcat(bolus_doses, basal_dose)

        p = ModelParameters(;
            rng           = patient_rng,
            meal_times    = meal_times,
            meal_doses    = meal_doses,
            insulin_times = insulin_times,
            insulin_doses = insulin_doses
        )

        covariates[:, i] .= [p.Gb, p.Ib, first_meals[i], meal_intervals[i], bolus_offsets[i], basal_times[i]]

        X₀ = [0.0, p.Gb, 0.0, p.Ib]
        remake(prob, u0 = X₀, p = p)
    end

    ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
    ensemble_sol = solve(
        ensemble_prob,
        SOSRI(),
        EnsembleThreads();
        callback  = clamp_cb,
        saveat    = 1.0,
        trajectories = n_samples,
        seed      = seed
    )
    X_states = [Array(sol) for sol in ensemble_sol]

    @info "Generating observations"
    U = Vector{Matrix{Float64}}()
    Y = Vector{Vector{Float64}}()
    T = Vector{Vector{Float64}}()

    for (i, sol) in enumerate(ensemble_sol)
        p = sol.prob.p
        push!(U, generate_inputs(p.meal_times, p.meal_doses, p.insulin_times, p.insulin_doses, tspan, sample_rate))

        obs_rng = Random.MersenneTwister(base_seed + UInt(n_samples) + UInt(i))
        σ_obs = rand(obs_rng, Uniform(0.0, 1.0))
        G_obs, t_obs = generate_observations(sol, sample_rate; rng = obs_rng, σ_obs = σ_obs)
        push!(Y, G_obs)
        push!(T, t_obs)
    end

    @info "Dataset generation complete"
    return Array{Float32}.(U), Array{Float32}.(X_states), Array{Float32}.(Y), Array{Float32}.(T), covariates
end

# U, X, Y, T, covariates = generate_dataset(; n_samples = 200)
