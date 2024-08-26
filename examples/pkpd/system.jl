using Pkg, Revise, DifferentialEquations, Random, Distributions, CairoMakie, Lux


function plot_observations(sol, H_obs, y_obs, t_obs)

    fig = Figure(size = (1200, 900))

    t_end = maximum(sol.t)
    ax1 = Axis(fig[1, 1], title = "Tumor Size (x)", limits = ((1,t_end), nothing))
    ax2 = Axis(fig[2, 1], title = "Drug Concentration (c)" , limits = ((1,t_end), nothing))
    ax3 = Axis(fig[3, 1], title = "Radiotherapy Dose (d)",  limits = ((1,t_end), nothing))
    ax4 = Axis(fig[4, 1], title = "Immune Response (I)",  limits = ((1,t_end), nothing))
    ax5 = Axis(fig[5, 1], title = "Overall Health (S)",  limits = ((1,t_end), (0,1)))
    ax6 = Axis(fig[6, 1], title = "Observed cancer cell count (y)",  limits = ((1,t_end), nothing))
    ax7 = Axis(fig[7, 1], title = "Observed Health Status (H)",  limits = ((1,t_end), (0,6)), xlabel= "Time (days)")

    lines!(ax1, sol.t, sol[1, :], color = :blue, label = "x(t)")
    lines!(ax2, sol.t, sol[2, :], color = :red, label = "c(t)")
    lines!(ax3, sol.t, sol[3, :], color = :green, label = "d(t)")
    lines!(ax4, sol.t, sol[4, :], color = :purple, label = "I(t)")
    lines!(ax5, sol.t, sol[5, :], color = :orange, label = "S(t)")
    scatter!(ax6, t_obs, y_obs, color = :brown, label = "y(t)")
    scatter!(ax7, t_obs, H_obs, color = :pink, label = "H(t)")

    display(fig)
end

function health_to_score(S, λ=3)
    scaled_S = S * λ
    poisson_dist = Poisson(exp.(scaled_S))
    score = rand(poisson_dist)
    return clamp(score, 1, 5)
end
function generate_observations(sol::RODESolution; sample_rate)
    y_obs = rand.(Poisson.((sol[1, :])))
    H_obs = health_to_score.(sol[5, :])
    H_obs = H_obs[1:sample_rate:end]
    y_obs = y_obs[1:sample_rate:end]
    t_obs = sol.t[1:sample_rate:end]
    return H_obs, y_obs, t_obs
end

function generate_observations(ensemble_sol::EnsembleSolution; sample_rate)
    Y = []
    T = []

    for sol in ensemble_sol
        H_obs, y_obs, t_obs = generate_observations(sol; sample_rate)
        push!(Y, stack([H_obs, y_obs], dims=1))
        push!(T, t_obs)
    end

    return Y, T
end
Random.seed!(1234)

"""
    setup_params(;
        ρ = 8e-3,       # Tumor growth rate
        K = 100,        # Tumor carrying capacity
        β_c = 0.15,     # Linear effect of chemotherapy
        ω_c = 1,        # Chemotherapy sessions frequency (every X weeks)
        α_r = 0.4,      # Linear effect of radiotherapy
        β_r = 0.1,      # Quadratic effect of radiotherapy
        ω_r = 3,        # Radiotherapy sessions frequency (every X weeks)
        δ = 0.023,      # Reduced immune growth rate
        β_I = 0.15,     # Increased drug-induced immune suppression
        α_I = 0.16,     # Increased radiotherapy-induced immune suppression
        θ_I = 0.08,     # Immune stimulation by tumor
        λ_I = 0.002,    # Immune suppression by large tumors
        ω_I = 0.07,     # Immune decay rate
        I_max = 0.95,   # Max immune response
        γ_S = 5e-2,     # Immune effect on health
        θ_S = 40.0,     # Health recovery rate
        λ_S = 500.0     # Health impact of tumor
    )

Sets up the parameters for the PKPD model.

Returns an array of parameters in the order: 
[ω_c, ω_r, ρ, K, β_c, α_r, β_r, δ, β_I, α_I, θ_I, λ_I, ω_I, I_max, γ_S, θ_S, λ_S]
"""
function setup_params(;
    ρ = 8e-3,       # Tumor growth rate
    K = 100,        # Tumor carrying capacity
    β_c = 0.15,     # Linear effect of chemotherapy
    ω_c = 1,        # Chemotherapy sessions frequency (every X weeks)
    α_r = 0.4,      # Linear effect of radiotherapy
    β_r = 0.1,      # Quadratic effect of radiotherapy
    ω_r = 3,        # Radiotherapy sessions frequency (every X weeks)
    δ = 0.023,      # Reduced immune growth rate
    β_I = 0.15,     # Increased drug-induced immune suppression
    α_I = 0.16,     # Increased radiotherapy-induced immune suppression
    θ_I = 0.08,     # Immune stimulation by tumor
    λ_I = 0.002,    # Immune suppression by large tumors
    ω_I = 0.07,     # Immune decay rate
    I_max = 0.95,   # Max immune response
    γ_S = 5e-2,     # Immune effect on health
    θ_S = 40.0,     # Health recovery rate
    λ_S = 500.0     # Health impact of tumor
)
    p = [ω_c, ω_r, ρ, K, β_c, α_r, β_r, δ, β_I, α_I, θ_I, λ_I, ω_I, I_max, γ_S, θ_S, λ_S]
    return p
end

u_c(t, ω_c) = (t % (ω_c*7) < 1) && t>1 ? 1.0 : 0.0
u_r(t, ω_r) = (t % (ω_r*7) < 1) && t>1 ? 1.0 : 0.0

function generate_inputs(ω_c, ω_r, tspan; sample_rate)
    uc = [u_c(t, ω_c) for t in tspan][1:sample_rate:end]
    ur = [u_r(t, ω_r) for t in tspan][1:sample_rate:end]
    return stack([uc, ur], dims=1) 
end

function model!(dX, X, p, t)
    x, c, d, I, S = X  
    ω_c, ω_r, ρ, K, β_c, α_r, β_r, δ, β_I, α_I, θ_I, λ_I, ω_I, I_max, γ_S, θ_S, λ_S = p
    dX[1] = (ρ * log(K / (relu(x) + 1e-5)) - β_c * c - (α_r * d + β_r * d^2)) * x
    dX[2] = -0.5 * c + u_c(t, ω_c)
    dX[3] = -0.5 * d + u_r(t, ω_r)
    dX[4] = δ * (1 - I / I_max) * I - β_I * c - α_I * d + θ_I * (I_max - I) / (1 + λ_I * x) - ω_I * I
    health_tumor_effect = θ_S * (1 - S) / (1 + λ_S * x)
    health_immune_effect = -γ_S * ((I / I_max) - 1)^2
    dX[5] = health_tumor_effect + health_immune_effect
end

diffusion(dX, X, p, t) = dX[1] = 1e-2*sqrt(X[1]^2)

function condition(X, t, integrator) 
    X[5] 
end

function affect!(integrator)
    terminate!(integrator)
end

# Initial conditions
x₀ = 30.0  
c₀ = 0.0 
d₀ = 0.0  
I₀ = 0.8  
S₀ = 0.9 

# Initial conditions vector
X₀ = [x₀, c₀, d₀, I₀, S₀]

# Time span
tspan = (0.0, 365.0)
p = setup_params()
prob = SDEProblem(model!, diffusion, X₀, tspan, p)
cb = ContinuousCallback(condition, affect!)
sol = solve(prob, SOSRI(), callback = cb, saveat=1.0)
H_obs, y_obs, t_obs = generate_observations(sol, sample_rate=7)
fig = plot_observations(sol, H_obs, y_obs, t_obs)


function generate_dataset(X₀, tspan, sample_rate, n_samples)
    p = setup_params()
    ω_cs = rand([2, 3, 4, 5, 6], n_samples)
    ω_rs = rand([2, 3, 4, 5, 6], n_samples)
    prob = SDEProblem(model!, diffusion, X₀, tspan, p)
    cb = ContinuousCallback(condition, affect!)

    function prob_func(prob, i, repeat)
        remake(prob, u0=X₀.+[rand(-10:10),0,0,0,0], p = (ω_cs[i], ω_rs[i], p[3:end]...))
    end

    ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
    ensemble_sol = solve(ensemble_prob, SOSRI(),  EnsembleThreads(); callback = cb, saveat=1.0, trajectories=n_samples)
    X = ensemble_sol.u
    Y, T = generate_observations(ensemble_sol; sample_rate) 
    U = collect.(generate_inputs(ω_cs[i], ω_rs[i], ensemble_sol[i].t; sample_rate) for i in 1:n_samples)
    return U, X, Y, T
end


U, X, Y, T = generate_dataset(X₀, tspan, 7, 512);
fig = Figure()
ax = Axis(fig[1, 1])
hist!(ax, length.(T), bins=50)
display(fig)