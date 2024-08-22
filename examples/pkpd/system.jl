using Pkg, Revise, DifferentialEquations, Random, Distributions, CairoMakie, Lux


function plot_observations(sol, H_obs, y_obs, t_obs)
    fig = Figure(size = (1200, 900))
    # Define axes
    ax1 = Axis(fig[1, 1], title = "Tumor Size (x)", limits = ((1,1000), nothing))
    ax2 = Axis(fig[2, 1], title = "Drug Concentration (c)" , limits = ((1,1000), nothing))
    ax3 = Axis(fig[3, 1], title = "Radiotherapy Dose (d)",  limits = ((1,1000), nothing))
    ax4 = Axis(fig[4, 1], title = "Immune Response (I)",  limits = ((1,1000), nothing))
    ax5 = Axis(fig[5, 1], title = "Overall Health (S)",  limits = ((1,1000), (0,1)))
    ax6 = Axis(fig[6, 1], title = "Observed cancer cell count (y)",  limits = ((1,1000), nothing))
    ax7 = Axis(fig[7, 1], title = "Observed Health Status (H)",  limits = ((1,1000), (0,5)), xlabel= "Time (days)")
    # Plot the solutions on each axis
    lines!(ax1, sol.t, sol[1, :], color = :blue, label = "x(t)")
    lines!(ax2, sol.t, sol[2, :], color = :red, label = "c(t)")
    lines!(ax3, sol.t, sol[3, :], color = :green, label = "d(t)")
    lines!(ax4, sol.t, sol[4, :], color = :purple, label = "I(t)")
    lines!(ax5, sol.t, sol[5, :], color = :orange, label = "S(t)")

    CairoMakie.lines!(ax6, t_obs, y_obs, color = :brown, label = "y(t)")
    CairoMakie.lines!(ax7, t_obs, H_obs, color = :pink, label = "H(t)")

    # Display the figure
    display(fig)
end

function health_to_score(S, λ=3.0)
    # Scale S to be a parameter for Poisson distribution
    scaled_S = S * λ
    poisson_dist = Poisson(scaled_S)
    # Draw a value from the Poisson distribution
    score = rand(poisson_dist)
    # Clamp the score between 1 and 5
    return clamp(score, 0, 5)
end
function generate_observations(sol;sample_rate)
    H_obs = Int[]
    y_obs = Int[]

    y_obs = rand.(Poisson.((sol[1, :])))
    H_obs = health_to_score.(sol[5, :])

    #only sample every 10th observation 
    H_obs = H_obs[1:sample_rate:end]
    y_obs = y_obs[1:sample_rate:end]
    t_obs = sol.t[1:sample_rate:end]
    return H_obs, y_obs, t_obs

end


# Set seed for reproducibility
Random.seed!(1234)

function setup_params()
    ρ = 1e-3       # Tumor growth rate
    K = 100        # Tumor carrying capacity

    # Drug efficacy 
    β_c = 0.3 # Linear effect

    # Radiotherapy effects
    α_r = 0.5 # Linear effect
    β_r = 0.1 # Quadratic effect

    # Immune response
    δ = 0.02      # Reduced immune growth rate
    β_I = 0.1    # Increased drug-induced immune suppression
    α_I = 0.2     # Increased radiotherapy-induced immune suppression
    θ_I = 0.08    #Immune stimulation by tumor
    λ_I = 0.003   #Immune suppression by large tumors
    ω_I = 0.03    #Immune decay rate
    I_max = 0.95   # Max immune response

    # Overall health
    γ_I = 7e-2 # Immune effect on health
    θ_S = 80.0 # Health recovery rate
    λ_S = 2000.0 # Health impact of tumor

    p = [ρ, K, β_c, α_r, β_r, δ, β_I, α_I, θ_I, λ_I, ω_I, I_max, γ_I, θ_S, λ_S]
    return p
end

# Chemotherapy input: Administered every 30 days, stops after 100 days
function u_c(t)
    if t > 1000
        return 0.0
    else
        return (t % 80 < 1) ? 1.0 : 0.0
    end
end

# Radiotherapy input: Administered every 40 days, stops after 100 days
function u_r(t)
    if t > 1000
        return 0.0
    else
        return (t % 200 < 1) ? 1.0 : 0.0
    end
end

function model!(du, u, p, t)
    x, c, d, I, S = u  # Unpack variables
    # Unpack parameters
    ρ, K, β_c, α_r, β_r, δ, β_I, α_I, θ_I, λ_I, ω_I, I_max, γ_I, θ_S, λ_S = p

    # Tumor growth dynamics
    du[1] = (ρ * log(K / (relu(x) + 1e-5)) - β_c * c - (α_r * d + β_r * d^2)) * x

    # Drug concentration dynamics
    du[2] = -0.5 * c + u_c(t)

    # Radiotherapy dynamics
    du[3] = -0.5 * d + u_r(t)

    du[4] = δ * (1 - I / I_max) * I - β_I * c - α_I * d +
    θ_I * (I_max - I) / (1 + λ_I * x) - ω_I * I
    # Overall health dynamics: includes effects from tumor size (x) and immune response (I)
    health_tumor_effect = θ_S * (1 - S) / (1 + λ_S * x)
    health_immune_effect = -γ_I * ((I / I_max) - 1)^2
    du[5] = health_tumor_effect + health_immune_effect

end

# Only apply the diffusion term to the tumor size
diffusion(du, u, p, t) = du[1] = 1e-2*sqrt(u[1]^2)

function condition(u, t, integrator) # Event when condition(u,t,integrator) == 0
    u[5]
end

function affect!(integrator) # Action when condition(u,t,integrator) == 0
    terminate!(integrator)
end
# Initial conditions
x₀ = 20  # Initial tumor size (10% of K)
c₀ = 0.0  # Initial drug concentration (close to zero)
d₀ = 0.0  # Initial radiotherapy dose (close to zero)
I₀ = 0.8  # Initial immune response (80% of I_max)
S₀ = 0.7  # Initial overall health (close to 1)

# Initial conditions vector
u₀ = [x₀, c₀, d₀, I₀, S₀]

# Solve the system
# Time span
tspan = (1.0, 1000.0)  # Simulate for 1500 days
p = setup_params()
prob = SDEProblem(model!, diffusion, u₀, tspan, p)
cb = ContinuousCallback(condition, affect!)
sol = solve(prob, EM(), callback = cb, dt=0.01)
H_obs, y_obs, t_obs = generate_observations(sol, sample_rate=20)
fig = plot_observations(sol, H_obs, y_obs, t_obs)



