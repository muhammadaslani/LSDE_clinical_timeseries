"""
    SDE(drift, drift_aug, diffusion, solver; kwargs...)

Constructs an SDE model.

Arguments:

  - `drift`: The drift of the generative SDE. 
  - `drift_aug`: The drift of the augmented SDE.
  - `diffusion`: The shared diffusion offunction (de::ODE)(x::AbstractArray, u::Union{Nothing, AbstractArray}, ts::AbstractArray, p::ComponentVector, st::NamedTuple)
    u_cont(t) = interp!(ts, u, t, Val(:linear))
    dxdt(x, p, t) = dxdt_u(de.vector_field, x, u_cont(t), t, p.vector_field, st.vector_field)
    ff = ODEFunction{false}(dxdt; tgrad = basic_tgrad)
    prob = ODEProblem{false}(ff, x, (ts[1], ts[end]), p)
    return solve(prob, de.solver; sensealg = InterpolatingAdjoint(; autojacvec = ZygoteVJP()), de.kwargs...), st
endEs.
  - `solver': The nummerical solver used to solve the SDE.
  - `kwargs`: Additional keyword arguments to pass to the solver.
"""
struct SDE <: AbstractLuxContainerLayer{(:drift, :drift_aug, :diffusion)}
    drift
    drift_aug
    diffusion
    solver
    kwargs
end

function SDE(drift, drift_aug, diffusion, solver; kwargs...)
    return SDE(drift, drift_aug, diffusion, solver, kwargs)
end

function LuxCore.parameterlength(m::SDE)
    return LuxCore.parameterlength(m.drift) + LuxCore.parameterlength(m.drift_aug) + LuxCore.parameterlength(m.diffusion)
end


"""
    (de::SDE)(x::AbstractArray, u::AbstractArray, c::AbstractArray, ts::StepRangeLen, p::ComponentVector, st::NamedTuple)

The forward pass of the joint SDE. 
Used for fitting the parameters of the model to data


Arguments:

  - `x`: The initial hidden state.
  - `u`: The control input.
  - `c`: The context.
  - `ts`: The time steps.
  - `p`: The parameters.
  - `st`: The state.

returns: 
    - The solution of the SDE.
    - The state of the model.
"""
function (de::SDE)(x::AbstractArray, u::Union{Nothing,AbstractArray}, c::AbstractArray, ts::AbstractArray, p::ComponentVector, st::NamedTuple)
    #TBD to fix the interpolation

    u_cont1(t) = interp!(ts, u, t, Val(:linear))
    u_cont2(t) = interp!(ts, u, t, Val(:linear))
    c_cont(t) = interp!(ts, c, t, Val(:linear))

    function μ_augmented(x, p, t)
        c_t = c_cont(t)
        u_t = u_cont1(t)
        return de.drift_aug((x, u_t, c_t), p, st.drift_aug)[1]
    end

    function μ_generative(x, p, t)
        u_t = u_cont2(t)
        return de.drift((x, u_t), p, st.drift)[1]
    end

    function σ_shared(x, p, t)
        u_t = u_cont2(t)
        return de.diffusion(x, p, st.diffusion)[1]
    end

    function μ(x, p, t)
        x_ = x[1:end-1, :]
        f = μ_augmented(x_, p.drift_aug, t)
        h = μ_generative(x_, p.drift, t)
        g = σ_shared(x_, p.diffusion, t)
        s = (h .- f) ./ g
        f_logqp = 0.5f0 .* sum(s .^ 2, dims=1)
        return vcat(f, f_logqp)
    end

    function σ(x, p, t)
        x_ = x[1:end-1, :]
        g = σ_shared(x_, p.diffusion, t)
        g_logqp = CRC.@ignore_derivatives fill!(similar(x_, 1, size(x_)[end]), 0.0f0)
        return vcat(g, g_logqp)
    end

    ff = SDEFunction{false}(μ, σ)
    prob = SDEProblem{false}(ff, x, (ts[1], ts[end]), p)
    return solve(prob, de.solver; u0=x, p, saveat=ts, sensealg=TrackerAdjoint(), de.kwargs...), st
end

"""
    sample_generative(de::SDE, init_map, solver, px₀, u, ts, p, st, n_samples, dev; kwargs...)

Generates trajectories from the generative/prior SDE model.
Used for prediction and generation. 

Arguments:

  - `de`: The SDE to sample from.
  - `init_map`: The initial conditions map 
  - `solver`: The nummerical solver used to solve the SDE.
  - `px₀`: The distribution of the initial condition (mean and vaiance)
  - `u`: The control inputs.
  - `ts`: Array of time points at which to sample the trajectories.
  - `p`: The parameters.
  - `st`: The state.
  - `n_samples`: The number of samples to generate.
  - `dev`: The device on which to perform the computations.
  - `kwargs`: Additional keyword arguments to pass to the solver.

returns: 
    - The sampled trajectories.
"""
function sample_generative(de::SDE, init_map, solver, px₀, u, ts, p, st, n_samples, dev; kwargs...)
    tspan = (ts[1], ts[end])
    u_cont(t) = interp!(ts, u, t, Val(:linear))
    x₀ = init_map(sample_rp(px₀), p.init_map, st.init_map)[1]
    μ(x, p, t) = de.drift((x, u_cont(t)), p.drift, st.dynamics.drift)[1]
    σ(x, p, t) = de.diffusion(x, p.diffusion, st.dynamics.diffusion)[1]

    ff = SDEFunction{false}(μ, σ, tgrad=basic_tgrad)
    prob = SDEProblem{false}(ff, x₀, tspan, p.dynamics)

    function prob_func(prob, i, repeat)
        remake(prob, u0=init_map(sample_rp(px₀), p.init_map, st.init_map)[1])
    end

    ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)
    ensemble_sol = solve(ensemble_prob, solver, EnsembleThreads(); trajectories=n_samples, saveat=ts, kwargs...) |> dev
    x = permutedims((ensemble_sol), (1, 3, 2, 4))
    return x

end


"""
    sample_augmented(de::SDE, init_map, solver, px₀, u, c, ts, p, st, n_samples, dev; kwargs...)

Generates trajectories from the augmented SDE model.
Used for filtering and smoothing.

Arguments :

  - `de`: The SDE to sample from.
  - `init_map`: The initial conditions map 
  - `solver`: The nummerical solver used to solve the SDE.
  - `px₀`: The distribution of the initial condition (mean and vaiance)
  - `u`: The control input.
  - `c`: The context.
  - `ts`: Array of time points at which to sample the trajectories.
  - `p`: The parameters.
  - `st`: The state.
  - `n_samples`: The number of samples to generate.
  - `dev`: The device on which to perform the computations.
  - `kwargs`: Additional keyword arguments to pass to the solver.

"""
function sample_augmented(de::SDE, init_map, solver, px₀, u, c, ts, p, st, n_samples, dev; kwargs...)
    tspan = (ts[1], ts[end])
    u_cont(t) = interp!(ts, u, t, Val(:linear))
    c_cont(t) = interp!(ts, c, t, Val(:linear))

    x₀ = init_map(sample_rp(px₀), p.init_map, st.init_map)[1]

    μ(x, p, t) = de.drift_aug(vcat(x, c_cont(t), u_cont(t)), p.drift_aug, st.dynamics.drift_aug)[1]
    σ(x, p, t) = de.diffusion(x, p.diffusion, st.dynamics.diffusion)[1]

    ff = SDEFunction{false}(μ, σ, tgrad=basic_tgrad)
    prob = SDEProblem{false}(ff, x₀, tspan, p.dynamics)

    function prob_func(prob, i, repeat)
        remake(prob, u0=init_map(sample_rp(px₀), p.init_map, st.init_map)[1])
    end

    ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)
    ensemble_sol = solve(ensemble_prob, solver, EnsembleThreads(); trajectories=n_samples, saveat=ts, kwargs...) |> dev
    x = permutedims((ensemble_sol), (1, 3, 2, 4))
    return x
end


##############################################################################

"""
    ODE(vector_field, solver; kwargs...)

Constructs an ODE model.

Arguments:

  - `vector_field`: The vector field of the ODE. 
  - `solver': The nummerical solver used to solve the ODE.
  - `kwargs`: Additional keyword arguments to pass to the solver.

"""



struct ODE{VF} <: UDE
    vector_field::VF
    solver
    kwargs
end

function ODE(vector_field, solver; kwargs...)
    println("Creating an ODE")
    return ODE(vector_field, solver, kwargs)
end

"""
    (de::ODE)(x::AbstractArray, u::Union{Nothing, AbstractArray}, ts::AbstractArray, p::ComponentVector, st::NamedTuple)

The forward pass of the ODE.


Arguments:

  - `x`: The initial hidden state.
  - `u`: The control input.
  - `ts`: The time steps.
  - `p`: The parameters.
  - `st`: The state.

returns: 
    - The solution of the ODE.
    - The state of the model.

"""
function (de::ODE)(x::AbstractArray, u::Union{Nothing,AbstractArray}, ts::AbstractArray, p::ComponentArray, st::NamedTuple)
    u_cont(t) = interp!(ts, u, t, Val(:linear))
    dxdt(x, p, t) = dxdt_u(de.vector_field, x, u_cont(t), t, p.vector_field, st.vector_field)
    ff = ODEFunction{false}(dxdt; tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff, x, (ts[1], ts[end]), p)
    return solve(prob, de.solver; u0=x, p, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()), saveat=ts, de.kwargs...), st
end

"""
    sample_dynamics(de::ODE, x̂₀, u, ts, p, st, n_samples)

Samples trajectories from the ODE model.

Arguments:

  - `de`: The ODE model to sample from.
  - `x̂₀`: The initial hidden state.
  - `u`: Inputs for the input encoder. Can be `Nothing` or an array.
  - `ts`: Array of time points at which to sample the trajectories.
  - `p`: The parameters.
  - `st`: The state.
  - `n_samples`: The number of samples to generate.

returns: 
    - The sampled trajectories.
    - The state of the model.

"""

function sample_dynamics(de::ODE, x̂₀, u, ts, p, st, n_samples)
    u_cont(t) = interp!(ts, u, t, Val(:linear))
    x₀ = sample_rp(x̂₀)
    dxdt(x, p, t) = dxdt_u(de.vector_field, x, u_cont(t), t, p.vector_field, st.vector_field)
    ff = ODEFunction{false}(dxdt; tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff, x₀, (ts[1], ts[end]), p)

    function prob_func(prob, i, repeat)
        remake(prob, u0=sample_rp(x̂₀))
    end
    ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)
    ensemble_sol = solve(ensemble_prob, de.solver, EnsembleThreads(); trajectories=n_samples, saveat=ts, de.kwargs...)
    x = permutedims(Array(ensemble_sol), (1, 3, 2, 4))
    return x
end


function dxdt_u(model::Lux.AbstractLuxLayer, x, u, t, p, st)
    output, _ = model((x, u), p, st)
    return output
end

function dxdt_u(model::Lux.AbstractLuxLayer, x, u::Nothing, t, p, st)
    output, _ = model(x, p, st)
    return output
end

# Specialize the helper for DynamicalSystemLayer
function dxdt_u(model::DynamicalSystem, x, u, t, p, st)
    return model(x, u, t, p, st)
end

function dxdt_u(model::DynamicalSystem, x, u::Nothing, t, p, st)
    return model(x, nothing, t, p, st)
end




###########################################################################################
"""
    LSTM(vector_field; kwargs...)

Constructs an LSTM model.

Arguments:

  - `vector_field`: The vector field of the LSTM. 

"""


struct LSTM <: UDE
    vector_field
end

function LSTM(latent_dim, ctrl_dim)
    vector_field = LSTMCell(latent_dim + ctrl_dim => latent_dim)
    return LSTM(vector_field)
end


"""
    (model::LSTM)(x::AbstractArray, u::Union{Nothing, AbstractArray}, ts::AbstractArray, p::ComponentVector, st::NamedTuple)

The forward pass of the LSTM.


Arguments:

  - `x`: The initial hidden state.
  - `u`: The control input.
  - `ts`: The time steps.
  - `p`: The parameters.
  - `st`: The state.

returns: 
    - The predictions of the LSTM.
    - The state of the model.

"""
function (model::LSTM)(x₀::AbstractArray, u::Union{Nothing,AbstractArray}, ts::AbstractArray, ps::ComponentArray, st::NamedTuple)
    h = x₀
    c = zeros(eltype(x₀), size(x₀))
    ps = ps.vector_field
    st = st.vector_field
    output = zeros(eltype(x₀), size(x₀))
    outputs = [
        begin
            u_t = vcat(output, u[:, t, :])
            (output, (h, c)), st = model.vector_field((u_t, (h, c)), ps, st)
            output
        end for t in 1:length(ts)
    ]

    return stack(outputs, dims=2), (vector_field=st,)
end



"""
    sample_dynamics(model::LSTM, x̂₀, u, ts, p, st, n_samples)
Samples trajectories from the LSTM model.

Arguments:

  - `de`: The LSTM model to sample from.
  - `x̂₀`: The initial hidden state.
  - `u`: Inputs for the input encoder. Can be `Nothing` or an array.
  - `ts`: Array of time points at which to sample the trajectories.
  - `p`: The parameters.
  - `st`: The state.
  - `n_samples`: The number of samples to generate.

returns: 
    - The sampled trajectories.
    - The state of the model.

"""


function sample_dynamics(dynamics::LSTM, x̂₀, u, ts, ps, st, n_samples)

    x = []
    # Generate multiple samples
    for sample_idx in 1:n_samples
        x₀ = sample_rp(x̂₀)
        x_, st = dynamics(x₀, u, ts, ps, st)
        push!(x, x_)
    end
    return stack(x, dims=4)
end


###########################################################################################
# CDE Dynamics — generic, like ODE and SDE
###########################################################################################

"""
    CDE(vector_field)

A generic Controlled Differential Equation dynamics component,
following the same pattern as `ODE` and `SDE`.

Takes `(x₀, u, ts, ps, st)` where `u` is any input signal.
Internally interpolates `u` via `interp!` (same as ODE/SDE),
computes `dX/dt`, and integrates the CDE:

    dz/dt = F_θ(z) · dX/dt

# Fields

- `vector_field`: CDEField that maps z → (latent_dim, path_dim) matrix.
"""
struct CDE <: AbstractLuxContainerLayer{(:vector_field,)}
    vector_field  # CDEField: z → (latent_dim, path_dim) matrix
    solver
    kwargs
end

function CDE(vector_field, solver=Tsit5(); kwargs...)
    return CDE(vector_field, solver, kwargs)
end


"""
    (de::CDE)(x₀, u, ts, ps, st)

Forward pass of the CDE dynamics.

Arguments:

- `x₀`: Initial latent state `(latent_dim, B)`.
- `u`: Input signal `(input_dim, T, B)` — controls, observations, etc.
- `ts`: Time points `(T,)`.
- `ps`: Parameters.
- `st`: States.

Returns:

- `z`: Latent states `(latent_dim, T, B)`.
- `st`: Updated states.
"""
function (de::CDE)(x₀::AbstractArray{<:Real,2}, u::AbstractArray{<:Real,3},
    ts::AbstractVector, ps::ComponentArray, st::NamedTuple)
    T = length(ts)
    B = size(x₀, 2)

    # Append time channel to input signal → control path X(t)
    time_channel = reshape(Float32.(ts), 1, T, 1) .* ones(Float32, 1, 1, B)
    X = cat(u, time_channel; dims=1)   # (input_dim+1, T, B)

    # Build control path once — returns callable dXdt(t) for the derivative
    _, dXdt = CRC.@ignore_derivatives build_control_path(X, ts)

    # Solve the CDE: dz/dt = F_θ(z) · dX/dt
    function dzdt(z, p, t)
        dX = CRC.@ignore_derivatives dXdt(t)
        F3, _ = de.vector_field(z, p.vector_field, st.vector_field)
        dz = reshape(sum(F3 .* reshape(dX, 1, size(dX, 1), size(dX, 2)), dims=2),
            size(F3, 1), size(F3, 3))
        return dz
    end

    ff = ODEFunction{false}(dzdt)
    prob = ODEProblem{false}(ff, x₀, (ts[1], ts[end]), ps)
    sol = solve(prob, de.solver;
        u0=x₀, p=ps, saveat=ts,
        sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()),
        de.kwargs...)

    z_raw = Array(sol)                        # (latent_dim, B, T)
    z = permutedims(z_raw, (1, 3, 2))         # (latent_dim, T, B)

    return z, (vector_field=st.vector_field,)
end


"""
    sample_dynamics(de::CDE, x̂₀, u, ts, ps, st, n_samples)

Sample multiple trajectories from the CDE dynamics by re-sampling x₀ each time.

Arguments:

- `de`: The CDE dynamics.
- `x̂₀`: Probabilistic initial condition `(μ, σ)`.
- `u`: Input signal `(input_dim, T, B)`.
- `ts`: Time points.
- `ps`: Parameters.
- `st`: States.
- `n_samples`: Number of MC samples.

Returns:

- Stacked trajectories `(latent_dim, T, B, n_samples)`.
"""
function sample_dynamics(de::CDE, x̂₀, u, ts, ps, st, n_samples)
    trajectories = []
    for _ in 1:n_samples
        x₀ = sample_rp(x̂₀)
        z, _ = de(x₀, u, ts, ps, st)
        push!(trajectories, z)
    end
    return stack(trajectories, dims=4)
end