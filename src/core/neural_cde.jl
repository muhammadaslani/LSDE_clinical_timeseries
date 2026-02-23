"""
Neural Controlled Differential Equation (Neural CDE)
Kidger et al. 2020 — "Neural Controlled Differential Equations for Irregular Time Series"

Core CDE component: a pure integrator that takes an initial condition and a dX/dt
control path derivative, then integrates the CDE forward in time.

Architecture:
  dz/dt = f_θ(z(t)) · dX/dt     — CDE driven by control path derivative

Usage:
  cde = NeuralCDE(path_dim=3, latent_dim=16)
  z, st = cde(dXdt, tspan, z0, saveat, θ, st)
  # dXdt: callable (t) → (path_dim, B), tspan: (t0, tf)
  # z0: (latent_dim, B), saveat: (T_save,)
  # z: (latent_dim, T_save, B)
"""



# -----------------------------------------------------------------------
# Vector field: f_θ(z) → matrix (latent_dim × path_dim)
# dz/dt = reshape(f_θ(z), latent_dim, path_dim) * dX/dt
# tanh final activation bounds the matrix entries (stability, per paper App. B)
# -----------------------------------------------------------------------
function CDEField(latent_dim::Int, path_dim::Int; hidden_size::Int=64, depth::Int=1)
    out_dim = latent_dim * path_dim
    return @compact(
        net = Chain(
            Dense(latent_dim => hidden_size, tanh),
            [Dense(hidden_size => hidden_size, tanh) for _ in 1:depth]...,
            Dense(hidden_size => out_dim, tanh),
        ),
        latent_dim = latent_dim,
        path_dim = path_dim,
    ) do z
        # z: (latent_dim, B)
        B = size(z, 2)
        F = net(z)                                    # (latent_dim*path_dim, B)
        F3 = reshape(F, latent_dim, path_dim, B)       # (latent_dim, path_dim, B)
        @return F3
    end
end

# -----------------------------------------------------------------------
# NeuralCDE model struct — pure integrator (no init_map)
# -----------------------------------------------------------------------
struct NeuralCDE <: AbstractLuxContainerLayer{(:vector_field,)}
    vector_field # CDEField: z → (latent_dim, path_dim) matrix
end

"""
    NeuralCDE(; path_dim, latent_dim, hidden_size=64, depth=1)

Construct a NeuralCDE core component (pure integrator).

- `path_dim`:   dimension of the control path dX/dt
- `latent_dim`: dimension of the latent state z(t)
"""
function NeuralCDE(; path_dim::Int, latent_dim::Int,
    hidden_size::Int=64, depth::Int=1, kwargs...)
    return NeuralCDE(
        CDEField(latent_dim, path_dim; hidden_size, depth),
    )
end

# -----------------------------------------------------------------------
# Forward pass
# dXdt:   callable (t) → (path_dim, B) — control path derivative
# tspan:  (t0, tf)                     — integration time span
# z0:     (latent_dim, B)              — initial latent state
# saveat: (T_save,)                    — time points at which to return latent states
# Returns z: (latent_dim, T_save, B)
# -----------------------------------------------------------------------
function (model::NeuralCDE)(dXdt, tspan::Tuple,
    z0::AbstractArray{<:Real,2}, saveat::AbstractVector,
    ps::ComponentArray, st::NamedTuple;
    dt::Union{Nothing,Real}=nothing)

    # Solve the CDE, saving at requested time points
    function dzdt(z, p, t)
        dX = CRC.@ignore_derivatives dXdt(t)
        F3, _ = model.vector_field(z, p.vector_field, st.vector_field)
        dz = reshape(sum(F3 .* reshape(dX, 1, size(dX, 1), size(dX, 2)), dims=2), size(F3, 1), size(F3, 3))
        return dz
    end

    ff = ODEFunction{false}(dzdt)
    prob = ODEProblem{false}(ff, z0, tspan, ps)
    solver_kwargs = isnothing(dt) ? NamedTuple() : (; dt=Float64(dt))
    sol = solve(prob, Tsit5();
        u0=z0,
        p=ps,
        saveat=saveat,
        sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()),
        solver_kwargs...)

    # Return latent states at all saved time points
    #    Array(sol) returns (latent_dim, B, T_save) — time is last dimension
    #    Permute to (latent_dim, T_save, B) for consistency
    z_raw = Array(sol)                                              # (latent_dim, B, T_save)
    z_all = permutedims(z_raw, (1, 3, 2))                          # (latent_dim, T_save, B)

    st_new = (vector_field=st.vector_field,)
    return z_all, st_new
end

# -----------------------------------------------------------------------
# Control path construction
# -----------------------------------------------------------------------

"""
    build_control_path(Y, ts)

Build a control path from data using cubic spline interpolation.

- `Y`:  (path_dim, T, B) — data to interpolate (observations, controls, or concatenation)
- `ts`: (T,)             — corresponding time points

Returns `(splines, dXdt)`:
- `splines`: matrix of cubic splines, shape (path_dim, B)
- `dXdt`:    callable `(t) → (path_dim, B)` returning spline derivatives at time `t`,
             ready to pass directly to `NeuralCDE`
"""
function build_control_path(Y::AbstractArray{<:Real,3}, ts::AbstractVector)
    path_dim, T, B = size(Y)
    t_range = range(ts[1], ts[end]; length=T)
    splines = [cubic_spline_interpolation(t_range, Float64.(Y[i, :, b]); bc=Line(OnGrid()))
               for i in 1:path_dim, b in 1:B]

    function dXdt(t)
        return Float32[Interpolations.gradient(splines[i, b], t)[1]
                       for i in 1:path_dim, b in 1:B]   # (path_dim, B)
    end

    return splines, dXdt
end
