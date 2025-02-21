using Pkg, Revise, Rhythm, Lux, DifferentialEquations, Random, SciMLSensitivity, ComponentArrays, Optimisers, OptimizationOptimisers, Statistics
using MLUtils, Printf, SciMLSensitivity

ts = 0:0.1:9.9 |> Array{Float32};
x = rand32(2, 100, 1);
u = rand32(1, 100, 1);

drift = @compact(vf = Dense(3,2, tanh)) do xu 
    x, u = xu 
    @return vf(vcat(xu...))
end;

diffusion = @compact(vg = Scale(2, sigmoid)) do x 
    @return vg(x)
end;

dynamics = SDE(drift, Dense(5, 2, tanh), diffusion, EulerHeun(), saveat=ts, dt=0.1f0);
model = LatentSDE(dynamics=dynamics);

rng = Random.default_rng();
θ, st = Lux.setup(rng, model);
θ = θ |> ComponentArray{Float32};
ŷ, px₀, kl_path = model(x, u, ts, θ, st);


X = rand32(2, 100, 10);
U = rand32(1, 100, 10);


(x_train, u_train), (x_test, u_test) = splitobs((X, U), at=0.8);
train_loader = DataLoader((x_train, u_train), batchsize=2);
val_loader = DataLoader((x_test, u_test), batchsize=2);

function loss_fn(model, θ, st, data)
    x, u, ts, λ = data
    x̂, px₀, kl_pq = model(x, u, ts, θ, st)
    batch_size = size(x)[end]
    recon_loss = mse(x̂, x)/batch_size
    kl_init = kl_normal(px₀...)/batch_size
    kl_path = mean(kl_pq[end,:])
    kl_loss =  kl_init + kl_path
    loss = recon_loss + λ * kl_loss
    return loss, st, kl_loss
end


function eval_fn(model, θ, st, ts, data, config)
    y, u,= data
    Ex, Ey = smooth(model, config.solver, y, u, ts, θ, st, config.n_samples, config.dev; config.kwargs...)
    ŷₘ = dropmean(Ey, dims=4)
    return mse(ŷₘ, y)
end

function viz_fn(kwargs...)
    return
end

config = (lr =1e-3, epochs=100, optimizer=AdamW(1e-3) , solver=EM(), log_freq=10, viz_freq=10, save_path=".", stop_patience=50, lrdecay_patience=20, n_samples=10, dev=cpu_device(), kwargs=Dict(:saveat=>ts, :dt=>0.1f0))

train(model, θ, st, ts, loss_fn, eval_fn, viz_fn, train_loader, val_loader, config);




Base.@kwdef struct ModelParameters
    ρ::Float64 = 8e-3    # Tumor growth rate
    K::Float64 = 100.0   # Tumor carrying capacity
    β_c::Float64 = 0.15  # Linear effect of chemotherapy
    ω_c::Float64 = 1.0   # Chemotherapy sessions frequency (every X weeks)
    α_r::Float64 = 0.4   # Linear effect of radiotherapy
    β_r::Float64 = 0.1   # Quadratic effect of radiotherapy
    ω_r::Float64 = 3.0   # Radiotherapy sessions frequency (every X weeks)
    δ::Float64 = 0.023   # Reduced immune growth rate
    β_I::Float64 = 0.15  # Increased drug-induced immune suppression
    α_I::Float64 = 0.16  # Increased radiotherapy-induced immune suppression
    θ_I::Float64 = 0.08  # Immune stimulation by tumor
    λ_I::Float64 = 0.002 # Immune suppression by large tumors
    ω_I::Float64 = 0.07  # Immune decay rate
    I_max::Float64 = 0.95 # Max immune response
    γ_S::Float64 = 5e-2  # Immune effect on health
    θ_S::Float64 = 40.0  # Health recovery rate
    λ_S::Float64 = 500.0 # Health impact of tumor
end