using Pkg, Revise, Rhythm, Lux, DifferentialEquations, Random, SciMLSensitivity, ComponentArrays, Optimisers, OptimizationOptimisers, Statistics
using MLUtils, Printf, SciMLSensitivity
include("pkpd_standalone.jl")


function generate_dataloader(;n_samples=512, batchsize=64, split=0.8)
    U, X, Y, T = generate_dataset(;n_samples=n_samples);
    (u_train, x_train, y_train, t_train), (u_test, x_test, y_test, t_test) = splitobs((U, X, Y, T), at=split);
    train_loader = DataLoader((u_train, x_train, y_train, t_train), batchsize=batchsize);
    val_loader = DataLoader((u_test, x_test, y_test, t_test), batchsize=batchsize);
    dims = Dict(::input_dim => size(first(U),1), 
            ::state_dim => size(first(X),1), 
            ::output_dim => size(first(Y),1)
            )
    return train_loader, val_loader, dims
end



train_loader, val_loader, dims = generate_dataloader(;n_samples=512, batchsize=64, split=0.8);
config_path = "/Users/ahmed.elgazzar/Code/MyPackages/Rhythm.jl/configs/default.yml"
config = YAML.load_file(config_path)
model = create_latentsde(config["model"], dims)



function loss_fn(model, θ, st, data)
    u, x, y, ts = data
    # TODO: Convert to equally sized arrays via padding and get the mask
    x̂, px₀, kl_pq = model(y, u, ts, θ, st)
    # TODO: Add the mask to the loss function
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


train(model, θ, st, ts, loss_fn, eval_fn, viz_fn, train_loader, val_loader, config);

