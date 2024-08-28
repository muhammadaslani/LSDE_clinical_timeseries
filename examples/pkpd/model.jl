using Pkg, Revise, Rhythm, Lux, DifferentialEquations, Random, SciMLSensitivity, ComponentArrays, Optimisers, OptimizationOptimisers, Statistics
using MLUtils, Printf, SciMLSensitivity
using YAML
include("pkpd_standalone.jl")


function generate_dataloader(;n_samples=512, batchsize=64, split=0.8)
    U, X, Y, T = generate_dataset(;n_samples=n_samples);
    Y_padded, Masks, timepoints = pad_matrices(Y, T);
    timepoints = timepoints/7.f0
    Y_padded = Y_padded[2:2, :, :]
    X_padded, _ = pad_matrices(X, T; return_timepoints=false);
    U = cat(U..., dims=3);
    (u_train, x_train, y_train, mask_train), (u_test, x_test, y_test, mask_test) = splitobs((U, X_padded, Y_padded, Masks), at=split);
    train_loader = DataLoader((u_train, x_train, y_train, mask_train), batchsize=batchsize, shuffle=true);
    val_loader = DataLoader((u_test, x_test, y_test, mask_test), batchsize=batchsize);
    dims = Dict("input_dim" => size(U,1), 
            "state_dim" => size(X_padded,1), 
            "output_dim" => size(Y_padded,1)
            )
    return train_loader, val_loader, dims, timepoints
end


function loss_fn(model, θ, st, data)
    (u, x, y, mask), ts, λ = data
    ŷ, px₀, kl_pq = model(y, u, ts, θ, st)
    batch_size = size(x)[end]
    recon_loss = -poisson_loglikelihood(ŷ, y, mask)
    kl_init = kl_normal(px₀...)/batch_size
    kl_path = mean(kl_pq[end,:]) # Think about wether masking is needed here
    kl_loss =  kl_init + kl_path
    loss = recon_loss + λ * kl_loss
    return loss, st, kl_loss
end


function eval_fn(model, θ, st, ts, data, config)
    u, x, y, mask = data
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    Ex, Ey = smooth(model, solver, y, u, ts, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    ŷₘ = dropmean(Ey, dims=4)
    return poisson_loglikelihood(ŷₘ, y, mask)
end




rng = Random.MersenneTwister(1234)
train_loader, val_loader, dims, timepoints = generate_dataloader(;n_samples=512, batchsize=64, split=0.8);
config_path = "/Users/ahmed.elgazzar/Code/MyPackages/Rhythm.jl/configs/default.yml"
config = YAML.load_file(config_path);
exp_path = joinpath(config["experiment"]["path"], config["experiment"]["name"])
isdir(exp_path) ? exp_path : mkpath(exp_path)
model, θ, st = create_latentsde(config["model"], dims, rng);
θ_trained = train(model, θ, st, timepoints, loss_fn, eval_fn, viz_fn, train_loader, val_loader, config["training"], exp_path);

function viz_fn(model, θ, st, ts, data, config; ch=1, sample_n=1)
    u, x, y, mask = data
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    px₀ = (zeros(8, 64), ones(8, 64))
    Ex, Ey = generate(model, solver, px₀, u, ts, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    fig = Figure(size = (1200, 900), backgroundcolor = :transparent)
    ax1 = CairoMakie.Axis(fig[1,1], ylabel = "Intervention", backgroundcolor = :transparent, limits = (nothing, (0, 1.5)), yticks = [0, 1])
    ax2 = CairoMakie.Axis(fig[2,1], ylabel = "tumor size", backgroundcolor = :transparent)
    ax3 = CairoMakie.Axis(fig[3,1], xlabel = "time (weeks)", ylabel = "Cancer cell count", backgroundcolor = :transparent)
    
    chemo_times = ts[u[1,:,sample_n] .> 0]
    radio_times = ts[u[2,:,sample_n] .> 0]
    
    scatter!(ax1, chemo_times, fill(1, length(chemo_times)), 
             color = :green, marker = :diamond, markersize = 15, 
             label = "Chemotherapy session")
    scatter!(ax1, radio_times, fill(1, length(radio_times)), 
             color = :yellow, marker = :star5, markersize = 15, 
             label = "Radiotherapy session")
    
    ŷₘ = selectdim(dropmean(Ey, dims=4), 3, sample_n)
    ŷₛ = selectdim(dropmean(std(Ey, dims=4), dims=4), 3, sample_n)

    Ey_count = rand.(Poisson.(Ey))
    ŷₘ_count = selectdim(dropmean(Ey_count, dims=4), 3, sample_n)
    ŷₛ_count = selectdim(dropmean(std(Ey_count, dims=4), dims=4), 3, sample_n)

    
    lines!(ax2, ts, ŷₘ[ch,:], linewidth = 2, color = (:dodgerblue2, 0.5))
    band!(ax2, ts, ŷₘ[ch,:] .-   sqrt.(ŷₛ[ch,:]) , ŷₘ[ch,:] .+ sqrt.(ŷₛ[ch,:]), color= (:dodgerblue2, 0.5),  label = "Inferred Tumor size")
    
    scatter!(ax3, ts, y[ch,:, sample_n], color = :red, label = "Observations", markersize = 15)
    scatter!(ax3, ts, ŷₘ_count[ch,:], color = (:dodgerblue2, 0.7), label = "Predictions", markersize = 15)
    errorbars!(ax3, ts, ŷₘ_count[ch,:], ŷₛ_count[ch,:], color = (:dodgerblue2, 0.7), whiskerwidth = 8)
    linkxaxes!(ax1, ax2, ax3)

    # Add legends to all axes
    axislegend(ax1, position = :rb, backgroundcolor = :transparent)
    axislegend(ax2, position = :rt, backgroundcolor = :transparent)
    axislegend(ax3, position = :rt, backgroundcolor = :transparent)
    display(fig)

    return 
end


t_test = range(1,365, length=366);
viz_fn(model, θ_trained, st, timepoints, first(val_loader), config["training"]["validation"]; ch=1, sample_n=3)