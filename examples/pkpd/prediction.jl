using Pkg, Revise, Rhythm, Lux, DifferentialEquations, Random, SciMLSensitivity, ComponentArrays, Optimisers, OptimizationOptimisers, Statistics
using MLUtils, Printf, SciMLSensitivity
using YAML
include("pkpd_standalone.jl")


function generate_dataloader(;n_samples=512, batchsize=64, split=0.8)

    U, X, Y, T = generate_dataset(;n_samples=n_samples);
    U = cat(U..., dims=3);
    Y_padded, Masks, timepoints = pad_matrices(Y, T);
    Y_padded = Y_padded[2:2,:,:];
    X_padded, _ = pad_matrices(X, T; return_timepoints=false);

    timepoints = timepoints./7.f0
    ind_observed = 1:round(Int, split*length(timepoints)); ind_predict = round(Int, split*length(timepoints))+1:length(timepoints);
    println(size(Y_padded), size(U), size(Masks), size(timepoints))
    timepoints_observed = timepoints[ind_observed]; timepoints_predict = timepoints[ind_predict];

    Y_observed = Y_padded[:, ind_observed, :]; Y_predict = Y_padded[:, ind_predict, :];
    U_observed = U[:, ind_observed, :]; U_predict = U[:, ind_predict, :];
    Masks_observed = Masks[:, ind_observed, :]; Mask_predict = Masks[:, ind_predict, :];
    
    (u_train, y_obs_train, _, mask_obs_train, _), (u_test, y_observed_test, y_predict_test, _, mask_predict_test) = splitobs((U, Y_observed, Y_predict, Masks_observed, Mask_predict), at=split);
    train_loader = DataLoader((u_train, y_obs_train, mask_obs_train), batchsize=batchsize, shuffle=true);
    val_loader = DataLoader((u_test, y_observed_test, y_predict_test, repeat(timepoints_predict,1, size(y_predict_test,3)), mask_predict_test), batchsize=batchsize);

    dims = Dict("input_dim" => size(U,1), 
            "state_dim" => size(X_padded,1), 
            "output_dim" => size(Y_padded,1)
            )

    return train_loader, val_loader, dims, timepoints_observed, timepoints_predict
end


function loss_fn(model, θ, st, data)
    (u, y_o, mask), ts_o, λ = data
    ŷ, px₀, kl_pq = model(y_o, u, ts_o, θ, st)
    batch_size = size(u)[end]
    recon_loss = -poisson_loglikelihood(ŷ, y_o, mask)
    kl_init = kl_normal(px₀...)/batch_size
    kl_path = mean(kl_pq[end,:])
    kl_loss =  kl_init + kl_path
    loss = recon_loss + λ * kl_loss
    
    return loss, st, kl_loss
end


function eval_fn(model, θ, st, ts, data, config)
    u, y_o, y_p, t_p, mask = data
    t_p = t_p[:,1]
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    y_o = reverse(y_o, dims=2)
    Ex, Ey_p = predict(model, solver, y_o, u, t_p, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    ŷₘ_p = dropmean(Ey_p, dims=4)
    println(size(ŷₘ_p), size(y_p), size(mask))
    return poisson_loglikelihood(ŷₘ_p, y_p, mask)
end


function viz_fn(model, θ, st, ts, data, config; ch=1, sample_n=1)

    u, y_o, y_p, t_p, mask = data
    y = cat(y_o, y_p, dims=2)
    t_p = t_p[:,1]

    solver = eval(Meta.parse(config["solver"]))

    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    px₀ = (zeros(8, 64), ones(8, 64))
    y_o = reverse(y_o, dims=2)
    Ex, Ey_p = predict(model, solver, y_o, u, t_p, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    
    # Apply mask to Ey
    Ey_masked = Ey_p .* mask

    
    fig = Figure(size = (1200, 900), backgroundcolor = :transparent)
    ax1 = CairoMakie.Axis(fig[1,1], ylabel = "Intervention", backgroundcolor = :transparent, limits = (nothing, (0, 1.5)), yticks = [0, 1])
    ax2 = CairoMakie.Axis(fig[2,1], ylabel = "Tumor size", backgroundcolor = :transparent)
    ax3 = CairoMakie.Axis(fig[3,1], xlabel = "Time (weeks)", ylabel = "Cancer cell count", backgroundcolor = :transparent)
    
    chemo_times = ts[u[1,:,sample_n] .> 0]
    radio_times = ts[u[2,:,sample_n] .> 0]
    
    scatter!(ax1, chemo_times, fill(1, length(chemo_times)), 
             color = :darkgreen, marker = :utriangle, markersize = 15, 
             label = "Chemotherapy session")
    scatter!(ax1, radio_times, fill(1, length(radio_times)), 
             color = :darkorange, marker = :star5, markersize = 15, 
             label = "Radiotherapy session")
    
    ŷₘ = selectdim(dropmean(Ey_masked, dims=4), 3, sample_n)
    ŷₛ = selectdim(dropmean(std(Ey_masked, dims=4), dims=4), 3, sample_n)

    Ey_count = rand.(Poisson.(Ey_masked))
    ŷₘ_count = selectdim(dropmean(Ey_count, dims=4), 3, sample_n)
    ŷₛ_count = selectdim(dropmean(std(Ey_count, dims=4), dims=4), 3, sample_n)


    lines!(ax2, t_p, ŷₘ[ch, :], linewidth = 2, color = (:dodgerblue2, 0.5))
    band!(ax2, t_p, ŷₘ[ch, :].- sqrt.(ŷₛ[ch, :]), ŷₘ[ch, :] .+ sqrt.(ŷₛ[ch, :]), color= (:dodgerblue2, 0.5), label = "Inferred Tumor size")
    
    println(size(y))
    scatter!(ax3, ts, y[ch,:, sample_n], color = :red, label = "Observations", markersize = 15)
    scatter!(ax3, t_p, ŷₘ_count[ch, :], color = (:dodgerblue2, 0.7), label = "Predictions", markersize = 15)
    errorbars!(ax3, t_p, ŷₘ_count[ch, :], ŷₛ_count[ch, :], color = (:dodgerblue2, 0.7), whiskerwidth = 8)
    
    linkxaxes!(ax1, ax2, ax3)

    # Add legends to all axes
    axislegend(ax1, position = :rb, backgroundcolor = :transparent)
    axislegend(ax2, position = :rt, backgroundcolor = :transparent)
    axislegend(ax3, position = :rt, backgroundcolor = :transparent)
    display(fig)

    return fig
end


rng = Random.MersenneTwister(1234)
train_loader, val_loader, dims, timepoints_observed, timepoints_predict = generate_dataloader(;n_samples=512, batchsize=64, split=0.8);
config_path = "/Users/ahmed.elgazzar/Code/MyPackages/Rhythm.jl/configs/default.yml"
config = YAML.load_file(config_path);
exp_path = joinpath(config["experiment"]["path"], config["experiment"]["name"])
isdir(exp_path) ? exp_path : mkpath(exp_path)
model, θ, st = create_latentsde(config["model"], dims, rng);
θ_trained = train(model, θ, st, timepoints_observed, loss_fn, eval_fn, viz_fn, train_loader, val_loader, config["training"], exp_path);



t_test = range(1,365, length=366);
fig = viz_fn(model, θ_trained, st, vcat(timepoints_observed, timepoints_predict), first(val_loader), config["training"]["validation"];
 ch=1, sample_n=6)
save(joinpath(exp_path, "results.pdf"), fig)