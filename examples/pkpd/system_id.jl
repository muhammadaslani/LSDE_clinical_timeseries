using Pkg, Revise, Rhythm, Lux, DifferentialEquations, Random, SciMLSensitivity, ComponentArrays, Optimisers, OptimizationOptimisers, Statistics
using MLUtils, Printf, SciMLSensitivity, OneHotArrays
using YAML
include("pkpd_standalone.jl")

set_theme!(atom_one_dark_theme)
function generate_dataloader(;n_samples=512, batchsize=64, split=0.8)
    U, X, Y₁,Y₂, T = generate_dataset(;n_samples=n_samples);
    Y₁_padded, Masks₁, timepoints = pad_matrices(Y₁, T);
    Y₂_padded, Masks₂, timepoints = pad_matrices(Y₂, T);
    timepoints = timepoints/(7.f0)/52.0f0
    X_padded, _ = pad_matrices(X, T; return_timepoints=false);
    U = cat(U..., dims=3);
    (u_train, x_train, y₁_train,y₂_train, mask₁_train, mask₂_train), (u_test, x_test, y₁_test, y₂_test, mask₁_test, mask₂_test) = splitobs((U, X_padded, Y₁_padded,Y₂_padded, Masks₁, Masks₂), at=split);
    train_loader = DataLoader((u_train, x_train, y₁_train,y₂_train, mask₁_train, mask₂_train), batchsize=batchsize, shuffle=true);
    val_loader = DataLoader((u_test, x_test, y₁_test, y₂_test, mask₁_test, mask₂_test), batchsize=batchsize);
    dims = Dict(
                "input_dim" => size(U,1), 
                "state_dim" => size(X_padded,1), 
                "output_dim" => [size(Y₁_padded,1), size( Y₂_padded,1)],
            )
    return train_loader, val_loader, dims, timepoints
end


function loss_fn(model, θ, st, data)
    (u, x, y₁,y₂, mask₁, mask₂), ts, λ = data
    ŷ, px₀, kl_pq = model(vcat(y₁,y₂),u, ts, θ, st)
    ŷ₁, ŷ₂ = ŷ
    batch_size = size(x)[end]
    recon_loss1 =  CrossEntropy_Loss( ŷ₁,y₁, mask₁)
    recon_loss2 = -poisson_loglikelihood(ŷ₂, y₂, mask₂)
    recon_loss = recon_loss1 + recon_loss2
    kl_init = kl_normal(px₀...)/batch_size
    kl_path = mean(kl_pq[end,:]) # Think about wether masking is needed here
    kl_loss =  kl_init + kl_path
    loss = recon_loss + λ * kl_loss
    return loss, st, kl_loss
end


function eval_fn(model, θ, st, ts, data, config)
    u, x, y₁,y₂, mask₁, mask₂ = data
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    px₀ = (zeros32(config["latent_dim"], size(y₁)[end]), ones32(config["latent_dim"], size(y₁)[end]))
    Ex, Ey = generate(model, solver, px₀, u, ts, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    ŷ₁_m = dropmean(Ey[1], dims=4)
    ŷ₂_m = dropmean(Ey[2], dims=4)
    val_loss_1 = CrossEntropy_Loss( ŷ₁_m,y₁, mask₁)
    val_loss_2 = poisson_loglikelihood(ŷ₂_m, y₂, mask₂)
    return val_loss_1+ val_loss_2
end


function viz_fn(model, θ, st, ts, data, config; sample_n=1)
    u, x, y₁,y₂, mask₁, mask₂ = data
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    px₀ = (zeros32(config["latent_dim"], size(y₁)[end]), ones32(config["latent_dim"], size(y₁)[end]))
    Ex, Ey = generate(model, solver, px₀, u, ts, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    Ey₁, Ey₂= Ey;   # Ey₁ is the predicted porbability for health status classes, Ey₂ is the predicted tumor size

    ts = ts.*52.0f0 

    # Apply mask to Ey
    Ey₁_masked = Ey₁ .* mask₁
    Ey₂_masked = Ey₂ .* mask₂


    # Find valid indices
    valid_indices = findall(mask₁[1, :, sample_n] .== 1)
    valid_ts = ts[valid_indices]
    valid_y₁ = y₁[:, valid_indices, sample_n]
    valid_y₂ = y₂[1, valid_indices, sample_n]
    
    fig = Figure(size = (1200, 900));
    ax1 = CairoMakie.Axis(fig[1,1], xlabel = "Time (weeks)", ylabel = "Interventions", limits = (nothing, (0, 1.5)), yticks = [0, 1])
    ax2 = CairoMakie.Axis(fig[2,1], xlabel = "Time (weeks)", ylabel = "Health status",limits = (nothing, (-0.5, 6.0)))
    ax3 = CairoMakie.Axis(fig[3,1], xlabel = "Time (weeks)", ylabel = "Tumor size")
    ax4 = CairoMakie.Axis(fig[4,1], xlabel = "Time (weeks)", ylabel = "Cell count")

    
    chemo_times = ts[u[1,:,sample_n] .> 0]
    radio_times = ts[u[2,:,sample_n] .> 0]
    
    scatter!(ax1, chemo_times, fill(1, length(chemo_times)), 
             color = atom_one_dark[:red], marker = :utriangle, markersize = 15, 
             label = "Chemotherapy session")
    scatter!(ax1, radio_times, fill(1, length(radio_times)), 
             color = atom_one_dark[:yellow], marker = :star5, markersize = 15, 
             label = "Radiotherapy session")
    
    ŷ₁_m = selectdim(dropmean(Ey₁_masked, dims=4), 3, sample_n)
    ŷ₁_s = selectdim(dropmean(std(Ey₁_masked, dims=4), dims=4), 3, sample_n)
    ŷ₂_m = selectdim(dropmean(Ey₂_masked, dims=4), 3, sample_n)
    ŷ₂_s = selectdim(dropmean(std(Ey₂_masked, dims=4), dims=4), 3, sample_n)
    

    #estimate cell counts from Inferred tumor size
    Ey₂_count = rand.(Poisson.(Ey₂_masked))
    ŷ₂_m_count = selectdim(dropmean(Ey₂_count, dims=4), 3, sample_n)
    ŷ₂_s_count = selectdim(dropmean(std(Ey₂_count, dims=4), dims=4), 3, sample_n)


    valid_x = x[:,1:7:end,:][:, valid_indices, sample_n]
    # getting valid values for each obersvation and prediction 
    valid_ŷ₁_m = ŷ₁_m[:, valid_indices]
    valid_ŷ₁_s = ŷ₁_s[:, valid_indices]
    valid_ŷ₂_m = ŷ₂_m[1, valid_indices]
    valid_ŷ₂_s = ŷ₂_s[1, valid_indices]
    valid_ŷ₂_count_m = ŷ₂_m_count[1, valid_indices]
    valid_ŷ₂_count_s = ŷ₂_s_count[1, valid_indices]
    # one cold encoding for health status
    valid_ŷ₁_class = onecold(valid_ŷ₁_m, Array(0:5))
    valid_y₁_class = onecold(valid_y₁, Array(0:5))

    scatter!(ax2, valid_ts, valid_y₁_class, color = atom_one_dark[:purple], label = "Observed health status", markersize = 15)
    scatter!(ax2, valid_ts, valid_ŷ₁_class, color = (atom_one_dark[:cyan], 0.5), label = "Predicted health status", markersize = 15)

    lines!(ax3, valid_ts, valid_x[1,:], linewidth = 2, color = (atom_one_dark[:purple], 0.7), label="True Tumor size (unobserved)")
    lines!(ax3, valid_ts, valid_ŷ₂_m, linewidth = 2, color = (atom_one_dark[:cyan], 0.5))
    band!(ax3, valid_ts, valid_ŷ₂_m .- sqrt.(valid_ŷ₂_s), valid_ŷ₂_m .+ sqrt.(valid_ŷ₂_s), color= (atom_one_dark[:cyan], 0.5), label = "Inferred Tumor size")

    scatter!(ax4, valid_ts, valid_y₂, color = atom_one_dark[:purple], label = "Observed tumor cell count", markersize = 15)
    scatter!(ax4, valid_ts, valid_ŷ₂_count_m, color = (atom_one_dark[:cyan], 0.5), label = "Predicted tumor cell count", markersize = 15)
    errorbars!(ax4, valid_ts, valid_ŷ₂_count_m, valid_ŷ₂_count_s, color = (atom_one_dark[:cyan], 0.5), whiskerwidth = 8)

    linkxaxes!(ax1, ax2, ax3, ax4)

    # Add legends to all axes
    axislegend(ax1, position = :rb, backgroundcolor = :transparent)
    axislegend(ax2, position = :rt, backgroundcolor = :transparent)
    axislegend(ax3, position = :rt, backgroundcolor = :transparent)
    axislegend(ax4, position = :rt, backgroundcolor = :transparent)

    return fig
end


rng = Random.MersenneTwister(1234)
train_loader, val_loader,dims, timepoints = generate_dataloader(;n_samples=128, batchsize=32, split=0.6);
config = YAML.load_file("./configs/default.yml");
exp_path = joinpath(config["experiment"]["path"], config["experiment"]["name"])
isdir(exp_path) ? exp_path : mkpath(exp_path)
model, θ, st = create_latentsde(config["model"], dims, rng);
θ_trained = train(model, θ, st, timepoints, loss_fn, eval_fn, viz_fn, train_loader, val_loader, config["training"], exp_path);

fig = viz_fn(model, θ_trained, st, timepoints, first(train_loader), config["training"]["validation"]; sample_n=2)
save(joinpath(exp_path, "results_prediction_.pdf"), fig)
