using Pkg, Revise, Rhythm, Lux, DifferentialEquations, Random, SciMLSensitivity, ComponentArrays, Optimisers, OptimizationOptimisers, Statistics
using MLUtils, Printf, SciMLSensitivity
using YAML, OneHotArrays
include("pkpd_standalone.jl")


function generate_dataloader(;n_samples=512, batchsize=64, split=0.6)

    U, X, Y₁,Y₂, T = generate_dataset(;n_samples=n_samples);
    Y₁_padded, Masks₁, timepoints = pad_matrices(Y₁, T);
    Y₂_padded, Masks₂, timepoints = pad_matrices(Y₂, T);
    timepoints = timepoints/(7.f0)/52.0f0
    X_padded, _ = pad_matrices(X, T; return_timepoints=false);
    U = cat(U..., dims=3);
    ind_observed = 1:round(Int, split*length(timepoints)); ind_predict = round(Int, split*length(timepoints))+1:length(timepoints);
    timepoints_observed = timepoints[ind_observed]; timepoints_predict = timepoints[ind_predict];

    Y₁_observed = Y₁_padded[:, ind_observed, :]; Y₁_predict = Y₁_padded[:, ind_predict, :];
    Y₂_observed = Y₂_padded[:, ind_observed, :]; Y₂_predict = Y₂_padded[:, ind_predict, :];
    Masks₁_observed = Masks₁[:, ind_observed, :]; Masks₁_predict = Masks₁[:, ind_predict, :];
    Masks₂_observed = Masks₂[:, ind_observed, :]; Masks₂_predict = Masks₂[:, ind_predict, :];
    
    (u_train, y₁_obs_train, y₂_obs_train, y₁_predict_train, y₂_predict_train, mask₁_obs_train, mask₂_obs_train, mask₁_predict_train, mask₂_predict_train),
     (u_test, y₁_obs_test, y₂_obs_test, y₁_predict_test, y₂_predict_test, mask₁_obs_test, mask₂_obs_test, mask₁_predict_test, mask₂_predict_test) = 
    splitobs((U, Y₁_observed, Y₂_observed, Y₁_predict, Y₂_predict, Masks₁_observed, Masks₂_observed, Masks₁_predict, Masks₂_predict), at=split);


    train_loader = DataLoader((u_train, y₁_obs_train,y₂_obs_train, mask₁_obs_train, mask₂_obs_train), batchsize=batchsize, shuffle=true);
    val_loader = DataLoader((u_test, y₁_obs_test, y₂_obs_test, y₁_predict_test, y₂_predict_test,
                             repeat(timepoints_predict,1, size(y₁_predict_test,3)), mask₁_predict_test, mask₂_predict_test), batchsize=batchsize);

    dims = Dict("input_dim" => size(U,1), 
            "state_dim" => size(X_padded,1), 
            "output_dim" => [size(Y₁_padded,1), size( Y₂_padded,1)]
            )

    return train_loader, val_loader, dims, timepoints_observed, timepoints_predict
end


function loss_fn(model, θ, st, data)
    (u, y₁_o,y₂_o, mask₁, mask₂), ts_o, λ = data
    ŷ, px₀, kl_pq = model(vcat(y₁_o,y₂_o), u, ts_o, θ, st)
    ŷ₁, ŷ₂ = ŷ
    batch_size = size(u)[end]
    recon_loss1 =  CrossEntropy_Loss( ŷ₁, y₁_o, mask₁)
    recon_loss2 = -poisson_loglikelihood(ŷ₂, y₂_o, mask₂)
    recon_loss = recon_loss1 + recon_loss2
    kl_init = kl_normal(px₀...)/batch_size
    kl_path = mean(kl_pq[end,:])
    kl_loss =  kl_init + kl_path
    loss = recon_loss + λ * kl_loss
    
    return loss, st, kl_loss
end


function eval_fn(model, θ, st, ts, data, config)
    u, y₁_o,y₂_o, y₁_p,y₂_p, t_p, mask₁, mask₂ = data
    t_p = t_p[:,1]
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    y₁_o = reverse(y₁_o, dims=2)
    y₂_o= reverse(y₂_o, dims=2)
    Ex, Ey_p = predict(model, solver, vcat(y₁_o,y₂_o), u, t_p, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    Ey₁_p, Ey₂_p = Ey_p
    ŷ₁ₘ_p = dropmean(Ey₁_p, dims=4)
    ŷ₂ₘ_p = dropmean(Ey₂_p, dims=4)
    recon_loss1 =  CrossEntropy_Loss( ŷ₁ₘ_p, y₁_p, mask₁)
    recon_loss2 = poisson_loglikelihood(ŷ₂ₘ_p, y₂_p, mask₂)
    return recon_loss1 + recon_loss2
end


function viz_fn(model, θ, st, t_o, data, config; sample_n=1)

    u, y₁_o,y₂_o, y₁_p, y₂_p, t_p, mask₁, mask₂ = data;
    t_p = t_p[:,1]

    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"]);
    
    Ex, Ey_p = predict(model, solver, vcat(reverse(y₁_o, dims=2),reverse(y₂_o, dims=2)), u, t_p, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    Ey₁_p, Ey₂_p = Ey_p;
    t_p = t_p[:,1].*52.0f0 
    t_o = t_o.*52.0f0
    ts=vcat(t_o, t_p)  

    # Apply mask to Ey
    Ey₁_masked = Ey₁_p .* mask₁;
    Ey₂_masked = Ey₂_p .* mask₂;
        
    fig = Figure(size = (1200, 900));
    ax1 = CairoMakie.Axis(fig[1,1], xlabel = "Time (weeks)", ylabel = "Interventions", limits = (nothing, (0, 1.5)), yticks = [0, 1])
    ax2 = CairoMakie.Axis(fig[2,1], xlabel = "Time (weeks)", ylabel = "Health status",limits = (nothing, (-0.5, 6.0)))
    ax3 = CairoMakie.Axis(fig[3,1], xlabel = "Time (weeks)", ylabel = "Tumor size")
    ax4 = CairoMakie.Axis(fig[4,1], xlabel = "Time (weeks)", ylabel = "Cell count")

    chemo_times = ts[u[1,:,sample_n] .> 0]
    radio_times = ts[u[2,:,sample_n] .> 0]
    
    scatter!(ax1, chemo_times, fill(1, length(chemo_times)), 
             color = :darkgreen, marker = :utriangle, markersize = 15, 
             label = "Chemotherapy session")
    scatter!(ax1, radio_times, fill(1, length(radio_times)), 
             color = :darkorange, marker = :star5, markersize = 15, 
             label = "Radiotherapy session")
    
    ŷ₁_m = selectdim(dropmean(Ey₁_masked, dims=4), 3, sample_n)
    ŷ₁_s = selectdim(dropmean(std(Ey₁_masked, dims=4), dims=4), 3, sample_n)
    ŷ₂_m = selectdim(dropmean(Ey₂_masked, dims=4), 3, sample_n)
    ŷ₂_s = selectdim(dropmean(std(Ey₂_masked, dims=4), dims=4), 3, sample_n)

    Ey₂_count = rand.(Poisson.(Ey₂_masked))
    ŷ₂_m_count = selectdim(dropmean(Ey₂_count, dims=4), 3, sample_n)
    ŷ₂_s_count = selectdim(dropmean(std(Ey₂_count, dims=4), dims=4), 3, sample_n)

    y₁_o_class = onecold(y₁_o, Array(0:5))
    y₁_p_class = onecold(y₁_p, Array(0:5))
    ŷ₁_class = onecold(ŷ₁_m, Array(0:5))

    scatter!(ax2, t_o, y₁_o_class[:, sample_n], color = :red, label = "Observations", markersize = 15)
    scatter!(ax2, t_p, y₁_p_class[:, sample_n], color = (:red, 0.4), label = "Ground truth",  markersize = 15)
    scatter!(ax2, t_p, ŷ₁_class, label = "Predictions",color = (:dodgerblue2, 0.7), markersize = 10)

    lines!(ax3, t_p, ŷ₂_m[1,:], color = (:dodgerblue2, 0.5), linewidth = 2)
    band!(ax3, t_p, ŷ₂_m[1,:].- sqrt.(ŷ₂_s[1,:]), ŷ₂_m[1,:] .+ sqrt.(ŷ₂_s[1,:]), color= (:dodgerblue2, 0.5), label = "Inferred Tumor size")

    scatter!(ax4, t_o, y₂_o[1,:,sample_n], color = :red, label = "Observations", markersize = 15)
    scatter!(ax4, t_p, y₂_p[1,:,sample_n],  color = (:red,0.4), label = "Ground truth", markersize = 15)
    scatter!(ax4, t_p, ŷ₂_m_count[1, :], color = (:dodgerblue2, 0.7), label = "Predictions", markersize = 10)
    errorbars!(ax4, t_p, ŷ₂_m_count[1, :], ŷ₂_s_count[1, :], color = (:dodgerblue2, 0.7), whiskerwidth = 8)
    
    linkxaxes!(ax1, ax2, ax3, ax4)

    # Add legends to all axes
    axislegend(ax1, position = :lb, backgroundcolor = :transparent)
    axislegend(ax2, position = :lt, backgroundcolor = :transparent)
    axislegend(ax3, position = :lt, backgroundcolor = :transparent)
    axislegend(ax4, position = :lb, backgroundcolor = :transparent)
    display(fig)


    return fig
end

function predictor(model, θ, st, t_o, t_p, history, u,  config; sample_n=1)
    u_o, y₁_o,y₂_o= history;
    dt=t_o[2]-t_o[1];

    t_p=Array(t_o[end].+1.0f0/52.0f0:dt:t_o[end].+t_p./52.0f0);
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"]);
    
    Ex, Ey_p = predict(model, solver, vcat(reverse(y₁_o, dims=2),reverse(y₂_o, dims=2)), u, t_p, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    Ey₁_p, Ey₂_p = Ey_p;
    t_p = t_p.*52.0f0 
    t_o = t_o.*52.0f0
    ts=vcat(t_o, t_p)  

    fig = Figure(size = (1200, 900));
    ax1 = CairoMakie.Axis(fig[1,1], xlabel = "Time (weeks)", ylabel = "Interventions", limits = (nothing, (0, 1.5)), yticks = [0, 1])
    ax2 = CairoMakie.Axis(fig[2,1], xlabel = "Time (weeks)", ylabel = "Health status",limits = (nothing, (-0.5, 6.0)))
    ax3 = CairoMakie.Axis(fig[3,1], xlabel = "Time (weeks)", ylabel = "Tumor size")
    ax4 = CairoMakie.Axis(fig[4,1], xlabel = "Time (weeks)", ylabel = "Cell count")

    chemo_times_o = t_o[u_o[1,:,sample_n] .> 0]
    radio_times_o = t_o[u_o[2,:,sample_n] .> 0]
    chemo_times_p = t_p[u_p[1,:,sample_n] .> 0]
    radio_times_p = t_p[u_p[2,:,sample_n] .> 0]

    scatter!(ax1, chemo_times_o, fill(1, length(chemo_times_o)), 
             color = :darkgreen, marker = :utriangle, markersize = 15, 
             label = "Chemotherapy session past administration")
    scatter!(ax1, radio_times_o, fill(1, length(radio_times_o)), 
             color = :darkorange, marker = :star5, markersize = 15, 
             label = "Radiotherapy session past administration")
    scatter!(ax1, chemo_times_p, fill(1, length(chemo_times_p)), 
             color = :red, marker = :utriangle, markersize = 15, 
             label = "Chemotherapy session future administration")
    scatter!(ax1, radio_times_p, fill(1, length(radio_times_p)), 
             color = :blue, marker = :star5, markersize = 15, 
             label = "Radiotherapy session future administration")

             
    ŷ₁_m = selectdim(dropmean(Ey₁_p, dims=4), 3, sample_n)
    ŷ₂_m = selectdim(dropmean(Ey₂_p, dims=4), 3, sample_n)
    ŷ₂_s = selectdim(dropmean(std(Ey₂_p, dims=4), dims=4), 3, sample_n)

    ŷ₂_count = rand.(Poisson.(Ey₂_p))
    ŷ₂_m_count = selectdim(dropmean(ŷ₂_count, dims=4), 3, sample_n)
    ŷ₂_s_count = selectdim(dropmean(std(ŷ₂_count, dims=4), dims=4), 3, sample_n)

    y₁_o_class = onecold(y₁_o, Array(0:5))
    ŷ₁_class = onecold(ŷ₁_m, Array(0:5))

    scatter!(ax2, t_o, y₁_o_class[:, sample_n], color = :red, label = "Observations", markersize = 15)
    scatter!(ax2, t_p, ŷ₁_class, label = "Predictions",color = (:dodgerblue2, 0.7), markersize = 15)

    lines!(ax3, t_p, ŷ₂_m[1,:], color = (:dodgerblue2, 0.5), linewidth = 2)
    band!(ax3, t_p, ŷ₂_m[1,:].- sqrt.(ŷ₂_s[1,:]), ŷ₂_m[1,:] .+ sqrt.(ŷ₂_s[1,:]), color= (:dodgerblue2, 0.5), label = "Inferred Tumor size")

    scatter!(ax4, t_o, y₂_o[1,:,sample_n], color = :red, label = "Observations", markersize = 15)
    scatter!(ax4, t_p, ŷ₂_m_count[1, :], color = (:dodgerblue2, 0.7), label = "Predictions", markersize = 15)
    errorbars!(ax4, t_p, ŷ₂_m_count[1, :], ŷ₂_s_count[1, :], color = (:dodgerblue2, 0.7), whiskerwidth = 8)
    
    linkxaxes!(ax1, ax2, ax3, ax4)

    # Add legends to all axes
    axislegend(ax1, position = :lb, backgroundcolor = :transparent)
    axislegend(ax2, position = :lt, backgroundcolor = :transparent)
    axislegend(ax3, position = :lt, backgroundcolor = :transparent)
    axislegend(ax4, position = :lb, backgroundcolor = :transparent)
    display(fig)
    return fig

end 

rng = Random.MersenneTwister(1234)
split=0.6
train_loader, val_loader, dims, timepoints_observed, timepoints_predict = generate_dataloader(;n_samples=256, batchsize=32, split=split);
config_path = "configs/default.yml"
config = YAML.load_file(config_path);
exp_path = joinpath(config["experiment"]["path"], config["experiment"]["name"])
isdir(exp_path) ? exp_path : mkpath(exp_path)
model, θ, st = create_latentsde(config["model"], dims, rng);
θ_trained = train(model, θ, st, timepoints_observed, loss_fn, eval_fn, viz_fn, train_loader, val_loader, config["training"], exp_path);

#t_test = range(1,365, length=366);
fig = viz_fn(model, θ_trained, st, timepoints_observed, first(val_loader), config["training"]["validation"]; sample_n=7)
#save(joinpath(exp_path, "results.pdf"), fig)

u, y₁_o,y₂_o, y₁_p, y₂_p, t_p, mask₁, mask₂=first(val_loader);
u_o=u[:,1:round(Int,split*size(u)[2]),:];
u_p=u[:,round(Int,split*size(u)[2])+1:end,:];
history=(u_o, y₁_o,y₂_o);
t_p=size(u_p)[2];

predictor(model, θ_trained, st, timepoints_observed, t_p, history, u_p, config["training"]["validation"]; sample_n=2)