using  Revise, Rhythm, Lux, DifferentialEquations, Random, SciMLSensitivity, ComponentArrays, Optimisers, OptimizationOptimisers, Statistics
using MLUtils, Printf, SciMLSensitivity, OneHotArrays
using YAML
include("pkpd_standalone.jl")

function generate_dataloader(; n_samples=512, batchsize=64, split=(0.5,0.3))
    U, X, Y₁, Y₂, T, covariates = generate_dataset(n_samples=n_samples)
    Y₁_padded, Masks₁, timepoints = pad_matrices(Y₁, T)
    Y₂_padded, Masks₂, _ = pad_matrices(Y₂, T)
    X_padded, _ = pad_matrices(X, T; return_timepoints=false)
    Y₁_irreg, Y₂_irreg, Masks₁, Masks₂ = irregularize(Y₁_padded,Y₂_padded, Masks₁, Masks₂)
    timepoints /= (7.0f0 * 52.0f0)  # Normalize timepoints
  
    covars=repeat(reshape(covariates,2,1,size(covariates,2)),1,size(Y₁_padded)[2],1)
    U = cat(U..., dims=3)
    data = (U, X_padded, covars, Y₁_irreg, Y₂_irreg, Masks₁, Masks₂)
    (train_data,test_data, val_data) = splitobs(data, at=split)

    train_loader = DataLoader(train_data, batchsize=batchsize, shuffle=true)
    test_loader = DataLoader(test_data, batchsize=batchsize, shuffle=true)
    val_loader = DataLoader(val_data, batchsize=batchsize, shuffle=false)
    dims = Dict(
        "obs_dim" => [size(covars,1),size(Y₁_irreg, 1), size(Y₂_irreg, 1)],
        "input_dim" => size(U, 1),
        "state_dim" => size(X_padded, 1),
        "output_dim" => [size(Y₁_irreg, 1), size(Y₂_irreg, 1)]
    )

    return train_loader, test_loader, val_loader, dims, timepoints, covars
end

function loss_fn(model, θ, st, data)
    (u, x, covars, y₁, y₂, mask₁, mask₂), ts, λ = data
    ŷ, px₀, kl_pq = model(vcat(covars,y₁, y₂), u, ts, θ, st)
    ŷ₁, ŷ₂ = ŷ
    
    val_indx₁= findall(mask₁.==1)
    val_indx₂= findall(mask₂.==1)

    recon_loss = CrossEntropyLoss(;agg=mean,logits=true)(ŷ₁[val_indx₁], y₁[val_indx₁]) - poisson_loglikelihood(ŷ₂[val_indx₂], y₂[val_indx₂])
    kl_loss = kl_normal(px₀...) / size(x)[end] + mean(kl_pq[end, :])
    loss = recon_loss + λ * kl_loss
    return loss, st, kl_loss
end

function eval_fn(model, θ, st, ts, data, config)
    u, x, covars, y₁, y₂, mask₁, mask₂ = data
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    px₀ = (zeros32(config["latent_dim"], size(y₁)[end]), ones32(config["latent_dim"], size(y₁)[end]))
    Ex, Ey = generate(model, solver, px₀, u, ts, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    ŷ₁_m, ŷ₂_m = dropmean(Ey[1], dims=4), dropmean(Ey[2], dims=4)
    val_indx₁= findall(mask₁.==1)
    val_indx₂= findall(mask₂.==1)
    return CrossEntropyLoss(;agg=mean, logits=true,  epsilon=1e-10)(ŷ₁_m[val_indx₁], y₁[val_indx₁]) - poisson_loglikelihood(ŷ₂_m[val_indx₂], y₂[val_indx₂])
end

function viz_fn_sys_id2(model, θ, st, ts, data, config; sample_n=1)
    u, x, covars, y₁, y₂, mask₁, mask₂ = data
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    px₀ = (zeros32(config["latent_dim"], size(y₁)[end]), ones32(config["latent_dim"], size(y₁)[end]))
    Ex, Ey = generate(model, solver, px₀, u, ts, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    Ey₁, Ey₂ = Ey   # Ey₁ is the predicted porbability for health status classes, Ey₂ is the predicted tumor size

    ts = ts .* 52.0f0
    # Find valid indices
    valid_indx = findall(mask₁[1, :, sample_n] .== 1)
    ts_valid = ts[valid_indx]
    y₁_valid, y₂_valid= y₁[:, valid_indx, sample_n],y₂[:, valid_indx, sample_n]
    
    ŷ₁_m ,ŷ₁_s= dropmean(Ey₁, dims=4),dropmean(std(Ey₁, dims=4), dims=4)
    ŷ₂_m ,ŷ₂_s= dropmean(Ey₂, dims=4), dropmean(std(Ey₂, dims=4), dims=4)
    
    #estimate cell counts from Inferred tumor size
    Ey₂_count = rand.(Poisson.(Ey₂))
    ŷ₂_count_m, ŷ₂_count_s = dropmean(Ey₂_count, dims=4),dropmean(std(Ey₂_count, dims=4), dims=4)


    max_valid_time= ts_valid[end]
    valid_indices_chemo = findall(i -> u[1,i, sample_n] == 1 && ts[i] <= max_valid_time, 1:length(ts))
    valid_indices_radio = findall(i -> u[2,i, sample_n] == 1 && ts[i] <= max_valid_time, 1:length(ts))
    fig = Figure(size=(900, 600))


    ax1= CairoMakie.Axis(fig[1, 1], xlabel="Time (weeks)", ylabel="Interventions", limits=(nothing, (0, 1.5)), yticks=[0,1])
    ax2 = CairoMakie.Axis(fig[2, 1], xlabel="Time (weeks)", ylabel="Health status", limits=(nothing, (-0.5, 6.0)))
    ax3 = CairoMakie.Axis(fig[3, 1], xlabel="Time (weeks)", ylabel="Tumor size")
    ax4 = CairoMakie.Axis(fig[4, 1], xlabel="Time (weeks)", ylabel="Cell count")
    scatter!(ax1, ts[valid_indices_chemo], ones(length(u[valid_indices_chemo])),marker = :utriangle,markersize = 15,color = :blue)
    scatter!(ax1, ts[valid_indices_radio], ones(length(u[valid_indices_radio])),marker = :star5,markersize = 15,color = :red)
    xlims!(ax1, minimum(ts), maximum(ts))
    lines!(ax3, Array(0:365)/7, x[1,:, sample_n], linewidth=2, color=(atom_one_dark[:gray], 1.0), label="True Tumor size (unobserved)")
    lines!(ax3, ts, ŷ₂_m[1,:,sample_n], linewidth=2, color=(atom_one_dark[:cyan], 0.9), label="Inferred Tumor size")
    band!(ax3, ts, ŷ₂_m[1,:,sample_n] .- sqrt.(ŷ₂_s[1,:,sample_n]), ŷ₂_m[1,:,sample_n] .+ sqrt.(ŷ₂_s[1,:,sample_n]), color=(atom_one_dark[:cyan], 0.3))
    lines!(ax1, ax3, ts_valid, ŷ₂_m[1,valid_indx,sample_n], linewidth=3, color=(atom_one_dark[:purple], 0.7), label=" valid")
    scatter!(ax3, ts, ŷ₂_m[1,:,sample_n], color=(atom_one_dark[:cyan], 0.7), markersize=10)
    scatter!(ax3, ts_valid, ŷ₂_m[1,valid_indx,sample_n], color=(atom_one_dark[:purple], 0.7), markersize=10)

    display(fig)
    return fig
end

function viz_fn_sys_id(model, θ, st, ts, data, config; sample_n=1)
    u, x, covars, y₁, y₂, mask₁, mask₂ = data
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    px₀ = (zeros32(config["latent_dim"], size(y₁)[end]), ones32(config["latent_dim"], size(y₁)[end]))
    Ex, Ey = generate(model, solver, px₀, u, ts, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    Ey₁, Ey₂ = Ey   # Ey₁ is the predicted porbability for health status classes, Ey₂ is the predicted tumor size
    ts= ts .* 365.0f0
    ŷ₁_m= dropmean(Ey₁, dims=4)
    ŷ₁_conf=maximum(softmax(ŷ₁_m, dims=1), dims=1)
    ŷ₂_m ,ŷ₂_s= dropmean(Ey₂, dims=4), dropmean(std(Ey₂, dims=4), dims=4)
    #estimate cell counts from Inferred tumor size
    Ey₂_count = rand.(Poisson.(Ey₂))
    ŷ₂_count_m, ŷ₂_count_s = dropmean(Ey₂_count, dims=4),dropmean(std(Ey₂_count, dims=4), dims=4)
    # one cold encoding for health status

    # sampling predictions every 7 days
    ŷ₁_m_w, ŷ₁_conf_w, ŷ₂_m_w, ŷ₂_s_w= ŷ₁_m[:,1:7:end,:],  ŷ₁_conf[:,1:7:end,:], ŷ₂_m[:,1:7:end,:],ŷ₂_s[:,1:7:end,:]
    ŷ₂_count_m_w , ŷ₂_count_s_w = ŷ₂_count_m[:,1:7:end,:], ŷ₂_count_s[:,1:7:end,:]
    ts_w= ts[1:7:end]
    ŷ₁_class_w = onecold(ŷ₁_m_w, Array(0:5))
    y₁_class = onecold(y₁, Array(0:5))
    ts_w_valid = ts_w[findall(mask₂[1,:,sample_n] .== 1)]
    max_valid_time= ts_w_valid[end]
    ts_valid = ts[findall(i-> ts[i]<=max_valid_time, 1:length(ts))]
    x_valid = x[:,findall(i-> ts[i]<=max_valid_time, 1:length(ts)),sample_n]
    ŷ₂_m_valid = ŷ₂_m[1,findall(i-> ts[i]<=max_valid_time, 1:length(ts)),sample_n]
    ŷ₂_s_valid = ŷ₂_s[1,findall(i-> ts[i]<=max_valid_time, 1:length(ts)),sample_n]
    ŷ₂_m_w_valid = ŷ₂_m_w[1,findall(i-> ts_w[i]<=max_valid_time .&& mask₂[1,i,sample_n] == 1, 1:length(ts_w)),sample_n ]
    ŷ₂_s_w_valid = ŷ₂_s_w[1,findall(i-> ts_w[i]<=max_valid_time .&& mask₂[1,i,sample_n] == 1, 1:length(ts_w)),sample_n]
    ŷ₂_m_CI_lower_valid= ŷ₂_m_valid .- 1.96.*ŷ₂_s_valid./sqrt(config["mcmc_samples"])
    ŷ₂_m_CI_upper_valid= ŷ₂_m_valid .+ 1.96.*ŷ₂_s_valid./sqrt(config["mcmc_samples"])
    ŷ₁_class_w_valid = ŷ₁_class_w[findall(i-> ts_w[i]<=max_valid_time .&& mask₁[1,i,sample_n] == 1, 1:length(ts_w)),sample_n]
    ŷ₁_conf_w_valid=ŷ₁_conf_w[1,findall(i-> ts_w[i]<=max_valid_time .&& mask₁[1,i,sample_n] == 1, 1:length(ts_w)),sample_n]
    y₁_class_w_valid= y₁_class[findall(i-> ts_w[i]<=max_valid_time .&& mask₁[1,i,sample_n] == 1, 1:length(ts_w)),sample_n]
    ŷ₂_count_m_w_valid = ŷ₂_count_m_w[1,findall(i-> ts_w[i]<=max_valid_time .&& mask₁[1,i,sample_n] == 1, 1:length(ts_w)),sample_n]
    ŷ₂_count_s_w_valid = ŷ₂_count_s_w[1,findall(i-> ts_w[i]<=max_valid_time .&& mask₁[1,i,sample_n] == 1, 1:length(ts_w)),sample_n]
    y₂_w_valid= y₂[1,findall(i-> ts_w[i]<=max_valid_time .&& mask₁[1,i,sample_n] == 1, 1:length(ts_w)),sample_n]
    ## finding valid interventions (chemothjerapy and radiotherapy)
    valid_indices_chemo = findall(i -> u[1,i, sample_n] == 1 && ts_w[i] <= max_valid_time, 1:length(ts_w))
    valid_indices_radio = findall(i -> u[2,i, sample_n] == 1 && ts_w[i] <= max_valid_time, 1:length(ts_w))

    fig = Figure(size=(1200, 800))
    ax1= CairoMakie.Axis(fig[1, 1], xlabel="Time (Days)", ylabel="Interventions", limits=(nothing, (0, 1.5)), yticks=[0,1])
    ax2 = CairoMakie.Axis(fig[2, 1], xlabel="Time (Days)", ylabel="Health status", limits=(nothing, (-0.5, 6.0)))
    ax3 = CairoMakie.Axis(fig[3, 1], xlabel="Time (Days)", ylabel="Tumor size")
    ax4 = CairoMakie.Axis(fig[4, 1], xlabel="Time (Days)", ylabel="Cell count")

    scatter!(ax1, ts_w[valid_indices_chemo], ones(length(u[valid_indices_chemo])),marker = :utriangle,markersize = 10,color = :blue, label="Chemotherapy regimen")
    scatter!(ax1, ts_w[valid_indices_radio], ones(length(u[valid_indices_radio])),marker = :star5,markersize = 15,color = :red, label="Radiotherapy regimen")
    xlims!(ax1, minimum(ts_valid), maximum(ts_valid)+1)
    
    scatter!(ax2, ts_w_valid, y₁_class_w_valid, color=(atom_one_dark[:blue], 0.9),markersize=20, label="True Health status")
    scatter!(ax2, ts_w_valid, ŷ₁_class_w_valid, color=(atom_one_dark[:red], 0.9),markersize=10, label="Predicted Health status")
    errorbars!(ax2, ts_w_valid,ŷ₁_class_w_valid, ŷ₁_conf_w_valid, color=(atom_one_dark[:red], 0.3),whiskerwidth=8)
    xlims!(ax2, minimum(ts_valid), maximum(ts_valid)+1)

    lines!(ax3, ts_valid, x_valid[1,:], linewidth=2, color=(atom_one_dark[:gray], 1.0), label="True Tumor size (daily)")
    lines!(ax3, ts_valid, ŷ₂_m_valid, linewidth=2, color=(atom_one_dark[:cyan], 0.9), label="Inferred Tumor size (daily)")
    #scatter!(ax3, ts_valid, ŷ₂_m_valid, markersize=5, color=(atom_one_dark[:cyan], 0.9))
    band!(ax3, ts_valid, ŷ₂_m_CI_lower_valid, ŷ₂_m_CI_upper_valid, color=(atom_one_dark[:cyan], 0.3))
    scatter!(ax3, ts_w_valid, ŷ₂_m_w_valid, markersize=10, color=(atom_one_dark[:red], 0.9), label="Inferred Tumor size (weekly, irregular)")
    xlims!(ax3, minimum(ts_valid), maximum(ts_valid)+1)

    scatter!(ax4, ts_w_valid, y₂_w_valid, color=(atom_one_dark[:blue], 0.7),markersize=20, label="True Cell count")
    #scatter!(ax4, ts_w_valid, ŷ₂_count_m_w_valid, color=(atom_one_dark[:red]),markersize=10, label="Inferred Cell count")
    scatter!(ax4, ts_w_valid, ŷ₂_m_w_valid, color=(atom_one_dark[:red]),markersize=10, label="Predicted Cell count")
    errorbars!(ax4, ts_w_valid, ŷ₂_m_w_valid .- 1.96.*ŷ₂_s_w_valid/sqrt(config["mcmc_samples"]), ŷ₂_m_w_valid .+ 1.96.*ŷ₂_s_w_valid/sqrt(config["mcmc_samples"]), color=(atom_one_dark[:red], 0.3),whiskerwidth=8)
    #errorbars!(ax4, ts_w_valid, ŷ₂_count_s_w_valid .- 1.96.*sqrt.(ŷ₂_count_s_w_valid)/config["mcmc_samples"], ŷ₂_count_m_w_valid .+1.96.* sqrt.(ŷ₂_count_s_w_valid)/config["mcmc_samples"], color=(atom_one_dark[:green], 0.8),whiskerwidth=8)
     
     linkxaxes!(ax1, ax2, ax3, ax4)
     # Add legends to all axes
     axislegend(ax1, position=:rb, backgroundcolor=:transparent)
     axislegend(ax2, position=:rt, backgroundcolor=:transparent)
     axislegend(ax3, position=:rt, backgroundcolor=:transparent)
     axislegend(ax4, position=:rt, backgroundcolor=:transparent)
     display(fig)
    
     display(fig)
    return fig
end

  


## prediction
function predict_future(model, θ, st, history, u_p, t_p, config)
    u_o, x_o,covars_o, y₁_o, y₂_o = history
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    Ex, Ey_p = predict(model, solver, vcat(covars_o,reverse(y₁_o, dims=2), reverse(y₂_o, dims=2)), u_p, t_p, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    return Ex, Ey_p[1], Ey_p[2]
end

# visualization of prediction performance (validation)
function vis_fn_pred(observed_time, predict_time, observed_data, future_true_data, predicted_data; sample_n=1)
    u_o, x_o,covars_o, y₁_o, y₂_o, mask₁_o, mask₂_o = observed_data
    u_t, x_t,covars_t, y₁_t, y₂_t, mask₁_t, mask₂_t = future_true_data
    u_p= u_t
    Ex, Ey₁_p, Ey₂_p = predicted_data
    t_o, t_p = observed_time, predict_time
    t_o, t_p = t_o .* 52.0f0*7, t_p .* 52.0f0*7 # Convert time to days
    timepoints = vcat(t_o, t_p)

    #results 
    y₁_p_m = dropmean(Ey₁_p, dims=4)
    y₂_p_m = dropmean(Ey₂_p, dims=4)
    y₂_p_s = dropmean(std(Ey₂_p, dims=4), dims=4)

    y₂_count_p = rand.(Poisson.(Ey₂_p))
    y₂_count_p_m = dropmean(y₂_count_p, dims=4)
    y₂_count_p_s = dropmean(std(y₂_count_p, dims=4), dims=4)

    y₁_o_class = onecold(y₁_o, Array(0:5))
    y₁_t_class = onecold(y₁_t, Array(0:5))
    y₁_p_class = onecold(y₁_p_m, Array(0:5))

    println("y2 MSE: ", MSELoss()(y₂_t , y₂_p_m))

    ##plots 
    valid_indx_o= findall(mask₁_o[1,:,sample_n].==1)
    valid_indx_p= findall(mask₁_t[1,:,sample_n].==1)

    t_o_valid, u_o_valid= t_o[valid_indx_o], u_o[:,valid_indx_o,sample_n]
    t_p_valid, u_p_valid= t_p[valid_indx_p], u_p[:,valid_indx_p,sample_n]
    x_max, x_min, y_min, y_max = t_p[end] + 0.5, -0.5, -2, 45

    fig = Figure(size=(1200, 900))
    ax1 = CairoMakie.Axis(fig[1, 1], xlabel="Time (weeks)", ylabel="Interventions", limits=((x_min, x_max), (0.0, 1.5)), yticks=[0, 1])
    ax2 = CairoMakie.Axis(fig[2, 1], xlabel="Time (weeks)", ylabel="Health status", limits=((x_min, x_max), (-0.5, 5.5)))
    ax3 = CairoMakie.Axis(fig[3, 1], xlabel="Time (weeks)", ylabel="Tumor size", limits=((x_min, x_max), (y_min, 50)))
    ax4 = CairoMakie.Axis(fig[4, 1], xlabel="Time (weeks)", ylabel="Cell count", limits=((x_min, x_max), (y_min, 50)))

    chemo_times_o, radio_times_o = t_o_valid[u_o_valid[1, :] .> 0], t_o_valid[u_o_valid[2, :] .> 0]
    chemo_times_p, radio_times_p = t_p_valid[u_p_valid[1, :] .> 0], t_p_valid[u_p_valid[2, :] .> 0]

    scatter!(ax1, chemo_times_o, fill(1, length(chemo_times_o)), color=:darkgreen, marker=:utriangle, markersize=15, label="Chemotherapy session")
    scatter!(ax1, radio_times_o, fill(1, length(radio_times_o)), color=:darkorange, marker=:star5, markersize=15, label="Radiotherapy session")
    scatter!(ax1, chemo_times_p, fill(1, length(chemo_times_p)), color=:darkgreen, marker=:utriangle, markersize=15)
    scatter!(ax1, radio_times_p, fill(1, length(radio_times_p)), color=:darkorange, marker=:star5, markersize=15)

    scatter!(ax2, t_o_valid, y₁_o_class[valid_indx_o, sample_n], color=:red, markersize=15, label="Health status class(observed)")
    scatter!(ax2, t_p_valid, y₁_t_class[valid_indx_p, sample_n], color=(:red, 0.5), markersize=15, label="True Health status class(future)")
    scatter!(ax2, t_p_valid, y₁_p_class[valid_indx_p, sample_n], color=(:dodgerblue2, 0.7), markersize=10, label="Predicted Health status class")

    lines!(ax3, Array(1:t_o[end]), x_o[1, :, sample_n], linewidth=2, color=:red, label="Tumor size(observed)")
    lines!(ax3, Array(t_o[end]+1:t_p[end]), x_t[1, :, sample_n], linewidth=2, color=(:red, 0.5), linestyle=:dash, label="True Tumor size (future)")
    lines!(ax3, t_p, y₂_p_m[1, :, sample_n], color=(:dodgerblue2, 0.5), linewidth=2)
    band!(ax3, t_p, y₂_p_m[1, :,sample_n] .- sqrt.(y₂_p_s[1, :,sample_n]), y₂_p_m[1, :,sample_n] .+ sqrt.(y₂_p_s[1, :,sample_n]), color=(:dodgerblue2, 0.5), label="Inferred Tumor size (future)")

    scatter!(ax4, t_o_valid, y₂_o[1, valid_indx_o, sample_n], color=:red, markersize=15, label="Cell count (observed)")
    scatter!(ax4, t_p_valid, y₂_t[1, valid_indx_p, sample_n], color=(:red, 0.5), markersize=15, label="True Cell count (future)")
    scatter!(ax4, t_p_valid, y₂_count_p_m[1, valid_indx_p,sample_n], color=(:dodgerblue2, 0.7), markersize=10, label="Predicted Cell count")
    errorbars!(ax4, t_p_valid, y₂_count_p_m[1, valid_indx_p,sample_n], y₂_count_p_s[1, valid_indx_p,sample_n], color=(:dodgerblue2, 0.7), whiskerwidth=8)

    linkxaxes!(ax1, ax2, ax3, ax4)

    poly!(ax1, [-0.5, length(t_o)*7-1, length(t_o)*7-1, -0.5], [0, 0, 50, 50], color=(:blue, 0.1), label="observation period (history)")
    poly!(ax1, [length(t_o)*7-1, length(t_o)*7 + length(t_p)*7, length(t_o)*7 + length(t_p), length(t_o)-1], [0, 0, 50, 50], color=(:red, 0.1), label="prediction period (future)")
    poly!(ax2, [-0.5, length(t_o)*7-1, length(t_o)*7-1, -0.5], [-0.5, -0.5, 50, 50], color=(:blue, 0.1))
    poly!(ax2, [length(t_o)*7-1, length(t_o)*7 + length(t_p)*7, length(t_o)*7 + length(t_p)*7, length(t_o)*7-1], [-0.5, -0.5, 50, 50], color=(:red, 0.1))
    poly!(ax3, [-0.5, length(t_o)*7-1, length(t_o)*7-1, -0.5], [-2, -2, 50, 50], color=(:blue, 0.1))
    poly!(ax3, [length(t_o)*7-1, length(t_o)*7 + length(t_p)*7, length(t_o)*7 + length(t_p)*7, length(t_o)*7-1], [-2, -2, 50, 50], color=(:red, 0.1))
    poly!(ax4, [-0.5, length(t_o)*7-1, length(t_o)*7-1, -0.5], [-2, -2, 50, 50], color=(:blue, 0.1))
    poly!(ax4, [length(t_o)*7-1, length(t_o)*7 + length(t_p)*7, length(t_o)*7 + length(t_p)*7, length(t_o)*7-1], [-2, -2, 50, 50], color=(:red, 0.1))

    fig[1, 2] = Legend(fig, ax1, framevisible=false)
    fig[2, 2] = Legend(fig, ax2, framevisible=false)
    fig[3, 2] = Legend(fig, ax3, framevisible=false)
    fig[4, 2] = Legend(fig, ax4, framevisible=false)

    display(fig)
    return fig
end

## system identification 
rng = Random.MersenneTwister(123);
train_loader, test_loader, val_loader, dims, timepoints, covars = generate_dataloader(; n_samples=256, batchsize=64, split=(0.5,0.3));
config = YAML.load_file("./configs/PkPD_config.yml");
exp_path = joinpath(config["experiment"]["path"], config["experiment"]["name"])
isdir(exp_path) ? exp_path : mkpath(exp_path)
model, θ, st = create_latentsde(config["model"], dims, rng);
θ_trained = train(model, θ_trained, st, timepoints, loss_fn, eval_fn, viz_fn_sys_id, train_loader, test_loader, config["training"], exp_path);

## visualization of the system identification
fig=viz_fn_sys_id(model, θ_trained, st, Array(0:365)/365, first(val_loader), config["training"]["validation"]; sample_n=5);
#save(joinpath(exp_path, "results_system_id_.pdf"), fig);

# validation of model prediction performance
u, x,covars, y₁, y₂, mask₁, mask₂= first(val_loader);
#x=x[:,1:7:end,:];
spl=5
ind_observed = 1:spl; ind_predict = spl+1:length(timepoints);
observed_data = (u[:,ind_observed,:], x[:,1:(ind_observed[end]-1)*7,:] ,covars[:,ind_observed,:], y₁[:,ind_observed,:], y₂[:,ind_observed,:], mask₁[:,ind_observed,:], mask₂[:,ind_observed,:]);
future_true_data = (u[:,ind_predict,:],x[:,(ind_observed[end]-1)*7+1:364,:],covars[:,ind_predict,:], y₁[:,ind_predict,:], y₂[:,ind_predict,:], mask₁[:,ind_predict,:], mask₂[:,ind_predict,:]);
observed_time = timepoints[ind_observed]; predict_time = timepoints[ind_predict];
predict_u= u[:,ind_predict,:];
Ex, Ey₁, Ey₂ = predict_future(model, θ_trained, st, observed_data, predict_u ,predict_time , config["training"]["validation"]);
predicted_data = (Ex, Ey₁, Ey₂);

fig=vis_fn_pred(observed_time, predict_time, observed_data, future_true_data, predicted_data; sample_n=1);
#save(joinpath(exp_path, "results_prediction.pdf"), fig)

u, x, covars, y₁, y₂, mask₁, mask₂= first(val_loader);

ŷ, px₀, kl_pq = model(vcat(covars,y₁, y₂), u, timepoints, θ, st)
ŷ₁, ŷ₂ = ŷ
val_indx₁= findall(mask₁.==1)
val_indx₂= findall(mask₂.==1)

recon_loss = CrossEntropyLoss()(ŷ₁[val_indx₁], y₁[val_indx₁]) - poisson_loglikelihood(ŷ₂[val_indx₂], y₂[val_indx₂])
kl_loss = kl_normal(px₀...) / size(x)[end] + mean(kl_pq[end, :])
loss = recon_loss + 0.1 * kl_loss