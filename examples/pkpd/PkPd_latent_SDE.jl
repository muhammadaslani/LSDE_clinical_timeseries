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
    return fig
end


## prediction
function predict_future(model, θ, st, observed_data, predict_u, predict_time, config)
    u_o, x_o,covars_o, y₁_o, y₂_o = observed_data
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    Ex, Ey_p = predict(model, solver, vcat(covars_o,reverse(y₁_o, dims=2), reverse(y₂_o, dims=2)), predict_u, predict_time, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    return Ex, Ey_p[1], Ey_p[2]
end

# visualization of prediction performance (validation)
function vis_fn_pred(observed_time, predict_timepoints, observed_data, future_true_data, predicted_data; sample_n=1)
    u_o, x_o,covars_o, y₁_o, y₂_o, mask₁_o, mask₂_o = observed_data
    u_t, x_t,covars_t, y₁_t, y₂_t, mask₁_t, mask₂_t = future_true_data
    u_p= u_t
    Ex, Ey₁_p, Ey₂_p = predicted_data
    t_o, t_p = observed_time, predict_timepoints
    t_o_d= t_o .* 52.0f0*7 # Convert time to days from normalized values but sampled weekly
    ts_o_d=Array(1:t_o_d[end])
    t_p_d= t_p .* length(t_p) # Convert back to days from normalized values
    t_p_w= t_p_d[1:7:end]
    #ts_d = vcat(t_o_d, t_p_d)

    #results 
    ŷ₁_m = dropmean(Ey₁_p, dims=4)
    ŷ₂_m = dropmean(Ey₂_p, dims=4)
    ŷ₂_s = dropmean(std(Ey₂_p, dims=4), dims=4)

    ŷ₂_count = rand.(Poisson.(Ey₂_p))
    ŷ₂_count_m = dropmean(ŷ₂_count, dims=4)
    ŷ₂_count_s = dropmean(std(ŷ₂_count, dims=4), dims=4)

    #sampling prediction weekly 
    ŷ₁_m_w = ŷ₁_m[:, 1:7:end, :]
    ŷ₂_m_w = ŷ₂_m[:, 1:7:end, :]
    ŷ₂_s_w = ŷ₂_s[:, 1:7:end, :]
    ŷ₂_count_m_w = ŷ₂_count_m[:, 1:7:end, :]
    ŷ₂_count_s_w = ŷ₂_count_s[:, 1:7:end, :]
    y₁_o_class = onecold(y₁_o, Array(0:5))
    y₁_t_class = onecold(y₁_t, Array(0:5))
    ŷ₁_class_w = onecold(ŷ₁_m_w, Array(0:5))
    ## max time for observed and predicted data
    max_t_o= maximum(t_o_d)
    max_t_p= maximum(t_p_w)
    max_t_o_valid= maximum(t_o_d[mask₂_o[1,:,sample_n] .== 1])
    max_t_p_valid= maximum(t_p_w[mask₂_t[1,:,sample_n] .== 1])
    t_o_d_valid= t_o_d[findall(i-> t_o_d[i]<=max_t_o_valid .&& mask₁_o[1,i,sample_n] == 1, 1:length(t_o_d))]
    t_p_w_valid= t_p_w[findall(i-> t_p_w[i]<=max_t_p_valid .&& mask₁_t[1,i,sample_n] == 1, 1:length(t_p_w))]
    y₁_o_class_valid=y₁_o_class[findall(i-> t_o_d[i]<=max_t_o_valid .&& mask₁_o[1,i,sample_n] == 1, 1:length(t_o_d)),sample_n]
    y₁_t_class_valid=y₁_t_class[findall(i-> t_p_w[i]<=max_t_p_valid .&& mask₁_t[1,i,sample_n] == 1, 1:length(t_p_w)),sample_n]
    ŷ₁_class_w_valid=ŷ₁_class_w[findall(i-> t_p_w[i]<=max_t_p_valid .&& mask₁_t[1,i,sample_n] == 1, 1:length(t_p_w)),sample_n]
    y₂_o_valid=y₂_o[1,findall(i-> t_o_d[i]<=max_t_o_valid .&& mask₂_o[1,i,sample_n] == 1, 1:length(t_o_d)),sample_n]
    y₂_t_valid=y₂_t[1,findall(i-> t_p_w[i]<=max_t_p_valid .&& mask₂_t[1,i,sample_n] == 1, 1:length(t_p_w)),sample_n]
    ŷ₂_m_w_valid=ŷ₂_m_w[1,findall(i-> t_p_w[i]<=max_t_p_valid .&& mask₂_t[1,i,sample_n] == 1, 1:length(t_p_w)),sample_n]
    ŷ₂_s_w_valid=ŷ₂_s_w[1,findall(i-> t_p_w[i]<=max_t_p_valid .&& mask₂_t[1,i,sample_n] == 1, 1:length(t_p_w)),sample_n]
    ŷ₂_count_m_w_valid=ŷ₂_count_m_w[1,findall(i-> t_p_w[i]<=max_t_p_valid .&& mask₂_t[1,i,sample_n] == 1, 1:length(t_p_w)),sample_n]
    ŷ₂_count_s_w_valid=ŷ₂_count_s_w[1,findall(i-> t_p_w[i]<=max_t_p_valid .&& mask₂_t[1,i,sample_n] == 1, 1:length(t_p_w)),sample_n]
    #chemptherapy and radiotherapy sessions 
    valid_indices_chemo_o = findall(i -> u_o[1,i, sample_n] == 1 && t_o_d[i] <= max_t_o, 1:length(t_o_d))
    valid_indices_radio_o = findall(i -> u_o[2,i, sample_n] == 1 && t_o_d[i] <= max_t_o, 1:length(t_o_d))
    valid_indices_chemo_p = findall(i -> u_p[1,i, sample_n] == 1 && t_p_d[i] <= max_t_p, 1:length(t_p_w))
    valid_indices_radio_p = findall(i -> u_p[2,i, sample_n] == 1 && t_p_d[i] <= max_t_p, 1:length(t_p_w))

    #plotting
    y_max₁, y_max₂, y_max₃=maximum(ŷ₂_count_m_w_valid), maximum(y₂_o_valid), maximum(y₂_t_valid)
    y₂_count_max=maximum([y_max₁, y_max₂, y_max₃])+maximum([y_max₁, y_max₂, y_max₃])/10
    x_min, x_max= -0.5, max_t_p_valid+max_t_p_valid/50
    y_min, y_max=  -0.5, y₂_count_max+1

    fig = Figure(size=(1200, 900))
    ax1 = CairoMakie.Axis(fig[1, 1], xlabel="Time (days)", ylabel="Interventions",limits=((x_min, x_max), (0.0, 1.5)),  yticks=[0, 1],xgridvisible = false, ygridvisible = false)
    ax2 = CairoMakie.Axis(fig[2, 1], xlabel="Time (days)", ylabel="Health status",limits=((x_min, x_max), (-0.5, 5.5)),xgridvisible = false, ygridvisible = false)
    ax3 = CairoMakie.Axis(fig[3, 1], xlabel="Time (days)", ylabel="Tumor size",limits=((x_min, x_max), (y_min, y_max)),xgridvisible = false, ygridvisible = false)
    ax4 = CairoMakie.Axis(fig[4, 1], xlabel="Time (days)", ylabel="Cell count",limits=((x_min, x_max), (y_min, y_max)),xgridvisible = false, ygridvisible = false)

    scatter!(ax1, t_o_d[valid_indices_chemo_o], ones(length(u_o[valid_indices_chemo_o])),marker = :utriangle,markersize = 10,color = :blue, label="Chemotherapy regimen")
    scatter!(ax1, t_o_d[valid_indices_radio_o], ones(length(u_o[valid_indices_radio_o])),marker = :star5,markersize = 15,color = :red, label="Radiotherapy regimen")
    scatter!(ax1, t_p_w[valid_indices_chemo_p], ones(length(u_p[valid_indices_chemo_p])),marker = :utriangle,markersize = 10,color = :blue)
    scatter!(ax1, t_p_w[valid_indices_radio_p], ones(length(u_p[valid_indices_radio_p])),marker = :star5,markersize = 15,color = :red)

    scatter!(ax2, t_o_d_valid, y₁_o_class_valid, color = :blue, label="Observed")
    scatter!(ax2, t_p_w_valid, y₁_t_class_valid, color = :green, label="True")
    scatter!(ax2, t_p_w_valid, ŷ₁_class_w_valid, color = :red, label="Predicted")

    lines!(ax3, Array(1:366), vcat(x_o[1,:, sample_n], x_t[1,:, sample_n]), color = :blue, label="Observed")
    #lines!(ax3, t_p_d, x_t[1,:, sample_n], color = :blue, label="True")
    lines!(ax3, t_p_d, ŷ₂_m[1,:, sample_n], color = :red, label="Predicted (daily)")
    scatter!(ax3, t_p_w_valid, ŷ₂_m_w_valid, color = :red, label="Predicted (weekly irregular)")

    scatter!(ax4, t_o_d_valid, y₂_o_valid, color = :blue, label="Observed")
    scatter!(ax4, t_p_w_valid, y₂_t_valid, color = :green, label="True")
    scatter!(ax4, t_p_w_valid, ŷ₂_count_m_w_valid, color = :red, label="Predicted")

    linkxaxes!(ax1, ax2, ax3, ax4)

    poly!(ax1, [-10, length(t_o)*7-1, length(t_o)*7-1, -10], [-10, -10, 500, 500], color=(:blue, 0.1), label="observation period (history)")
    poly!(ax1, [length(t_o)*7-1, length(t_o)*7 + length(t_p)*7, length(t_o)*7 + length(t_p)*7, length(t_o)*7-1], [-10, -10, 500, 500], color=(:red, 0.1), label="prediction period (future)")
    poly!(ax2, [-10, length(t_o)*7-1, length(t_o)*7-1, -10], [-10, -10, 500, 500], color=(:blue, 0.1))
    poly!(ax2, [length(t_o)*7-1, length(t_o)*7 + length(t_p)*7, length(t_o)*7 + length(t_p)*7, length(t_o)*7-1], [-10,-10, 500, 50], color=(:red, 0.1))
    poly!(ax3, [-10, length(t_o)*7-1, length(t_o)*7-1, -10], [-10, -10, 500, 500], color=(:blue, 0.1))
    poly!(ax3, [length(t_o)*7-1, length(t_o)*7 + length(t_p)*7, length(t_o)*7 + length(t_p)*7, length(t_o)*7-1], [-10, -10, 500, 500], color=(:red, 0.1))
    poly!(ax4, [-10, length(t_o)*7-1, length(t_o)*7-1, -10], [-10, -10, 500, 500], color=(:blue, 0.1))
    poly!(ax4, [length(t_o)*7-1, length(t_o)*7 + length(t_p)*7, length(t_o)*7 + length(t_p)*7, length(t_o)*7-1], [-10, -10, 500, 500], color=(:red, 0.1))

    fig[1, 2] = Legend(fig, ax1, framevisible=false)
    fig[2, 2] = Legend(fig, ax2, framevisible=false)
    fig[3, 2] = Legend(fig, ax3, framevisible=false)
    fig[4, 2] = Legend(fig, ax4, framevisible=false)
    display(fig)
    return fig
end

## system identification 
rng = Random.MersenneTwister(123);
#train_loader, test_loader, val_loader, dims, timepoints, covars = generate_dataloader(; n_samples=256, batchsize=64, split=(0.5,0.3));

#latent SDE
config_lsde = YAML.load_file("./configs/PkPD_config_LSDE.yml");
exp_path = joinpath(config["experiment"]["path"], config["experiment"]["name"])
isdir(exp_path) ? exp_path : mkpath(exp_path)
lsde_model, lsde_θ, lsde_st = create_latentsde(config_LSDE["model"], dims, rng);
lsde_θ_trained = train(lsde_model, lsde_θ, lsde_st, timepoints, loss_fn, eval_fn, viz_fn_sys_id, train_loader, test_loader, config_lsde["training"], exp_path);

#latent ODE
config_lode = YAML.load_file("./configs/PkPD_config_LODE.yml");
lode_model, lode_θ, lode_st = create_latentsde(config_lode["model"], dims, rng);
lode_θ_trained = train(lode_model, lode_θ, lode_st, timepoints, loss_fn, eval_fn, viz_fn_sys_id, train_loader, test_loader, config_lode["training"], exp_path);

## visualization of the system identification
#lsde
fig=viz_fn_sys_id(lsde_model, lsde_θ_trained, lsde_st, Array(0:365)/365, first(val_loader), config_lsde["training"]["validation"]; sample_n=2);
#lode
fig=viz_fn_sys_id(lode_model, lode_θ_trained, lode_st, Array(0:365)/365, first(val_loader), config_lode["training"]["validation"]; sample_n=2);

#save(joinpath(exp_path, "results_system_id_.pdf"), fig);

# validation of model prediction performance
u, x,covars, y₁, y₂, mask₁, mask₂= first(val_loader);
#x=x[:,1:7:end,:];
spl=10
ind_observed = 1:spl; ind_predict = spl+1:length(timepoints)
observed_data = (u[:,ind_observed,:], x[:,1:spl*7,:] ,covars[:,ind_observed,:], y₁[:,ind_observed,:], y₂[:,ind_observed,:], mask₁[:,ind_observed,:], mask₂[:,ind_observed,:]);
future_true_data = (u[:,ind_predict,:],x[:,spl*7+1:end,:],covars[:,ind_predict,:], y₁[:,ind_predict,:], y₂[:,ind_predict,:], mask₁[:,ind_predict,:], mask₂[:,ind_predict,:]);
observed_time = timepoints[ind_observed];
predict_time = timepoints[ind_predict]; #weekly
predict_time_d= Array(predict_time[1] .*52.0f0.*7:365); #daily
predict_timepoints=Array(predict_time_d)/length(predict_time_d);
predict_u= u[:,ind_predict,:];

#lsde
lsde_Ex, lsde_Ey₁, lsde_Ey₂ = predict_future(lsde_model, lsde_θ_trained, lsde_st, observed_data, predict_u ,predict_timepoints , config_lsde["training"]["validation"]);
lsde_predicted_data = (lsde_Ex, lsde_Ey₁, lsde_Ey₂);
fig=vis_fn_pred(observed_time, predict_timepoints, observed_data, future_true_data, lsde_predicted_data; sample_n=3);

#lode
lode_Ex, lode_Ey₁, lode_Ey₂ = predict_future(lode_model, lode_θ_trained, lode_st, observed_data, predict_u ,predict_timepoints , config_lode["training"]["validation"]);
lode_predicted_data = (lode_Ex, lode_Ey₁, lode_Ey₂);
fig=vis_fn_pred(observed_time, predict_timepoints, observed_data, future_true_data, lode_predicted_data; sample_n=7);
#save(joinpath(exp_path, "results_prediction.pdf"), fig)