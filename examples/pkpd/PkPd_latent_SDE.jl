using  Revise, Rhythm, Lux, DifferentialEquations, Random, SciMLSensitivity, ComponentArrays, Optimisers, OptimizationOptimisers, Statistics
using MLUtils, Printf, SciMLSensitivity, OneHotArrays
using YAML
include("pkpd_standalone.jl")

function generate_dataloader(; n_samples=512, batchsize=64, split=(0.5,0.3))
    U, X, Y₁, Y₂, T, covariates = generate_dataset(n_samples=n_samples)
    Y₁_padded, Masks₁, timepoints = pad_matrices(Y₁, T)
    Y₂_padded, Masks₂, _ = pad_matrices(Y₂, T)
    X_padded, _ = pad_matrices(X, T; return_timepoints=false)
    #Y₁_irreg, Y₂_irreg, Masks₁, Masks₂ = irregularize(Y₁_padded,Y₂_padded, Masks₁, Masks₂)
    timepoints /= (7.0f0 * 52.0f0)  # Normalize timepoints
    covars=repeat(reshape(covariates,2,1,size(covariates,2)),1,size(Y₁_padded)[2],1)
    U = cat(U..., dims=3)
    data = (U, X_padded, covars, Y₁_padded, Y₂_padded, Masks₁, Masks₂)
    (train_data,test_data, val_data) = splitobs(data, at=split)

    train_loader = DataLoader(train_data, batchsize=batchsize, shuffle=true)
    test_loader = DataLoader(test_data, batchsize=batchsize, shuffle=true)
    val_loader = DataLoader(val_data, batchsize=batchsize, shuffle=false)
    dims = Dict(
        "obs_dim" => [size(covars,1),size(Y₁_padded, 1), size(Y₂_padded, 1)],
        "input_dim" => size(U, 1),
        "state_dim" => size(X_padded, 1),
        "output_dim" => [size(Y₁_padded, 1), size(Y₂_padded, 1)]
    )

    return train_loader, test_loader, val_loader, dims, timepoints, covars
end

function loss_fn(model, θ, st, data)
    (u, x,covars, y₁, y₂, mask₁, mask₂), ts, λ = data
    ŷ, px₀, kl_pq = model(vcat(covars,y₁, y₂), u, ts, θ, st)
    ŷ₁, ŷ₂ = ŷ
    recon_loss = CrossEntropy_Loss(ŷ₁, y₁, mask₁) - poisson_loglikelihood(ŷ₂, y₂, mask₂)
    kl_loss = kl_normal(px₀...) / size(x)[end] + mean(kl_pq[end, :])
    loss = recon_loss + λ * kl_loss
    return loss, st, kl_loss
end

function eval_fn(model, θ, st, ts, data, config)
    u, x,covars, y₁, y₂, mask₁, mask₂ = data
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    px₀ = (zeros32(config["latent_dim"], size(y₁)[end]), ones32(config["latent_dim"], size(y₁)[end]))
    Ex, Ey = generate(model, solver, px₀, u, ts, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    ŷ₁_m, ŷ₂_m = dropmean(Ey[1], dims=4), dropmean(Ey[2], dims=4)
    return CrossEntropy_Loss(ŷ₁_m, y₁, mask₁) - poisson_loglikelihood(ŷ₂_m, y₂, mask₂)
end


function viz_fn_sys_id(model, θ, st, ts, data, config; sample_n=1)
    u, x, covars, y₁, y₂, mask₁, mask₂ = data
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    px₀ = (zeros32(config["latent_dim"], size(y₁)[end]), ones32(config["latent_dim"], size(y₁)[end]))
    Ex, Ey = generate(model, solver, px₀, u, ts, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    Ey₁, Ey₂ = Ey   # Ey₁ is the predicted porbability for health status classes, Ey₂ is the predicted tumor size

    ts = ts .* 52.0f0
    # Apply mask to Ey
    Ey₁_masked = Ey₁ .* mask₁
    Ey₂_masked = Ey₂ .* mask₂
    # Find valid indices
    valid_indices = findall(mask₁[1, :, sample_n] .== 1)
    valid_ts = ts[valid_indices]
    valid_y₁ = y₁[:, valid_indices, sample_n]
    valid_y₂ = y₂[1, valid_indices, sample_n]

    fig = Figure(size=(1200, 900))
    ax1 = CairoMakie.Axis(fig[1, 1], xlabel="Time (weeks)", ylabel="Interventions", limits=(nothing, (0, 1.5)), yticks=[0, 1])
    ax2 = CairoMakie.Axis(fig[2, 1], xlabel="Time (weeks)", ylabel="Health status", limits=(nothing, (-0.5, 6.0)))
    ax3 = CairoMakie.Axis(fig[3, 1], xlabel="Time (weeks)", ylabel="Tumor size")
    ax4 = CairoMakie.Axis(fig[4, 1], xlabel="Time (weeks)", ylabel="Cell count")

    chemo_times = ts[u[1, :, sample_n].>0]
    radio_times = ts[u[2, :, sample_n].>0]

    scatter!(ax1, chemo_times, fill(1, length(chemo_times)), color=atom_one_dark[:red], marker=:utriangle, markersize=20, label="Chemotherapy session")
    scatter!(ax1, radio_times, fill(1, length(radio_times)), color=atom_one_dark[:yellow], marker=:star5, markersize=15, label="Radiotherapy session")
    ŷ₁_m = selectdim(dropmean(Ey₁_masked, dims=4), 3, sample_n)
    ŷ₁_s = selectdim(dropmean(std(Ey₁_masked, dims=4), dims=4), 3, sample_n)
    ŷ₂_m = selectdim(dropmean(Ey₂_masked, dims=4), 3, sample_n)
    ŷ₂_s = selectdim(dropmean(std(Ey₂_masked, dims=4), dims=4), 3, sample_n)
    #estimate cell counts from Inferred tumor size
    Ey₂_count = rand.(Poisson.(Ey₂_masked))
    ŷ₂_m_count = selectdim(dropmean(Ey₂_count, dims=4), 3, sample_n)
    ŷ₂_s_count = selectdim(dropmean(std(Ey₂_count, dims=4), dims=4), 3, sample_n)


    # getting valid values for each obersvation and prediction 
    valid_x = x[:, 1:7:end, :][:, valid_indices, sample_n]
    valid_ŷ₁_m = ŷ₁_m[:, valid_indices]
    valid_ŷ₁_s = ŷ₁_s[:, valid_indices]
    valid_ŷ₂_m = ŷ₂_m[1, valid_indices]
    valid_ŷ₂_s = ŷ₂_s[1, valid_indices]
    valid_ŷ₂_count_m = ŷ₂_m_count[1, valid_indices]
    valid_ŷ₂_count_s = ŷ₂_s_count[1, valid_indices]
    # one cold encoding for health status
    valid_ŷ₁_class = onecold(valid_ŷ₁_m, Array(0:5))
    valid_y₁_class = onecold(valid_y₁, Array(0:5))

    scatter!(ax2, valid_ts, valid_y₁_class, color=atom_one_dark[:purple], label="Observed health status", markersize=15)
    scatter!(ax2, valid_ts, valid_ŷ₁_class, color=(atom_one_dark[:cyan], 0.5), label="Predicted health status", markersize=10)

    lines!(ax3, valid_ts, valid_x[1,:], linewidth=2, color=(atom_one_dark[:purple], 0.7), label="True Tumor size (unobserved)")
    lines!(ax3, valid_ts, valid_ŷ₂_m, linewidth=2, color=(atom_one_dark[:cyan], 0.5))
    band!(ax3, valid_ts, valid_ŷ₂_m .- sqrt.(valid_ŷ₂_s), valid_ŷ₂_m .+ sqrt.(valid_ŷ₂_s), color=(atom_one_dark[:cyan], 0.5), label="Inferred Tumor size")

    scatter!(ax4, valid_ts, valid_y₂, color=atom_one_dark[:purple], label="Observed tumor cell count", markersize=15)
    scatter!(ax4, valid_ts, valid_ŷ₂_count_m, color=(atom_one_dark[:cyan], 0.5), label="Predicted tumor cell count", markersize=10)
    errorbars!(ax4, valid_ts, valid_ŷ₂_count_m, valid_ŷ₂_count_s, color=(atom_one_dark[:cyan], 0.5), whiskerwidth=8)

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
function predict_future(model, θ, st, history, u_p, t_p, config)
    u_o, x_o, y₁_o, y₂_o = history
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    Ex, Ey_p = predict(model, solver, vcat(reverse(y₁_o, dims=2), reverse(y₂_o, dims=2)), u_p, t_p, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    return Ex, Ey_p[1], Ey_p[2]
end

# visualization of prediction performance (validation)
function validate_predictions(model, θ, st, t_o, t_p, history, u_p, ground_truth, predictions; sample_n=1)
    u_o, x_o, y₁_o, y₂_o = history
    u_t, x_t, y₁_t, y₂_t, mask₁_t, mask₂_t = ground_truth

    Ex, Ey₁_p, Ey₂_p = predictions
    t_p, t_o = t_p .* 52.0f0, t_o .* 52.0f0
    ts = vcat(t_o, t_p)
    valid_indices = findall(mask₁_t[1, :, sample_n] .== 1)

    u_p_valid, t_p_valid = u_p[:, valid_indices, sample_n], t_p[valid_indices]
    x_max, x_min, y_min, y_max = t_o[end] + valid_indices[end] + 0.5, -0.5, -2, 45

    fig = Figure(size=(1200, 900))
    ax1 = CairoMakie.Axis(fig[1, 1], xlabel="Time (weeks)", ylabel="Interventions", limits=((x_min, x_max), (0.0, 1.5)), yticks=[0, 1])
    ax2 = CairoMakie.Axis(fig[2, 1], xlabel="Time (weeks)", ylabel="Health status", limits=((x_min, x_max), (-0.5, 5.5)))
    ax3 = CairoMakie.Axis(fig[3, 1], xlabel="Time (weeks)", ylabel="Tumor size", limits=((x_min, x_max), (y_min, 50)))
    ax4 = CairoMakie.Axis(fig[4, 1], xlabel="Time (weeks)", ylabel="Cell count", limits=((x_min, x_max), (y_min, 50)))

    chemo_times_o, radio_times_o = t_o[u_o[1, :, sample_n] .> 0], t_o[u_o[2, :, sample_n] .> 0]
    chemo_times_p, radio_times_p = t_p_valid[u_p_valid[1, :] .> 0], t_p_valid[u_p_valid[2, :] .> 0]

    scatter!(ax1, chemo_times_o, fill(1, length(chemo_times_o)), color=:darkgreen, marker=:utriangle, markersize=15, label="Chemotherapy session")
    scatter!(ax1, radio_times_o, fill(1, length(radio_times_o)), color=:darkorange, marker=:star5, markersize=15, label="Radiotherapy session")
    scatter!(ax1, chemo_times_p, fill(1, length(chemo_times_p)), color=:darkgreen, marker=:utriangle, markersize=15)
    scatter!(ax1, radio_times_p, fill(1, length(radio_times_p)), color=:darkorange, marker=:star5, markersize=15)

    ŷ₁_m = selectdim(dropmean(Ey₁_p, dims=4), 3, sample_n)
    ŷ₂_m = selectdim(dropmean(Ey₂_p, dims=4), 3, sample_n)
    ŷ₂_s = selectdim(dropmean(std(Ey₂_p, dims=4), dims=4), 3, sample_n)

    ŷ₂_count = rand.(Poisson.(Ey₂_p))
    ŷ₂_m_count = selectdim(dropmean(ŷ₂_count, dims=4), 3, sample_n)
    ŷ₂_s_count = selectdim(dropmean(std(ŷ₂_count, dims=4), dims=4), 3, sample_n)

    y₁_o_class = onecold(y₁_o, Array(0:5))
    y₁_t_class = onecold(y₁_t, Array(0:5))
    ŷ₁_class = onecold(ŷ₁_m, Array(0:5))

    scatter!(ax2, t_o, y₁_o_class[:, sample_n], color=:red, markersize=15, label="Health status class(observed)")
    scatter!(ax2, t_p_valid, y₁_t_class[valid_indices, sample_n], color=(:red, 0.5), markersize=15, label="True Health status class(future)")
    scatter!(ax2, t_p_valid, ŷ₁_class[valid_indices], color=(:dodgerblue2, 0.7), markersize=10, label="Predicted Health status class")
    lines!(ax3, t_o, x_o[1, :, sample_n], linewidth=2, color=:red, label="Tumor size(observed)")
    lines!(ax3, t_p_valid, x_t[1, valid_indices, sample_n], linewidth=2, color=(:red, 0.5), linestyle=:dash, label="True Tumor size (future)")
    lines!(ax3, t_p_valid, ŷ₂_m[1, valid_indices], color=(:dodgerblue2, 0.5), linewidth=2)
    band!(ax3, t_p_valid, ŷ₂_m[1, valid_indices] .- sqrt.(ŷ₂_s[1, valid_indices]), ŷ₂_m[1, valid_indices] .+ sqrt.(ŷ₂_s[1, valid_indices]), color=(:dodgerblue2, 0.5), label="Inferred Tumor size (future)")

    scatter!(ax4, t_o, y₂_o[1, :, sample_n], color=:red, markersize=15, label="Cell count (observed)")
    scatter!(ax4, t_p_valid, y₂_t[1, valid_indices, sample_n], color=(:red, 0.5), markersize=15, label="True Cell count (future)")
    scatter!(ax4, t_p_valid, ŷ₂_m_count[1, valid_indices], color=(:dodgerblue2, 0.7), markersize=10, label="Predicted Cell count")
    errorbars!(ax4, t_p_valid, ŷ₂_m_count[1, valid_indices], ŷ₂_s_count[1, valid_indices], color=(:dodgerblue2, 0.7), whiskerwidth=8)

    linkxaxes!(ax1, ax2, ax3, ax4)

    poly!(ax1, [-0.5, length(t_o)-1, length(t_o)-1, -0.5], [0, 0, 50, 50], color=(:blue, 0.1), label="observation period (history)")
    poly!(ax1, [length(t_o)-1, length(t_o) + length(t_p), length(t_o) + length(t_p), length(t_o)-1], [0, 0, 50, 50], color=(:red, 0.1), label="prediction period (future)")
    poly!(ax2, [-0.5, length(t_o)-1, length(t_o)-1, -0.5], [-0.5, -0.5, 50, 50], color=(:blue, 0.1))
    poly!(ax2, [length(t_o)-1, length(t_o) + length(t_p), length(t_o) + length(t_p), length(t_o)-1], [-0.5, -0.5, 50, 50], color=(:red, 0.1))
    poly!(ax3, [-0.5, length(t_o)-1, length(t_o)-1, -0.5], [-2, -2, 50, 50], color=(:blue, 0.1))
    poly!(ax3, [length(t_o)-1, length(t_o) + length(t_p), length(t_o) + length(t_p), length(t_o)-1], [-2, -2, 50, 50], color=(:red, 0.1))
    poly!(ax4, [-0.5, length(t_o)-1, length(t_o)-1, -0.5], [-2, -2, 50, 50], color=(:blue, 0.1))
    poly!(ax4, [length(t_o)-1, length(t_o) + length(t_p), length(t_o) + length(t_p), length(t_o)-1], [-2, -2, 50, 50], color=(:red, 0.1))

    fig[1, 2] = Legend(fig, ax1, framevisible=false)
    fig[2, 2] = Legend(fig, ax2, framevisible=false)
    fig[3, 2] = Legend(fig, ax3, framevisible=false)
    fig[4, 2] = Legend(fig, ax4, framevisible=false)

    display(fig)
    return fig
end

#visualization of future prediction
function visualize_predictions(model, θ, st, t_o, t_p, history, u_p, predictions; sample_n=1)
    u_o, x_o, y₁_o, y₂_o = history
    Ex, Ey₁_p, Ey₂_p = predictions
    t_p, t_o = t_p .* 52.0f0, t_o .* 52.0f0
    ts = vcat(t_o, t_p)

    x_max, x_min, y_min, y_max = ts[end], -0.5, -2, 45

    fig = Figure(size=(1200, 900))
    ax1 = CairoMakie.Axis(fig[1, 1], xlabel="Time (weeks)", ylabel="Interventions", limits=((x_min, x_max), (0.0, 1.5)), yticks=[0, 1])
    ax2 = CairoMakie.Axis(fig[2, 1], xlabel="Time (weeks)", ylabel="Health status", limits=((x_min, x_max), (-0.5, 5.5)))
    ax3 = CairoMakie.Axis(fig[3, 1], xlabel="Time (weeks)", ylabel="Tumor size", limits=((x_min, x_max), (y_min, 50)))
    ax4 = CairoMakie.Axis(fig[4, 1], xlabel="Time (weeks)", ylabel="Cell count", limits=((x_min, x_max), (y_min, 50)))

    chemo_times_o, radio_times_o = t_o[u_o[1, :, sample_n] .> 0], t_o[u_o[2, :, sample_n] .> 0]
    chemo_times_p, radio_times_p = t_p[u_p[1, :, sample_n] .> 0], t_p[u_p[2, :, sample_n] .> 0]

    scatter!(ax1, chemo_times_o, fill(1, length(chemo_times_o)), color=:darkgreen, marker=:utriangle, markersize=15, label="Chemotherapy session")
    scatter!(ax1, radio_times_o, fill(1, length(radio_times_o)), color=:darkorange, marker=:star5, markersize=15, label="Radiotherapy session")
    scatter!(ax1, chemo_times_p, fill(1, length(chemo_times_p)), color=:darkgreen, marker=:utriangle, markersize=15)
    scatter!(ax1, radio_times_p, fill(1, length(radio_times_p)), color=:darkorange, marker=:star5, markersize=15)

    ŷ₁_m = selectdim(dropmean(Ey₁_p, dims=4), 3, sample_n)
    ŷ₂_m = selectdim(dropmean(Ey₂_p, dims=4), 3, sample_n)
    ŷ₂_s = selectdim(dropmean(std(Ey₂_p, dims=4), dims=4), 3, sample_n)

    ŷ₂_count = rand.(Poisson.(Ey₂_p))
    ŷ₂_m_count = selectdim(dropmean(ŷ₂_count, dims=4), 3, sample_n)
    ŷ₂_s_count = selectdim(dropmean(std(ŷ₂_count, dims=4), dims=4), 3, sample_n)
    y₁_o_class = onecold(y₁_o, Array(0:5))
    ŷ₁_class = onecold(ŷ₁_m, Array(0:5))

    scatter!(ax2, t_o, y₁_o_class[:, sample_n], color=:red, markersize=15, label="Observed Health status class")
    scatter!(ax2, t_p, ŷ₁_class, color=(:dodgerblue2, 0.7), markersize=10, label="Predicted Health status class")

    lines!(ax3, t_o, x_o[1, :, sample_n], linewidth=2, color=:red, label="True Tumor size (unobserved)")
    lines!(ax3, t_p, ŷ₂_m[1, :], color=(:dodgerblue2, 0.5), linewidth=2)
    band!(ax3, t_p, ŷ₂_m[1, :] .- sqrt.(ŷ₂_s[1, :]), ŷ₂_m[1, :] .+ sqrt.(ŷ₂_s[1, :]), color=(:dodgerblue2, 0.5), label="Inferred Tumor size (future)")

    scatter!(ax4, t_o, y₂_o[1, :, sample_n], color=:red, markersize=15, label="Observed Cell count")
    scatter!(ax4, t_p, ŷ₂_m_count[1, :], color=(:dodgerblue2, 0.7), markersize=10, label="Predicted Cell count")
    errorbars!(ax4, t_p, ŷ₂_m_count[1, :], ŷ₂_s_count[1, :], color=(:dodgerblue2, 0.7), whiskerwidth=8)

    poly!(ax1, [-0.5, length(t_o)-1, length(t_o)-1, -0.5], [0, 0, 50, 50], color=(:blue, 0.1), label="Observation period (history)")
    poly!(ax1, [length(t_o)-1, length(t_o) + length(t_p), length(t_o) + length(t_p), length(t_o)-1], [0, 0, 50, 50], color=(:red, 0.1), label="Prediction period (future)")
    poly!(ax2, [-0.5, length(t_o)-1, length(t_o)-1, -0.5], [-0.5, -0.5, 50, 50], color=(:blue, 0.1))
    poly!(ax2, [length(t_o)-1, length(t_o) + length(t_p), length(t_o) + length(t_p), length(t_o)-1], [-0.5, -0.5, 50, 50], color=(:red, 0.1))
    poly!(ax3, [-0.5, length(t_o)-1, length(t_o)-1, -0.5], [-2, -2, 50, 50], color=(:blue, 0.1))
    poly!(ax3, [length(t_o)-1, length(t_o) + length(t_p), length(t_o) + length(t_p), length(t_o)-1], [-2, -2, 50, 50], color=(:red, 0.1))
    poly!(ax4, [-0.5, length(t_o)-1, length(t_o)-1, -0.5], [-2, -2, 50, 50], color=(:blue, 0.1))
    poly!(ax4, [length(t_o)-1, length(t_o) + length(t_p), length(t_o) + length(t_p), length(t_o)-1], [-2, -2, 50, 50], color=(:red, 0.1))

    linkxaxes!(ax1, ax2, ax3, ax4)
    fig[1, 2] = Legend(fig, ax1, framevisible=false)
    fig[2, 2] = Legend(fig, ax2, framevisible=false)
    fig[3, 2] = Legend(fig, ax3, framevisible=false)
    fig[4, 2] = Legend(fig, ax4, framevisible=false)

    display(fig)
    return fig
end


## system identification 
rng = Random.MersenneTwister(1234);
train_loader, test_loader, val_loader, dims, timepoints, covars = generate_dataloader(; n_samples=512, batchsize=128, split=(0.5,0.3));
config = YAML.load_file("./configs/PkPD_config.yml");
exp_path = joinpath(config["experiment"]["path"], config["experiment"]["name"])
isdir(exp_path) ? exp_path : mkpath(exp_path)
model, θ, st = create_latentsde(config["model"], dims, rng);
θ_trained = train(model, θ, st, timepoints, loss_fn, eval_fn, viz_fn_sys_id, train_loader, test_loader, config["training"], exp_path);

## visualization of the system identification
fig=viz_fn_sys_id(model, θ_trained, st, timepoints, first(val_loader), config["training"]["validation"]; sample_n=4);
#save(joinpath(exp_path, "results_system_id_.pdf"), fig);

# validation of model prediction performance
u, x,covars, y₁, y₂, mask₁, mask₂= first(val_loader);
x=x[:,1:7:end,:];
spl=5
ind_observed = 1:spl; ind_predict = spl+1:length(timepoints);
history = (u[:,ind_observed,:], x[:,ind_observed,:] , y₁[:,ind_observed,:], y₂[:,ind_observed,:]);
ground_truth = (u[:,ind_predict,:],x[:,ind_predict,:], y₁[:,ind_predict,:], y₂[:,ind_predict,:], mask₁[:,ind_predict,:], mask₂[:,ind_predict,:]);
timepoints_observed = timepoints[ind_observed]; timepoints_predict = timepoints[ind_predict];
u_p=u[:,ind_predict,:];

Ex, Ey₁, Ey₂ = predict_future(model, θ_trained, st, history, u_p ,timepoints_predict , config["training"]["validation"]);
predictions = (Ex, Ey₁, Ey₂);

fig=validate_predictions(model, θ, st, timepoints_observed, timepoints_predict, history, u_p, ground_truth, predictions; sample_n=4);
#save(joinpath(exp_path, "results_prediction.pdf"), fig)

# Visualization of future prediction performance
fig=visualize_predictions(model, θ, st, timepoints_observed, timepoints_predict, history, u_p, predictions; sample_n=1);
#save(joinpath(exp_path, "results_future_prediction.pdf"), fig)
