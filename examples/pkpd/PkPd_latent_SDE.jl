using  Revise, Rhythm, Lux, DifferentialEquations, Random, SciMLSensitivity, ComponentArrays, Optimisers, OptimizationOptimisers, Statistics
using MLUtils, Printf, SciMLSensitivity, OneHotArrays
using YAML
include("pkpd_standalone.jl")

function generate_dataloader(; n_samples=512, batchsize=64, split=(0.5,0.3), obs_fraction=0.5)
    U, X, Y₁, Y₂, T, covariates = generate_dataset(n_samples=n_samples);
    Y₁_padded, Masks₁, timepoints = pad_matrices(Y₁, T)
    Y₂_padded, Masks₂, _ = pad_matrices(Y₂, T)
    X_padded, _ = pad_matrices(X, T; return_timepoints=false)
    Y₁_irreg, Y₂_irreg, Masks₁, Masks₂ = irregularize(Y₁_padded,Y₂_padded, Masks₁, Masks₂)
    timepoints /= (7.0f0 * 52.0f0)  # Normalize timepoints
  
    covars=repeat(reshape(covariates,5,1,size(covariates,2)),1,size(Y₁_padded)[2],1)
    U = cat(U..., dims=3)
    #obs_n=Int(round(size(Y₁_padded)[2]*obs_fraction))
    U_obs, U_forcast=split_matrix(U, obs_fraction)
    X_obs, X_forcast=split_matrix(X_padded, obs_fraction)
    Covars_obs, Covars_forcast=split_matrix(covars, obs_fraction)
    Y₁_obs, Y₁_forcast=split_matrix(Y₁_irreg, obs_fraction)
    Y₂_obs, Y₂_forcast=split_matrix(Y₂_irreg, obs_fraction)
    Masks₁_obs, Masks₁_forcast=split_matrix(Masks₁, obs_fraction)
    Masks₂_obs, Masks₂_forcast=split_matrix(Masks₂, obs_fraction)
    timepoints_obs, timepoints_forecast= split_matrix(timepoints, obs_fraction)

    data_obs= (U_obs, X_obs, Covars_obs, Y₁_obs, Y₂_obs, Masks₁_obs, Masks₂_obs)
    data_forecast= (U_forcast, X_forcast, Covars_forcast, Y₁_forcast, Y₂_forcast, Masks₁_forcast, Masks₂_forcast)
    
    (train_data, test_data, val_data) = splitobs((data_obs, data_forecast), at=split)
    train_loader = DataLoader(train_data, batchsize=batchsize, shuffle=true)
    test_loader = DataLoader(test_data, batchsize=batchsize, shuffle=true)
    val_loader = DataLoader(val_data, batchsize=batchsize, shuffle=false)
    dims = Dict(
        "obs_dim" => [size(covars,1),size(Y₁_irreg, 1), size(Y₂_irreg, 1)],
        "input_dim" => size(U, 1),
        "state_dim" => size(X_padded, 1),
        "output_dim" => [size(Y₁_irreg, 1), size(Y₂_irreg, 1)]
    )
    return train_loader, test_loader, val_loader, dims, timepoints_obs, timepoints_forecast
end

function loss_fn(model, θ, st, data)
    (data_obs, data_forecast), ts, λ = data
    #data_obs, data_forecast= data_
    u_obs, x_obs, covars_obs, y₁_obs, y₂_obs, mask₁_obs, mask₂_obs = data_obs
    u_forecast, x_forecast, covars_forecast, y₁_forecast, y₂_forecast, mask₁_forecast, mask₂_forecast = data_forecast
    batch_size= size(y₁_obs)[end]
    ŷ, px₀, kl_pq = model(vcat(y₁_obs, y₂_obs, covars_obs), hcat(u_obs,u_forecast), ts, θ, st)
    ŷ₁, ŷ₂ = ŷ
    val_indx₁= findall(mask₁_forecast.==1)
    val_indx₂= findall(mask₂_forecast.==1)

    recon_loss = CrossEntropyLoss(;agg=sum, logits=true,  epsilon=1e-10)(ŷ₁[val_indx₁], y₁_forecast[val_indx₁])/batch_size - poisson_loglikelihood(ŷ₂[val_indx₂], y₂_forecast[val_indx₂])/ batch_size
    kl_loss = kl_normal(px₀...) / size(x_obs)[end] + mean(kl_pq[end, :])
    loss = recon_loss + λ * kl_loss
    return loss, st, kl_loss
end

function eval_fn(model, θ, st, ts, data, config)
    data_obs, data_forecast= data
    u_obs, x_obs, covars_obs, y₁_obs, y₂_obs, mask₁_obs, mask₂_obs = data_obs
    u_forecast, x_forecast, covars_forecast, y₁_forecast, y₂_forecast, mask₁_forecast, mask₂_forecast = data_forecast
    batch_size= size(y₁_obs)[end]
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    px₀ = (zeros32(config["latent_dim"], size(y₁_obs)[end]), ones32(config["latent_dim"], size(y₁_obs)[end]))
    Ex, Ey = generate(model, solver, px₀, hcat(u_obs,u_forecast), ts, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    ŷ₁_m, ŷ₂_m = dropmean(Ey[1], dims=4), dropmean(Ey[2], dims=4)
    val_indx₁= findall(mask₁_forecast.==1)
    val_indx₂= findall(mask₂_forecast.==1)
    return CrossEntropyLoss(;agg=sum, logits=true,  epsilon=1e-10)(ŷ₁_m[val_indx₁], y₁_forecast[val_indx₁])/ batch_size -poisson_loglikelihood(ŷ₂_m[val_indx₂], y₂_forecast[val_indx₂])/batch_size
end

## forecasting
function forecast(model, θ, st, obs_data, u_forecast, time_forecast, config)
    u_obs, x_obs, covars_obs, y₁_obs, y₂_obs, mask₁_obs, mask₂_obs = obs_data
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    Ex, Ey_p = predict(model, solver, vcat(reverse(y₁_obs, dims=2), reverse(y₂_obs, dims=2),covars_obs), u_forecast, time_forecast, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    return Ex, Ey_p[1], Ey_p[2]
end

# visualization of prediction performance (validation)

function vis_fn_forecast(obs_timepoints, for_timepoints, obs_data, future_true_data, forecasted_data; sample_n=1)
    u_o, x_o, covars_o, y₁_o, y₂_o, mask₁_o, mask₂_o = obs_data
    u_t, x_t, covars_t, y₁_t, y₂_t, mask₁_t, mask₂_t = future_true_data
    u_p= u_t
    Ex, Ey₁_p, Ey₂_p = forecasted_data
    t_o, t_p = obs_timepoints.* 52.0f0*7, for_timepoints.* 52.0f0*7

    #results 
    ŷ₁_m = dropmean(Ey₁_p, dims=4)
    ŷ₂_m = dropmean(Ey₂_p, dims=4)
    ŷ₂_s = dropmean(std(Ey₂_p, dims=4), dims=4)
    ŷ₂_count = rand.(Poisson.(Ey₂_p))
    ŷ₂_count_m = dropmean(ŷ₂_count, dims=4)
    ŷ₂_count_s = dropmean(std(ŷ₂_count, dims=4), dims=4)   
    y₁_o_class = onecold(y₁_o, Array(0:5))
    y₁_t_class = onecold(y₁_t, Array(0:5))
    ŷ₁_class = onecold(ŷ₁_m, Array(0:5))

    ## max time for observed and predicted data
    t_o_valid= t_o[mask₂_o[1,:,sample_n] .== 1]
    t_p_valid= t_p[mask₂_t[1,:,sample_n] .== 1]
    max_t_o_valid= maximum(t_o[mask₂_o[1,:,sample_n] .== 1])
    max_t_p_valid= maximum(t_p[mask₂_t[1,:,sample_n] .== 1])
    y₁_o_class_valid=y₁_o_class[findall(i-> t_o[i]<=max_t_o_valid .&& mask₁_o[1,i,sample_n] == 1, 1:length(t_o)),sample_n]
    y₁_t_class_valid=y₁_t_class[findall(i-> t_p[i]<=max_t_p_valid .&& mask₁_t[1,i,sample_n] == 1, 1:length(t_p)),sample_n]
    ŷ₁_class_valid=ŷ₁_class[findall(i-> t_p[i]<=max_t_p_valid .&& mask₁_t[1,i,sample_n] == 1, 1:length(t_p)),sample_n]
    y₂_o_valid=y₂_o[1,findall(i-> t_o[i]<=max_t_o_valid .&& mask₂_o[1,i,sample_n] == 1, 1:length(t_o)),sample_n]
    y₂_t_valid=y₂_t[1,findall(i-> t_p[i]<=max_t_p_valid .&& mask₂_t[1,i,sample_n] == 1, 1:length(t_p)),sample_n]
    ŷ₂_m_valid=ŷ₂_m[1,findall(i-> t_p[i]<=max_t_p_valid .&& mask₂_t[1,i,sample_n] == 1, 1:length(t_p)),sample_n]
    ŷ₂_s_valid=ŷ₂_s[1,findall(i-> t_p[i]<=max_t_p_valid .&& mask₂_t[1,i,sample_n] == 1, 1:length(t_p)),sample_n]
    ŷ₂_count_m_valid=ŷ₂_count_m[1,findall(i-> t_p[i]<=max_t_p_valid .&& mask₂_t[1,i,sample_n] == 1, 1:length(t_p)),sample_n]
    ŷ₂_count_s_valid=ŷ₂_count_s[1,findall(i-> t_p[i]<=max_t_p_valid .&& mask₂_t[1,i,sample_n] == 1, 1:length(t_p)),sample_n]
    
    ŷ₁_max = maximum(softmax(ŷ₁_m,dims=1), dims=1)
    ŷ₁_conf_valid=ŷ₁_max[1,findall(i-> t_p[i]<=max_t_p_valid .&& mask₁_t[1,i,sample_n] == 1, 1:length(t_p)),sample_n]
    ## errors and confidences
    ŷ₁_cross_entropy_valid=CrossEntropyLoss(;agg=sum, logits=true,  epsilon=1e-5)(ŷ₁_m[findall(mask₁_t.==1)], y₁_t[findall(mask₁_t.==1)])/length(ŷ₁_m[findall(mask₁_t.==1)])
    ŷ₁_entropy=prediction_entropy(ŷ₁_m)
    ŷ₁_entropy_valid=ŷ₁_entropy[1,findall(i-> t_p[i]<=max_t_p_valid .&& mask₁_t[1,i,sample_n] == 1, 1:length(t_p)),sample_n]
    ŷ₂_CI_low, ŷ₂_CI_up=ŷ₂_m[1,:,sample_n].-1.96*ŷ₂_s[1,:,sample_n], ŷ₂_m[1,:,sample_n].+1.96*ŷ₂_s[1,:,sample_n]

    ŷ₂_count_confidence_valid=1.96*sqrt.(ŷ₂_m_valid)
    ŷ₂_count_nll_valid=-poisson_loglikelihood(ŷ₂_m[findall(mask₂_t.==1)], y₂_t[findall(mask₂_t.==1)])/length(ŷ₂_m[findall(mask₂_t.==1)])

    println("Health Status cross entropy : ", ŷ₁_cross_entropy_valid)
    println("Cell count Negative log likelihood: ", ŷ₂_count_nll_valid)
    
    #chemptherapy and radiotherapy sessions 
    valid_indices_chemo_o = findall(i -> u_o[1,i, sample_n] == 1 && t_o[i] <= max_t_o_valid, 1:length(t_o))
    valid_indices_radio_o = findall(i -> u_o[2,i, sample_n] == 1 && t_o[i] <= max_t_o_valid, 1:length(t_o))
    valid_indices_chemo_p = findall(i -> u_p[1,i, sample_n] == 1 && t_p[i] <= max_t_p_valid, 1:length(t_p))
    valid_indices_radio_p = findall(i -> u_p[2,i, sample_n] == 1 && t_p[i] <= max_t_p_valid, 1:length(t_p))

    
    #plotting
    x_min, x_max= -2.0, max_t_p_valid+max_t_p_valid/50
    y_max_fig₃=maximum([maximum(ŷ₂_m_valid), maximum(x_o[1,:, sample_n])])+3
    y_max_fig₄=maximum([maximum(ŷ₂_count_m_valid), maximum(y₂_o_valid), maximum(y₂_t_valid)])+3
    y_min= -2.0

    fig = Figure(size=(1200, 900), fontsize=20)
    ax1 = CairoMakie.Axis(fig[1, 1], xlabel="Time (days)", ylabel="Interventions",limits=((x_min, x_max), (0.0, 1.5)),  yticks=[0, 1],xgridvisible = false, ygridvisible = false)
    ax2 = CairoMakie.Axis(fig[2, 1], xlabel="Time (days)", ylabel="Health status",limits=((x_min, x_max), (-2.0, 6)),xgridvisible = false, ygridvisible = false)
    ax3 = CairoMakie.Axis(fig[3, 1], xlabel="Time (days)", ylabel="Tumor size",limits=((x_min, x_max), (y_min,y_max_fig₃)),xgridvisible = false, ygridvisible = false)
    ax4 = CairoMakie.Axis(fig[4, 1], xlabel="Time (days)", ylabel="Cell count",limits=((x_min, x_max), (y_min, y_max_fig₄)),xgridvisible = false, ygridvisible = false)

    scatter!(ax1, t_o[valid_indices_chemo_o], ones(length(u_o[valid_indices_chemo_o])),marker = :utriangle,markersize = 10,color = :blue, label="Chemotherapy regimen")
    scatter!(ax1, t_o[valid_indices_radio_o], ones(length(u_o[valid_indices_radio_o])),marker = :star5,markersize = 10,color = :red, label="Radiotherapy regimen")
    scatter!(ax1, t_p[valid_indices_chemo_p], ones(length(u_p[valid_indices_chemo_p])),marker = :utriangle,markersize = 10,color = :blue)
    scatter!(ax1, t_p[valid_indices_radio_p], ones(length(u_p[valid_indices_radio_p])),marker = :star5,markersize = 10,color = :red)

    scatter!(ax2, t_o_valid, y₁_o_class_valid, color = :blue, markersize=10, label="Observed")
    scatter!(ax2, t_p_valid, y₁_t_class_valid, color = (:green,0.5), markersize=15, label="True")
    scatter!(ax2, t_p_valid, ŷ₁_class_valid, color = (:red, 0.9),markersize=10, label="Predicted")
    errorbars!(ax2, t_p_valid, ŷ₁_class_valid, ŷ₁_entropy_valid, color=(atom_one_dark[:red], 0.5), whiskerwidth=8, label="Prediction uncertainty")
    #errorbars!(ax2, t_p_valid, ŷ₁_class_valid, ŷ₁_conf_valid, color=(atom_one_dark[:green], 0.3), whiskerwidth=8, label="Prediction uncertainty")
    
    lines!(ax3, Array(1:366), vcat(x_o[1,:, sample_n], x_t[1,:, sample_n]), color = :blue, label="Observed (underlying tumor size)")
    lines!(ax3, t_p, ŷ₂_m[1,:, sample_n], color = :red, label="Predicted (contiuous)")
    scatter!(ax3, t_p_valid, ŷ₂_m_valid, color = :red, label="Predicted (weekly irregular)")
    band!(ax3, t_p, ŷ₂_CI_low, ŷ₂_CI_up, color=(atom_one_dark[:red], 0.5), label="Prediction uncertainty")

    scatter!(ax4, t_o_valid, y₂_o_valid, color = :blue, label="Observed")
    scatter!(ax4, t_p_valid, y₂_t_valid, color = (:green,0.5),markersize=15, label="True")
    scatter!(ax4, t_p_valid, ŷ₂_count_m_valid, color = (:red, 0.9), label="Predicted")
    errorbars!(ax4, t_p_valid, ŷ₂_count_m_valid, ŷ₂_count_confidence_valid, color=(atom_one_dark[:red], 0.5), whiskerwidth=8, label="Prediction uncertainty")

    poly!(ax1, [-10, t_o[end], t_o[end], -10], [-10, -10, 500, 500], color=(:blue, 0.05), label="observation period (history)")
    poly!(ax1, [t_o[end], t_o[end] + max_t_p_valid, t_o[end] + max_t_p_valid, t_o[end]], [-10, -10, 500, 500], color=(:red, 0.05), label="prediction period (future)")
    poly!(ax2, [-10, t_o[end], t_o[end], -10], [-10, -10, 500, 500], color=(:blue, 0.05))
    poly!(ax2, [t_o[end], t_o[end] + max_t_p_valid, t_o[end] + max_t_p_valid, t_o[end]], [-10, -10, 500, 500], color=(:red, 0.05))
    poly!(ax3, [-10, t_o[end], t_o[end], -10], [-10, -10, 500, 500], color=(:blue, 0.05))
    poly!(ax3, [t_o[end], t_o[end] + max_t_p_valid, t_o[end] + max_t_p_valid, t_o[end]], [-10, -10, 500, 500], color=(:red, 0.05))
    poly!(ax4, [-10, t_o[end], t_o[end], -10], [-10, -10, 500, 500], color=(:blue, 0.05))
    poly!(ax4, [t_o[end], t_o[end] + max_t_p_valid, t_o[end] + max_t_p_valid, t_o[end]], [-10, -10, 500, 500], color=(:red, 0.05))



    linkxaxes!(ax1, ax2, ax3, ax4)
    fig[1, 2] = Legend(fig, ax1, framevisible=false,halign=:left)
    fig[2, 2] = Legend(fig, ax2, framevisible=false,halign=:left)
    fig[3, 2] = Legend(fig, ax3, framevisible=false,halign=:left)
    fig[4, 2] = Legend(fig, ax4, framevisible=false,halign=:left)
    display(fig)
    return fig
end
## system identification 
rng = Random.MersenneTwister(123);
train_loader, test_loader, val_loader, dims, timepoints_obs, timepoints_forecast = generate_dataloader(; n_samples=512, batchsize=64, split=(0.4,0.5), obs_fraction=0.2);

#latent SDE
config_lsde = YAML.load_file("./configs/PkPD_config_LSDE.yml");
exp_path = joinpath(config_lsde["experiment"]["path"], config_lsde["experiment"]["name"])
isdir(exp_path) ? exp_path : mkpath(exp_path)
lsde_model, lsde_θ, lsde_st = create_latentsde(config_lsde["model"], dims, rng);
lsde_θ_trained = train(lsde_model, lsde_θ, lsde_st, timepoints_forecast, loss_fn, eval_fn, vis_fn_forecast, train_loader, test_loader, config_lsde["training"], exp_path);

#latent ODE
config_lode = YAML.load_file("./configs/PkPD_config_LODE.yml");
lode_model, lode_θ, lode_st = create_latentsde(config_lode["model"], dims, rng);
lode_θ_trained = train(lode_model, lode_θ_trained, lode_st, timepoints_forecast, loss_fn, eval_fn, vis_fn_forecast, train_loader, test_loader, config_lode["training"], exp_path);


# visualization of prediction performance (validation)
data_obs, data_forecast= va_loader.data;
u_obs, x_obs, covars_obs, y₁_obs, y₂_obs, mask₁_obs, mask₂_obs= data_obs;
u_forecast, x_forecast, covars_forecast, y₁_forecast, y₂_forecast, mask₁_forecast, mask₂_forecast= data_forecast;

#lsde
lsde_Ex, lsde_Ey₁, lsde_Ey₂ = forecast(lsde_model, lsde_θ_trained, lsde_st, data_obs, u_forecast ,timepoints_forecast , config_lsde["training"]["validation"]);
lsde_forecasted_data = (lsde_Ex, lsde_Ey₁, lsde_Ey₂);
lsde_fig=vis_fn_forecast(timepoints_obs, timepoints_forecast, data_obs, data_forecast, lsde_forecasted_data; sample_n=6);
#save("examples/pkpd/lsde_forecast.eps", lsde_fig)
#lode
lode_Ex, lode_Ey₁, lode_Ey₂ = forecast(lode_model, lode_θ_trained, lode_st, data_obs, u_forecast ,timepoints_forecast , config_lode["training"]["validation"]);
lode_forecasted_data = (lode_Ex, lode_Ey₁, lode_Ey₂);
lode_fig=vis_fn_forecast(timepoints_obs, timepoints_forecast, data_obs, data_forecast, lode_forecasted_data; sample_n=2);
#save("examples/pkpd/lode_forecast.eps", lode_fig)
