##dependencies
using Revise, Rhythm, Lux, DifferentialEquations, Random, SciMLSensitivity, ComponentArrays, Optimisers, OptimizationOptimisers, Statistics
using MLUtils, Printf, SciMLSensitivity, OneHotArrays, CairoMakie, Distributions
using YAML
using DataFrames
include("data_prep.jl");

##loading data
data, train_loader, val_loader , time_series_dataset= load_data(;n_samples=512, batch_size=128);
inputs_data,obs_data, outputs_data, outputs_masks=data;

n_timepoints = size(outputs_data)[2]
tspan=(1.0, n_timepoints)

timepoints = (range(tspan[1], tspan[2], length=n_timepoints))/500 |> Array{Float32};

## defining the model
dims = Dict(
    "input_dim" => size(inputs_data, 1),
    "obs_dim" => size(obs_data, 1),
    "output_dim" => size(outputs_data, 1),
)

## defining the loss function
function loss_fn(model, θ, st, data)
    (u,x,y,masks), ts, λ = data
    ŷ, px₀, kl_pq = model(x, u, ts, θ, st)
    μ, σ = ŷ[1], ŷ[2]
    recon_loss = normal_loglikelihood(masks.*μ, masks.*σ, y.*masks)/size(x)[end]
    kl_loss = kl_normal(px₀...) / size(x)[end] + mean(kl_pq[end, :])/size(x)[end]
    loss = recon_loss + λ * kl_loss
    return loss, st, kl_loss
end

## defining the evaluation function
function eval_fn(model, θ, st, ts, data, config)
    u, x, y, masks = data
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    px₀ = (zeros32(config["latent_dim"], size(y)[end]), ones32(config["latent_dim"], size(y)[end]))
    _, Ey = generate(model, solver, px₀, u, ts, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    μ, σ = dropmean(Ey[1],dims=4) , dropmean(Ey[2], dims=4)
    loss=normal_loglikelihood(masks.*μ, masks.*σ, y.*masks)/size(x)[end]
    return loss
end

function viz_fn_sys_id(model, θ, st, ts, data, config; sample_n=1)
    u, x, y, mask = data
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    px₀ = (zeros32(config["latent_dim"], size(y)[end]), ones32(config["latent_dim"], size(y)[end]))
    _, Ey = generate(model, solver, px₀, u, ts, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    μ, σ = Ey[1],exp.(Ey[2])
    valid_indx=findall(masks[1,:,sample_n].==1)
    y_val=y[1, valid_indx,:]
    μ_val=μ[1, valid_indx,:, :]
    σ_val=σ[1, valid_indx,:, :]
    ts_val=ts[valid_indx]
    dists=Normal.(μ_val, sqrt.(σ_val))
    ŷ_val=zeros(Float32, size(μ_val))
    for i in 1:size(μ_val)[1]
        for j in 1:size(μ_val)[2]
            for k in 1:size(μ_val)[3]
            ŷ_val[i,j,k]=rand(dists[i,j,k])
            end
        end
    end

    ŷ_val_mean = dropdims(mean(ŷ_val, dims=3), dims=3)
    ŷ_val_std = dropdims(std(ŷ_val, dims=3), dims=3)

    # standard error of mean for each sample since number of valid indices can be different for each sample
    ŷ_val_std_error=ŷ_val_std./sqrt(length(ŷ_val_mean[:,sample_n]))
    # 95% confidence interval
    ŷ_ci=1.96 * ŷ_val_std_error[:,sample_n]
    ŷ_ci_lower=ŷ_val_mean[:,sample_n].-ŷ_ci
    ŷ_ci_upper=ŷ_val_mean[:,sample_n].+ŷ_ci

    fig = Figure(size=(600, 400))
    ax1 = CairoMakie.Axis(fig[1, 1], xlabel="Time (hours)", ylabel="Variable of Interest")
    scatter!(ax1,ts_val, y_val[:,sample_n], color=:blue, label="True", markersize=15)
    lines!(ax1,ts_val, y_val[:,sample_n], color=:blue,  linewidth=3)
    scatter!(ax1,ts_val,  ŷ_val_mean[:,sample_n], color=:red, label="Predicted", markersize=15)
    lines!(ax1,ts_val, ŷ_val_mean[:,sample_n], color=:red,  linewidth=2)
    band!(ax1, ts_val, ŷ_ci_lower, ŷ_ci_upper, color=(:lightskyblue, 0.5), label="95% CI")    
    axislegend(ax1, position=:rt, backgroundcolor=:transparent)
    display(fig)
    println(MSELoss()(y_val, ŷ_val_mean))
    return fig
end 


## prediction
function predict_future(model, θ, st, history, u, t_p, config)
    u_o, x_o, y_o = history
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    Ex, Ey_p = predict(model, solver, reverse(x_o, dims=2), u, t_p, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    return Ex, Ey_p
end

function viz_fn_predictions(history, predictions,ground_truth, timepoints_o, timepoints_p; sample_n=1)
    u_o, x_o, y_o, masks_o = history
    y_p, masks_p=ground_truth
    μ,σ = predictions[1], exp.(predictions[2])
    valid_indx_o=findall(masks_o[1,:,sample_n].==1)
    valid_indx_p=findall(masks_p[1,:,sample_n].==1)
    y_o_val=y[1, valid_indx_o,:]
    y_p_val=y_p[1, valid_indx_p,:]
    μ_val=μ[1, valid_indx_p,:, :]
    σ_val=σ[1, valid_indx_p,:, :]

    dists=Normal.(μ_val, sqrt.(σ_val))
    ŷ_val=zeros(Float32, size(μ_val))
    for i in 1:size(μ_val)[1]
        for j in 1:size(μ_val)[2]
            for k in 1:size(μ_val)[3]
            ŷ_val[i,j,k]=rand(dists[i,j,k])
            end
        end
    end

    ŷ_val_mean = dropdims(mean(ŷ_val, dims=3), dims=3)
    ŷ_val_std = dropdims(std(ŷ_val, dims=3), dims=3)
    timepoints_o_val=timepoints_o[valid_indx_o].*500
    timepoints_p_val=timepoints_p[valid_indx_p].*500
    ŷ_val_std_error=ŷ_val_std./sqrt(length(ŷ_val_mean[:,sample_n]))
    # 95% confidence interval
    ŷ_ci=1.96 * ŷ_val_std_error[:,sample_n]
    ŷ_ci_lower=ŷ_val_mean[:,sample_n].-ŷ_ci
    ŷ_ci_upper=ŷ_val_mean[:,sample_n].+ŷ_ci



    fig = Figure(size=(600, 400))
    ax1 = CairoMakie.Axis(fig[1, 1], xlabel="Time (hours)", ylabel="Variable of Interest")
    scatter!(ax1, timepoints_o_val, y_o_val[:,sample_n], color=:blue, label="Observed", markersize=15)
    lines!(ax1, timepoints_o_val, y_o_val[ :,sample_n], color=:blue,  linewidth=3)
    scatter!(ax1, timepoints_p_val, ŷ_val_mean[ :,sample_n], color=:red, label="Predicted", markersize=15)
    lines!(ax1, timepoints_p_val, ŷ_val_mean[:,sample_n], color=:red,  linewidth=2, linestyle=:dash)
    scatter!(ax1, timepoints_p_val, y_p_val[ :,sample_n], color=:green, label="Ground Truth", markersize=15)
    lines!(ax1, timepoints_p_val, y_p_val[:,sample_n], color=:green,  linewidth=2)
    band!(ax1, timepoints_p_val, ŷ_ci_lower, ŷ_ci_upper, color=(:lightskyblue, 0.5), label="95% CI")
    axislegend(ax1, position=:rt, backgroundcolor=:transparent)
    display(fig)
    println(MSELoss()(y_p_val, ŷ_val_mean))
    return fig
end

rng = Random.MersenneTwister(1234);
config = YAML.load_file("./configs/ICU_config.yml");
exp_path = joinpath(config["experiment"]["path"], config["experiment"]["name"])
model, θ, st = create_latentsde(config["model"], dims, rng);
##training the model
θ_trained = train(model, θ_trained, st, timepoints, loss_fn, eval_fn, viz_fn_sys_id, train_loader, val_loader, config["training"], exp_path);

u,x,y,masks = first(val_loader);
viz_fn_sys_id(model, θ_trained, st, timepoints, first(val_loader), config["training"]["validation"]; sample_n=9);
##predicting future
spl=30;
ind_observed=1:spl; ind_predict=spl:length(timepoints);
history = (u[:,ind_observed,:], x[:, ind_observed, :], y[:, ind_observed,:], masks[:, ind_observed,:]);
ground_truth=(y[:, ind_predict,:], masks[:, ind_predict,:]);
timepoints_observed=timepoints[ind_observed];
timepoints_predict=timepoints[ind_predict];
u_predict = u[:,ind_predict,:];

Ex, Ey_p = predict_future(model, θ_trained, st, history, u, timepoints_predict, config["training"]["validation"]);
viz_fn_predictions(history, Ey_p, ground_truth,timepoints_observed, timepoints_predict, sample_n=7);