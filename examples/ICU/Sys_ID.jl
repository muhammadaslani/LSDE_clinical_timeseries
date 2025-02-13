##dependencies
using Revise, Rhythm, Lux, DifferentialEquations, Random, SciMLSensitivity, ComponentArrays, Optimisers, OptimizationOptimisers, Statistics
using MLUtils, Printf, SciMLSensitivity, OneHotArrays, CairoMakie, Distributions
using YAML
using DataFrames, CSV
include("data_prep.jl");

##loading data
data, train_loader ,test_loader, val_loader, time_series_dataset= load_data(;n_samples=128, batch_size=32);
inputs_data, obs_data, timeseries_data, masks=data;

n_timepoints = size(obs_data)[2]
tspan=(1.0, n_timepoints)

timepoints = (range(tspan[1], tspan[2], length=n_timepoints))/n_timepoints |> Array{Float32};

## defining the model
dims = Dict(
    "input_dim" => size(inputs_data, 1),
    "obs_dim" => size(obs_data, 1),
    "output_dim" => ones(Int, size(timeseries_data, 1)),)

## defining the loss function
function loss_fn(model, θ, st, data)
    (u, x, y, masks), ts, λ = data
    ŷ, px₀, kl_pq = model(x, u, ts, θ, st)
    μ = [ŷ[i][1] for i in eachindex(ŷ)]
    log_σ = [ŷ[i][2] for i in eachindex(ŷ)]
    recon_loss = sum(normal_loglikelihood(masks[i:i,:,:] .* μ[i], masks[i:i,:,:] .* log_σ[i], masks[i:i,:,:] .* y[i:i,:,:]) for i in eachindex(ŷ))/size(y)[end]
    kl_loss = kl_normal(px₀...) / size(y)[end] + mean(kl_pq[end, :])
    loss = recon_loss + λ * kl_loss
    return loss, st, kl_loss
end

## defining the evaluation function
function eval_fn(model, θ, st, ts, data, config)
    u,x, y, masks = data
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    _, Ey = predict(model, solver, x, u, ts, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    loss = sum(begin
        μ, log_σ = dropmean(Ey[i][1], dims=4), dropmean(Ey[i][2], dims=4)
        normal_loglikelihood(masks[i:i,:,:] .* μ, masks[i:i,:,:] .* log_σ, masks[i:i,:,:] .* y[i:i,:,:])
    end for i in eachindex(Ey))/size(y)[end]
    return loss
end

function viz_fn_sys_id(model, θ, st, ts, data, config; sample_n=1, var_of_intrst=1)
    u,x, y, masks = data
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    px₀ = (zeros32(config["latent_dim"], size(x)[end]), ones32(config["latent_dim"], size(x)[end]))
    _, Ey = generate(model, solver, px₀, u, ts, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    μ, σ = Ey[var_of_intrst][1], exp.(Ey[var_of_intrst][2])
    valid_indx=findall(masks[var_of_intrst,:,sample_n].==1)
    if isempty(valid_indx)
        error("No observations available for this sample of this variable (history): valid_indx_o is empty.")
    end

    # Extract valid time points and observations
    ts_val = ts[valid_indx] .* n_timepoints
    y_val = y[var_of_intrst, valid_indx, :]
    μ_val=μ[1, valid_indx,:, :]
    σ_val=σ[1, valid_indx,:, :]

    dists=Normal.(μ_val, sqrt.(σ_val))
    ŷ_val = rand.(dists)

    ŷ_val_mean = dropdims(mean(ŷ_val, dims=3), dims=3)
    ŷ_val_std = dropdims(std(ŷ_val, dims=3), dims=3)

    # standard error of mean for each sample since number of valid indices can be different for each sample
    ŷ_val_std_error=ŷ_val_std./sqrt(length(ŷ_val_mean[:,sample_n]))
    # 95% confidence interval
    ŷ_val_std_error = ŷ_val_std[:, sample_n] / sqrt(length(ŷ_val_mean[:, sample_n]))
    ŷ_ci_lower = ŷ_val_mean[:, sample_n] - 1.96 * ŷ_val_std_error
    ŷ_ci_upper = ŷ_val_mean[:, sample_n] + 1.96 * ŷ_val_std_error

    fig = Figure(size=(900, 600))
    ax1 = CairoMakie.Axis(fig[1, 1], xlabel="Time (hours)", ylabel="Variable of Interest")
    scatter!(ax1,ts_val, y_val[:,sample_n], color=:blue, label="True", markersize=15)
    lines!(ax1,ts_val, y_val[:,sample_n], color=:blue,  linewidth=3)
    scatter!(ax1,ts_val,  ŷ_val_mean[:,sample_n], color=:red, label="Predicted", markersize=15)
    lines!(ax1,ts_val, ŷ_val_mean[:,sample_n], color=:red,  linewidth=2)
    band!(ax1, ts_val, ŷ_ci_lower, ŷ_ci_upper, color=(:red, 0.3), label="95% CI")    
    axislegend(ax1, position=:rt, backgroundcolor=:transparent)
    display(fig)
    println("MSE for batch:",MSELoss()(  ŷ_val_mean, y_val))
    println("MSE for sample number $sample_n: ", MSELoss()(ŷ_val_mean[:, sample_n], y_val[:, sample_n]))
    return fig
end 


## prediction
function predict_future(model, θ, st, history, u, t_p, config)
    u_o, x_o, y_o, masks_o = history
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    Ex, Ey_p = predict(model, solver, reverse(x_o, dims=2), u, t_p, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    return Ex, Ey_p
end

function viz_fn_predictions(history, predictions,ground_truth, ts_o, ts_p; sample_n=1, var_of_intrst=1)
    u_o,x_o, y_o, masks_o = history
    y_p, masks_p=ground_truth

    μ,σ = predictions[var_of_intrst][1], exp.(predictions[var_of_intrst][2])
    valid_indx_o=findall(masks_o[var_of_intrst,:,sample_n].==1)
    valid_indx_p=findall(masks_p[var_of_intrst,:,sample_n].==1)
    if isempty(valid_indx_o)
        error("No observations available for this sample of this variable (history): valid_indx_o is empty.")
    end
    if isempty(valid_indx_p)
        error("No observations available for this sample of this variable (future): valid_indx_p is empty.")
    end
    ts_o_val = ts_o[valid_indx_o] .* n_timepoints
    ts_p_val = ts_p[valid_indx_p] .* n_timepoints
    y_o_val=y_o[var_of_intrst, valid_indx_o,:]
    y_p_val=y_p[var_of_intrst, valid_indx_p,:]
    μ_val=μ[1, valid_indx_p,:, :]
    σ_val=σ[1, valid_indx_p,:, :]

    dists=Normal.(μ_val, sqrt.(σ_val))
    ŷ_val=zeros(Float32, size(μ_val))
    ŷ_val = rand.(dists)

    ŷ_val_mean = dropdims(mean(ŷ_val, dims=3), dims=3)
    ŷ_val_std = dropdims(std(ŷ_val, dims=3), dims=3)
    ŷ_val_std_error=ŷ_val_std./sqrt(length(ŷ_val_mean[:,sample_n]))
    # 95% confidence interval
    ŷ_ci_lower=ŷ_val_mean[:,sample_n].-1.96 * ŷ_val_std_error[:,sample_n]
    ŷ_ci_upper=ŷ_val_mean[:,sample_n].+1.96 * ŷ_val_std_error[:,sample_n]



    fig = Figure(size=(900, 600))
    ax1 = CairoMakie.Axis(fig[1, 1], xlabel="Time (hours)", ylabel="Variable of Interest")
    scatter!(ax1, ts_o_val, y_o_val[:,sample_n], color=:blue, label="Observed", markersize=15)
    lines!(ax1, ts_o_val, y_o_val[ :,sample_n], color=:blue)
    scatter!(ax1, ts_p_val, y_p_val[ :,sample_n], color=:green, label="Ground Truth", markersize=15)
    lines!(ax1, ts_p_val, y_p_val[:,sample_n], color=:green)
    scatter!(ax1, ts_p_val, ŷ_val_mean[ :,sample_n], color=:red, label="Predicted", markersize=15)
    lines!(ax1, ts_p_val, ŷ_val_mean[:,sample_n], color=:red, linestyle=:dash)
    band!(ax1, ts_p_val, ŷ_ci_lower, ŷ_ci_upper, color=(:red, 0.3), label="95% CI")
    axislegend(ax1, position=:rt, backgroundcolor=:transparent)
    display(fig)
    println("MSE for batch:",MSELoss()(  ŷ_val_mean, y_p_val))
    println("MSE for sample number $sample_n: ", MSELoss()(ŷ_val_mean[:, sample_n], y_p_val[:, sample_n]))
    return fig
end 

rng = Random.MersenneTwister(1234);
config = YAML.load_file("./configs/ICU_config.yml");
exp_path = joinpath(config["experiment"]["path"], config["experiment"]["name"])
model, θ, st = create_latentsde(config["model"], dims, rng);
##training the model
θ_trained = train(model, θ_trained, st, timepoints, loss_fn, eval_fn, viz_fn_sys_id, train_loader, test_loader, config["training"], exp_path);

viz_fn_sys_id(model, θ_trained, st, timepoints, first(test_loader), config["training"]["validation"]; sample_n=1, var_of_intrst=2);
##predicting future
spl=30;
ind_observed=1:spl; ind_predict=spl:length(timepoints);
u,x,y,masks=first(test_loader);
history = (u[:,ind_observed,:],x[:,ind_observed,:],  y[:, ind_observed,:], masks[:, ind_observed,:]);
ground_truth=(y[:, ind_predict,:], masks[:, ind_predict,:]);
ts_observed=timepoints[ind_observed];
ts_predict=timepoints[ind_predict];

Ex, Ey_p = predict_future(model, θ_trained, st, history, u, ts_predict, config["training"]["validation"]);
viz_fn_predictions(history, Ey_p, ground_truth,ts_observed, ts_predict, sample_n=1, var_of_intrst=6);