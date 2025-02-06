##dependencies
using Revise, Rhythm, Lux, DifferentialEquations, Random, SciMLSensitivity, ComponentArrays, Optimisers, OptimizationOptimisers, Statistics
using MLUtils, Printf, SciMLSensitivity, OneHotArrays, CairoMakie
using YAML
using DataFrames
include("data_prep.jl");

##loading data
data, train_loader, val_loader , time_series_dataset= load_data(;n_samples=16, batch_size=8);
inputs_data,obs_data, outputs_data, outputs_masks=data;

n_timepoints = size(outputs_data)[2]
tspan=(1.0, n_timepoints)

timepoints = (range(tspan[1], tspan[2], length=n_timepoints))/50 |> Array{Float32};

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
    recon_loss = MSELoss()(masks.*ŷ, masks.*y)
    kl_loss = kl_normal(px₀...) / size(x)[end] + mean(kl_pq[end, :])
    loss = recon_loss + λ * kl_loss
    return loss, st, kl_loss
end

## defining the evaluation function
function eval_fn(model, θ, st, ts, data, config)
    u, x, y, mask = data
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    px₀ = (zeros32(config["latent_dim"], size(y)[end]), ones32(config["latent_dim"], size(y)[end]))
    _, Ey = generate(model, solver, px₀, u, ts, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    ŷ = dropmean(Ey, dims=4)
    return MSELoss()(mask.*ŷ, mask.*y )
end

function viz_fn_sys_id(model, θ, st, ts, data, config; sample_n=1)
    u, x, y, mask = data
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    px₀ = (zeros32(config["latent_dim"], size(y)[end]), ones32(config["latent_dim"], size(y)[end]))
    _, Ey = generate(model, solver, px₀, u, ts, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    ŷ = dropmean(Ey, dims=4)
    valid_indx=findall(mask[1,:,sample_n].==1)
    @show size(ŷ[1,valid_indx,sample_n])
    fig = Figure(size=(600, 400))
    ax1 = CairoMakie.Axis(fig[1, 1], xlabel="Time (hours)", ylabel="Variable of Interest")
    scatter!(ax1, y[1, valid_indx,sample_n], color=:blue, label="True", markersize=10)
    lines!(ax1, y[1, valid_indx,sample_n], color=:blue,  linewidth=2, linestyle=:dot)
    scatter!(ax1, ŷ[1, valid_indx,sample_n], color=:red, label="Predicted", markersize=10)
    lines!(ax1, ŷ[1, valid_indx,sample_n], color=:red,  linewidth=2, linestyle=:dot)
    axislegend(ax1, position=:rt, backgroundcolor=:transparent)
    display(fig)
    return fig
end 

## prediction
function predict_future(model, θ, st, history, u_p, t_p, config)
    u_o, x_o, y_o = history
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    Ex, Ey_p = predict(model, solver, reverse(x_o, dims=2), u_p, t_p, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    return Ex, Ey_p
end

function viz_fn_predictions(history, predictions,ground_truth, timepoints_observed, timepoints_predict; sample_n=1)
    u_o, x_o, y_o = history
    y_p = predictions
    y_p_m=dropmean(y_p, dims=4)
    fig = Figure(size=(600, 400))
    ax1 = CairoMakie.Axis(fig[1, 1], xlabel="Time (hours)", ylabel="Variable of Interest")
    scatter!(ax1, timepoints_observed, y_o[1, :,sample_n], color=:blue, label="Observed", markersize=10)
    lines!(ax1, timepoints_observed, y_o[1, :,sample_n], color=:blue,  linewidth=2, linestyle=:dot)
    scatter!(ax1, timepoints_predict, y_p_m[1, :,sample_n], color=:red, label="Predicted", markersize=10)
    lines!(ax1, timepoints_predict, y_p_m[1, :,sample_n], color=:red,  linewidth=2, linestyle=:dot)
    scatter!(ax1, timepoints_predict, ground_truth[1, :,sample_n], color=:green, label="Ground Truth", markersize=10)
    lines!(ax1, timepoints_predict, ground_truth[1, :,sample_n], color=:green,  linewidth=2, linestyle=:dot)
    axislegend(ax1, position=:rt, backgroundcolor=:transparent)
    display(fig)
    return fig
end


rng = Random.MersenneTwister(1234);
config = YAML.load_file("./configs/ICU_config.yml");
exp_path = joinpath(config["experiment"]["path"], config["experiment"]["name"])
model, θ, st = create_latentsde(config["model"], dims, rng);
##training the model
θ_trained = train(model, θ, st, timepoints, loss_fn, eval_fn, viz_fn_sys_id, train_loader, val_loader, config["training"], exp_path);

u,x,y,masks = first(train_loader);
ŷ, px₀, kl_pq = model(x, u, timepoints, θ_trained, st);

data= (u,x,y,masks);
viz_fn_sys_id(model, θ_trained, st, timepoints, data, config["training"]["validation"]; sample_n=5)

##predicting the future
u,x,y,masks = first(train_loader);
spl=30;
ind_observed=1:spl; ind_predict=spl:length(timepoints);
history = (u[:,ind_observed,:], x[:, ind_observed, :], y[:, ind_observed,:]);
ground_truth=y[:, ind_predict,:];
timepoints_observed=timepoints[ind_observed];
timepoints_predict=timepoints[ind_predict];
u_predict = u[:,ind_predict,:];
u_o, x_o, y_o = history;
Ex, Ey_p = predict_future(model, θ_trained, st, history, u_predict, timepoints_predict, config["training"]["validation"]);
viz_fn_predictions(history, Ey_p, ground_truth,timepoints_observed, timepoints_predict, sample_n=2);