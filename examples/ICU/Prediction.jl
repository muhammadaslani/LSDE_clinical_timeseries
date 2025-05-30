##dependencies
using Revise, Rhythm, Lux, DifferentialEquations, Random, SciMLSensitivity, ComponentArrays, Optimisers, OptimizationOptimisers, Statistics
using MLUtils, Printf, SciMLSensitivity, OneHotArrays, CairoMakie, Distributions
using YAML
using DataFrames, CSV
include("data_prep.jl");

##loading data
variables_of_interest = ["MAP","HR", "Temp"];
n_features = length(variables_of_interest);
data, train_loader, val_loader ,test_loader, time_series_dataset= load_data(split_at=24,n_samples=256, batch_size=64, variables_of_interest=variables_of_interest);
inputs_data_obs, obs_data_obs, output_data_obs, masks_obs, inputs_data_for, obs_data_for, output_data_for, masks_for=data;

n_timepoints = size(hcat(obs_data_obs, obs_data_for))[2]

tspan=(1.0, n_timepoints)
timepoints = (range(tspan[1], tspan[2], length=n_timepoints))/10 |> Array{Float32};

timepoints_obs = timepoints[1:size(obs_data_obs, 2)];
timepoints_for = timepoints[size(obs_data_obs, 2)+1:end];

## defining the model
dims = Dict(
    "input_dim" => size(inputs_data_obs, 1),
    "obs_dim" => size(obs_data_obs, 1),
    "output_dim" => ones(Int, size(output_data_for, 1)),
)

## defining the loss function
function loss_fn(model, θ, st, data)
    (u_obs, x_obs, y_obs, masks_obs, u_for, x_for, y_for, masks_for), ts, λ = data
    batch_size= size(y_for)[end]
    ŷ, px₀, kl_pq = model(x_obs, hcat(u_obs, u_for), ts, θ, st)
    recon_loss = 0.0f0
    for i in eachindex(ŷ)
        μ, log_σ² = ŷ[i][1], ŷ[i][2]
        valid_indx= findall(masks_for[i, :, :] .== 1)
        recon_loss += normal_loglikelihood(μ[1,valid_indx], log_σ²[1,valid_indx], y_for[i, valid_indx])/batch_size
    end 
    kl_loss = kl_normal(px₀...) / batch_size + mean(kl_pq[end, :])
    loss = recon_loss + λ * kl_loss
    return loss, st, (kl_loss, recon_loss, 0.0f0, 0.0f0)
end

## defining the evaluation function
function eval_fn(model, θ, st, ts, data, config)
    u_obs , x_obs, y_obs, masks_obs, u_for, x_for, y_for, masks_for = data
    batch_size= size(y_for)[end]
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    _, Ey = predict(model, solver, x_obs, hcat(u_obs,u_for), ts, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    loss=0.0f0
    for i in eachindex(Ey)
        μ, log_σ² = dropmean(Ey[i][1], dims=4), dropmean(Ey[i][2], dims=4)
        valid_indx= findall(masks_for[i, :, :] .== 1)
        loss += normal_loglikelihood(μ[1,valid_indx], log_σ²[1,valid_indx],y_for[i, valid_indx])/batch_size
    end
    return (loss, 0.0f0, 0.0f0) 
end


## defining the visualization function
function forecast(model, θ, st, obs_data, u_forecast, time_forecast, config)
    u_obs, x_obs, y_obs, masks_obs = obs_data    
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    _, Ey = predict(model, solver, x_obs, hcat(u_obs,u_forecast), time_forecast, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    μ = [Ey[i][1] for i in eachindex(Ey)]
    σ = [sqrt.(exp.(Ey[i][2])) for i in eachindex(Ey)]
    return μ, σ
end 

function viz_fn_forecast(t_obs, t_for, obs_data, future_true_data, forecasted_data; sample_n=1, plot=true)
    u_obs, x_obs, y_obs, masks_obs = obs_data
    u_for, x_for, y_for, masks_for = future_true_data
    μ, σ = forecasted_data
    t_obs = t_obs * 10 
    t_for = t_for * 10 

    y_labels = ["MAP", "HR", "BT"]
    fig = Figure(size=(1200, 600), fontsize=15)
    axes = CairoMakie.Axis[]
    rmse = []
    crps = []

    n_features = length(y_labels)  # Assuming one label per feature

    for i in 1:n_features
        y_label = y_labels[i]
        valid_indx_obs = findall(masks_obs[i, :, :] .== 1)
        valid_indx_for = findall(masks_for[i, :, :] .== 1)

        t_obs_val = t_obs[masks_obs[i, :, sample_n] .== 1]
        t_for_val = t_for[masks_for[i, :, sample_n] .== 1]

        y_obs_val = y_obs[i, valid_indx_obs]
        y_for_val = y_for[i, valid_indx_for]

        dists = Normal.(μ[i], σ[i])
        ŷ = rand.(dists)

        ŷ_mean = dropdims(mean(ŷ, dims=4), dims=4)
        ŷ_std = dropdims(std(ŷ, dims=4), dims=4)

        ŷ_mean_val = ŷ_mean[1, valid_indx_for]
        ŷ_std_val = ŷ_std[1, valid_indx_for]

        crps_ = empirical_crps(y_for[i:i, :, :], ŷ, masks_for[i:i, :, :])
        rmse_ = sqrt(MSELoss()(ŷ_mean_val, y_for_val))

        println("CRPS for $y_label: ", crps_)
        println("RMSE for $y_label: ", rmse_)

        push!(crps, crps_)
        push!(rmse, rmse_)

        ŷ_ci_lower = ŷ_mean .- ŷ_std
        ŷ_ci_upper = ŷ_mean .+ ŷ_std

        if plot
            if isempty(t_obs_val)
                println("No observational data available for $y_label in sample $sample_n")
                ax = CairoMakie.Axis(fig[i, 1], xlabel="Time (hours)", ylabel=y_labels[i], xgridvisible=false, ygridvisible=false)
                continue
            elseif isempty(t_for_val)
                println("No future data available for $y_label in sample $sample_n")
                ax = CairoMakie.Axis(fig[i, 1], xlabel="Time (hours)", ylabel=y_labels[i], xgridvisible=false, ygridvisible=false)
                continue
            else
                ax = CairoMakie.Axis(fig[i, 1], xlabel="Time (hours)", ylabel=y_labels[i], xgridvisible=false, ygridvisible=false)
                push!(axes, ax)
                scatter!(ax, t_obs_val, y_obs[i, masks_obs[i, :, sample_n] .== 1, sample_n], color=:blue, label="Past Observations", markersize=10)
                lines!(ax, t_obs_val, y_obs[i, masks_obs[i, :, sample_n] .== 1, sample_n], color=(:blue, 0.4), linewidth=2, linestyle=:dot)

                scatter!(ax, t_for_val, y_for[i, masks_for[i, :, sample_n] .== 1, sample_n], color=:green, label="Future Ground Truth", markersize=10)
                lines!(ax, t_for_val, y_for[i, masks_for[i, :, sample_n] .== 1, sample_n], color=(:green, 0.4), linestyle=:dot)

                scatter!(ax, t_for_val, ŷ_mean[1, masks_for[i, :, sample_n] .== 1, sample_n], color=:red, label="Model Predictions", markersize=10)
                lines!(ax, t_for_val, ŷ_mean[1, masks_for[i, :, sample_n] .== 1, sample_n], color=(:red, 0.4), linestyle=:dot)

                band!(ax, t_for_val, ŷ_ci_lower[1, masks_for[i, :, sample_n] .== 1, sample_n], ŷ_ci_upper[1, masks_for[i, :, sample_n] .== 1, sample_n], color=:red, alpha=0.2)

                if i == 1
                    poly!(ax, [0, t_obs[end], t_obs[end], 0], [-10, -10, 500, 500], color=(:blue, 0.05), label="Observation Period (Past)")
                    poly!(ax, [t_obs[end], t_for_val[end], t_for_val[end], t_obs[end]], [-10, -10, 500, 500], color=(:red, 0.05), label="Forecasting Period (Future)")
                else
                    poly!(ax, [0, t_obs[end], t_obs[end], 0], [-10, -10, 500, 500], color=(:blue, 0.05))
                    poly!(ax, [t_obs[end], t_for_val[end], t_for_val[end], t_obs[end]], [-10, -10, 500, 500], color=(:red, 0.05))
                end

                all_y_values = vcat(
                    y_obs[i, masks_obs[i, :, sample_n] .== 1, sample_n],
                    y_for[i, masks_for[i, :, sample_n] .== 1, sample_n],
                    ŷ_mean[1, masks_for[i, :, sample_n] .== 1, sample_n],
                    ŷ_ci_lower[1, masks_for[i, :, sample_n] .== 1, sample_n],
                    ŷ_ci_upper[1, masks_for[i, :, sample_n] .== 1, sample_n]
                )

                y_min = minimum(all_y_values) - 0.1 * (maximum(all_y_values) - minimum(all_y_values))
                y_max = maximum(all_y_values) + 0.1 * (maximum(all_y_values) - minimum(all_y_values))
                ylims!(ax, y_min, y_max)

                if i == 1
                    fig[i, 2] = Legend(fig, ax, framevisible=false, halign=:left)
                end
            end
        end
    end

    if plot
        linkxaxes!(axes...)
        colgap!(fig.layout, 10)
        display(fig)
        return fig, rmse, crps
    else
        return rmse, crps
    end
end

## model, training, and inference
rng = Random.MersenneTwister(123);
# latent SDE model
config_lsde = YAML.load_file("./configs/ICU_config_lsde.yml");
exp_path = joinpath(config_lsde["experiment"]["path"], config_lsde["experiment"]["name"])
lsde_model, lsde_θ, lsde_st = create_latentsde(config_lsde["model"], dims, rng);
lsde_θ_trained = train(lsde_model, lsde_θ, lsde_st, timepoints_for, loss_fn, eval_fn, viz_fn_forecast, train_loader, val_loader, config_lsde["training"], exp_path);

# latent ODE model
#latent ODE
config_lode = YAML.load_file("./configs/ICU_config_lode.yml");
lode_model, lode_θ, lode_st = create_latentsde(config_lode["model"], dims, rng);
lode_θ_trained = train(lode_model, lode_θ_trained, lode_st, timepoints_for, loss_fn, eval_fn, viz_fn_forecast, train_loader, val_loader, config_lode["training"], exp_path);

## forecasting
u_obs, x_obs, y_obs, masks_obs, u_forecast, x_forecast, y_forecast, masks_forecast = test_loader.data;
data_obs=(u_obs, x_obs, y_obs, masks_obs);
future_true_data=(u_forecast, x_forecast, y_forecast, masks_forecast);
t_for = timepoints_for;
t_obs= timepoints_obs;

## lsde forecast
μ, σ = forecast(lsde_model, lsde_θ_trained, lsde_st, data_obs, u_forecast, t_for, config_lsde["training"]["validation"]);
lsde_forecasted_data = (μ, σ);
fig, rmse, crps=viz_fn_forecast(t_obs, t_for, data_obs, future_true_data, lsde_forecasted_data; sample_n=1, plot=true);
#save("examples/ICU/ICU_lsde_forecast.eps", fig)

## lode forecast
μ, σ = forecast(lode_model, lode_θ_trained, lode_st, data_obs, u_forecast, t_for, config_lode["training"]["validation"]);
lode_forecasted_data = (μ, σ);
fig, rmse=viz_fn_forecast(t_obs, t_for, data_obs, future_true_data, lode_forecasted_data; sample_n=5);
#save("examples/ICU/ICU_lode_forecast.eps", fig)
