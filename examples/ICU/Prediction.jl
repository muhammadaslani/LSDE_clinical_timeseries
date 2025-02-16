##dependencies
using Revise, Rhythm, Lux, DifferentialEquations, Random, SciMLSensitivity, ComponentArrays, Optimisers, OptimizationOptimisers, Statistics
using MLUtils, Printf, SciMLSensitivity, OneHotArrays, CairoMakie, Distributions
using YAML
using DataFrames
include("data_prep.jl");

##loading data
data, train_loader, val_loader ,test_loader, time_series_dataset= load_data(24 ;n_samples=256, batch_size=32);
inputs_data_hist, obs_data_hist,output_data_hist, masks_hist,inputs_data_fut, obs_data_fut,output_data_fut, masks_fut=data;

n_timepoints = size(hcat(obs_data_hist, obs_data_fut))[2]

tspan_hist=(1.0, n_timepoints)
timepoints = (range(tspan[1], tspan[2], length=n_timepoints))/(n_timepoints) |> Array{Float32};

timepoints_hist = timepoints[1:size(obs_data_hist, 2)]
timepoints_fut = timepoints[size(obs_data_hist, 2)+1:end]

## defining the model
dims = Dict(
    "input_dim" => size(inputs_data, 1),
    "obs_dim" => size(obs_data_hist, 1),
    "output_dim" => ones(Int, size(output_data_fut, 1)),
)

## defining the loss function
function loss_fn(model, θ, st, data)
    (u_h,x_h, y_h, masks_h, u_f,x_f,y_f, masks_f), ts, λ = data
    ŷ, px₀, kl_pq = model(x_h, hcat(u_h, u_f), ts, θ, st)
    μ = [ŷ[i][1] for i in eachindex(ŷ)]
    log_σ = [ŷ[i][2] for i in eachindex(ŷ)]
    recon_loss = sum(normal_loglikelihood(masks_f[i:i,:,:] .* μ[i], masks_f[i:i,:,:] .* log_σ[i], masks_f[i:i,:,:] .* y_f[i:i,:,:]) for i in eachindex(ŷ))/size(y_f)[end]
    kl_loss = kl_normal(px₀...) / size(y_f)[end] + mean(kl_pq[end, :])
    loss = recon_loss + λ * kl_loss
    return loss, st, kl_loss
end

## defining the evaluation function
function eval_fn(model, θ, st, ts, data, config)
    u_h ,x_h, y_h, masks_h,u_f, x_f, y_f, masks_f = data
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    _, Ey = predict(model, solver, x_h, hcat(u_h,u_f), ts, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    loss = sum(begin
        μ, log_σ = dropmean(Ey[i][1], dims=4), dropmean(Ey[i][2], dims=4)
        normal_loglikelihood(masks_f[i:i,:,:] .* μ, masks_f[i:i,:,:] .* log_σ, masks_f[i:i,:,:] .* y_f[i:i,:,:])
    end for i in eachindex(Ey))/size(y_f)[end]
    return loss
end


## defining the visualization function
function viz_fn(model, θ, st, ts_h, ts_f, data, config; sample_n=1, var_of_intrst=1)
    u_h,x_h, y_h, masks_h,u_f,x_f, y_f, masks_f = data
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    _, Ey = predict(model, solver, x_h, hcat(u_h,u_f), ts_f, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    μ, σ = Ey[var_of_intrst][1], exp.(Ey[var_of_intrst][2])

    # Validate observation indices
    valid_indx_h = findall(masks_h[var_of_intrst, :, sample_n] .== 1)
    if isempty(valid_indx_h)
        error("No observations available for this sample of this variable (history): valid_indx_h is empty.")
    end

    valid_indx_f = findall(masks_f[var_of_intrst, :, sample_n] .== 1)
    if isempty(valid_indx_f)
        error("No observations available for this sample of this variable (future): valid_indx_f is empty.")
    end

    # Extract valid time points and observations
    ts_h_val = ts_h[valid_indx_h] .* 50
    y_h_val = y_h[var_of_intrst, valid_indx_h, :]
    ts_f_val = ts_f[valid_indx_f] .* 50
    y_f_val = y_f[var_of_intrst, valid_indx_f, :]
    @show ts_h_val, ts_f_val
    # Extract mean and variance for predictions based on the predicted gaussian distribution for each output (μ,σ)
    μ_val = μ[1, valid_indx_f, :, :]
    σ_val = σ[1, valid_indx_f, :, :]

    # Generate predicted distributions
    dists = Normal.(μ_val, sqrt.(σ_val))
    ŷ_val = rand.(dists)

    # Calculate mean and standard deviation of predictions
    ŷ_val_mean = dropdims(mean(ŷ_val, dims=3), dims=3)
    ŷ_val_std = dropdims(std(ŷ_val, dims=3), dims=3)
    ŷ_val_std_error = ŷ_val_std[:, sample_n] / sqrt(length(ŷ_val_mean[:, sample_n]))
    ŷ_ci_lower = ŷ_val_mean[:, sample_n] - 1.96 * ŷ_val_std_error
    ŷ_ci_upper = ŷ_val_mean[:, sample_n] + 1.96 * ŷ_val_std_error

    # Plot the results
    fig = Figure(size = (900, 600))
    ax1 = CairoMakie.Axis(fig[1, 1], xlabel="Time (hours)", ylabel="Variable of Interest")

    scatter!(ax1, ts_h_val, y_h_val[:, sample_n], color=:blue, label="History", markersize=15)
    lines!(ax1, ts_h_val, y_h_val[:, sample_n], color=:blue)
    scatter!(ax1, ts_f_val, y_f_val[:, sample_n], color=:green, label="True Future", markersize=15)
    lines!(ax1, ts_f_val, y_f_val[:, sample_n], color=:green)
    scatter!(ax1, ts_f_val, ŷ_val_mean[:, sample_n], color=:red, label="Predicted Future", markersize=10)
    lines!(ax1, ts_f_val, ŷ_val_mean[:, sample_n], color=:red, linestyle=:dot)
    band!(ax1, ts_f_val, ŷ_ci_lower, ŷ_ci_upper, color=:red, alpha=0.3)

    axislegend(ax1, position=:rt, backgroundcolor=:transparent)
    display(fig)

    # Print the mean squared error loss
    println("MSE for batch:",MSELoss()(  ŷ_val_mean, y_f_val))
    println("MSE for sample number $sample_n: ", MSELoss()(ŷ_val_mean[:, sample_n], y_f_val[:, sample_n]))
    return fig
end

## model, training, and inference
rng = Random.MersenneTwister(123);
config = YAML.load_file("./configs/ICU_config.yml");
exp_path = joinpath(config["experiment"]["path"], config["experiment"]["name"])
model, θ, st = create_latentsde(config["model"], dims, rng);
θ_trained = train(model, θ_trained, st, timepoints_fut, loss_fn, eval_fn, viz_fn, train_loader, val_loader, config["training"], exp_path);

viz_fn(model, θ_trained, st,timepoints_hist, timepoints_fut, first(test_loader), config["training"]["validation"]; sample_n=3, var_of_intrst=4);