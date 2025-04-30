##dependencies
using Revise, Rhythm, Lux, DifferentialEquations, Random, SciMLSensitivity, ComponentArrays, Optimisers, OptimizationOptimisers, Statistics
using MLUtils, Printf, SciMLSensitivity, OneHotArrays, CairoMakie, Distributions
using YAML
using DataFrames, CSV
include("data_prep.jl");

##loading data
variables_of_interest = ["MAP", "HR", "Temp"];
n_features = length(variables_of_interest);
data, train_loader, val_loader, test_loader, time_series_dataset = load_data(split_at=36, n_samples=256, batch_size=32, variables_of_interest=variables_of_interest);
inputs_data_obs, obs_data_obs, output_data_obs, masks_obs, inputs_data_for, obs_data_for, output_data_for, masks_for = data;

n_timepoints = size(hcat(obs_data_obs, obs_data_for))[2]

tspan = (1.0, n_timepoints)
timepoints = (range(tspan[1], tspan[2], length=n_timepoints)) / (n_timepoints) |> Array{Float32};

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
    u_obs, x_obs, y_obs, masks_obs, u_for, x_for, y_for, masks_for = data
    batch_size = size(y_for)[end]
    ŷ, st = model(x_obs, θ, st)
    μ = [ŷ[i][1] for i in eachindex(ŷ)]
    log_σ = [ŷ[i][2] for i in eachindex(ŷ)]
    recon_loss = 0.0
    for i in eachindex(ŷ)
        valid_indx = findall(masks_for[i, :, :] .== 1)
        recon_loss += normal_loglikelihood(μ[i][valid_indx], log_σ[i][valid_indx], y_for[i, valid_indx]) / batch_size
    end
    kl = 0.0
    return recon_loss, st, kl
end


## defining the visualization function
function forecast(model, θ, st, obs_data, u_forecast, config)
    u_obs, x_obs, y_obs, masks_obs = obs_data
    ŷ, st = model(x_obs, θ, st)
    μ = [ŷ[i][1] for i in eachindex(ŷ)]
    σ = [exp.(ŷ[i][2]) for i in eachindex(ŷ)]
    return μ, σ
end

function viz_fn_forecast(t_obs, t_for, obs_data, future_true_data, forecasted_data; sample_n=1)
    u_obs, x_obs, y_obs, masks_obs = obs_data
    u_for, x_for, y_for, masks_for = future_true_data
    μ, σ = forecasted_data
    t_obs = t_obs .* 50
    t_for = t_for .* 50

    y_labels = variables_of_interest
    fig = Figure(size=(1200, 600), fontsize=15)
    axes = CairoMakie.Axis[]
    rmse = []
    # Validate observation indices
    for i in 1:n_features
        valid_indx_obs = findall(masks_obs[i, :, :] .== 1)
        valid_indx_for = findall(masks_for[i, :, :] .== 1)
        y_label = y_labels[i]
        if isempty(findall(masks_obs[i, :, sample_n] .== 1))
            println("No observations data is available for $y_label in this sample: valid_indx_obs is empty.")
            ax = CairoMakie.Axis(fig[i, 1], xlabel="Time (hours)", ylabel=y_labels[i], xgridvisible=false, ygridvisible=false)
            push!(axes, ax)
            continue
        end
        if isempty(findall(masks_for[i, :, sample_n] .== 1))
            println("No future data is available for $y_label in this sample: valid_indx_for is empty.")
            ax = CairoMakie.Axis(fig[i, 1], xlabel="Time (hours)", ylabel=y_labels[i], xgridvisible=false, ygridvisible=false)
            push!(axes, ax)
            continue
        end

        # Extract valid time points and observations
        y_obs_val = y_obs[i, valid_indx_obs]
        y_for_val = y_for[i, valid_indx_for]

        # Extract mean and variance for predictions based on the predicted gaussian distribution for each output (μ,σ)
        # Generate predicted distributions
        dists = Normal.(μ[i], sqrt.(σ[i]))
        ŷ = [rand.(dists) for _ in 1:50]
        ŷ = cat(ŷ..., dims=3)
        # Calculate mean and standard deviation of predictions for whole dataset
        ŷ_mean = dropdims(mean(ŷ, dims=3), dims=3)
        ŷ_std = dropdims(std(ŷ, dims=3), dims=3)
        ŷ_mean_val = ŷ_mean[valid_indx_for]
        ŷ_std_val = ŷ_std[valid_indx_for]
        rmse_ = sqrt(MSELoss()(ŷ_mean_val, y_for_val))
        println("RMSE for $y_label: ", rmse_)
        push!(rmse, rmse_)
        # Calculate mean and standard deviation of predictions for sample number sample_n
        ŷ_std_val_error = ŷ_std[masks_for[i, :, sample_n].==1, sample_n] / sqrt(length(ŷ_mean[masks_for[i, :, sample_n].==1, sample_n]))
        ŷ_ci_lower = ŷ_mean[masks_for[i, :, sample_n].==1, sample_n] - 1.96 * ŷ_std_val_error
        ŷ_ci_upper = ŷ_mean[masks_for[i, :, sample_n].==1, sample_n] + 1.96 * ŷ_std_val_error

        # Valid time points for observations and future data
        t_obs_val = t_obs[masks_obs[i, :, sample_n].==1]
        t_for_val = t_for[masks_for[i, :, sample_n].==1]
        # Plot the results
        ax = CairoMakie.Axis(fig[i, 1], xlabel="Time (hours)", ylabel=y_labels[i], xgridvisible=false, ygridvisible=false)
        push!(axes, ax)
        scatter!(ax, t_obs_val, y_obs[i, masks_obs[i, :, sample_n].==1, sample_n], color=:blue, label="Past Observations", markersize=10)
        lines!(ax, t_obs_val, y_obs[i, masks_obs[i, :, sample_n].==1, sample_n], color=(:blue, 0.4), linewidth=2, linestyle=:dot)
        scatter!(ax, t_for_val, y_for[i, masks_for[i, :, sample_n].==1, sample_n], color=:green, label="Future Ground Truth", markersize=10)
        lines!(ax, t_for_val, y_for[i, masks_for[i, :, sample_n].==1, sample_n], color=(:green, 0.4), linestyle=:dot)
        scatter!(ax, t_for_val, ŷ_mean[masks_for[i, :, sample_n].==1, sample_n], color=:red, label="Model Predictions", markersize=10)
        lines!(ax, t_for_val, ŷ_mean[masks_for[i, :, sample_n].==1, sample_n], color=(:red, 0.4), linestyle=:dot)
        band!(ax, t_for_val, ŷ_ci_lower, ŷ_ci_upper, color=:red, alpha=0.3)
        # Add poly! with labels only for the top subplot (i == 1)
        if i == 1
            poly!(ax, [0, t_obs[end], t_obs[end], 0], [-10, -10, 500, 500], color=(:blue, 0.1), label="Observation Period (Past)")
            poly!(ax, [t_obs[end], t_for[end], t_for[end], t_obs[end]], [-10, -10, 500, 500], color=(:red, 0.1), label="Forecasting Period (Future)")
        else
            poly!(ax, [0, t_obs[end], t_obs[end], 0], [-10, -10, 500, 500], color=(:blue, 0.1))
            poly!(ax, [t_obs[end], t_for[end], t_for[end], t_obs[end]], [-10, -10, 500, 500], color=(:red, 0.1))
        end
        all_y_values = vcat(y_obs[i, masks_obs[i, :, sample_n].==1, sample_n], y_for[i, masks_for[i, :, sample_n].==1, sample_n], ŷ_mean[masks_for[i, :, sample_n].==1, sample_n], ŷ_ci_lower, ŷ_ci_upper)  # Combine all y-data

        y_min = minimum(all_y_values) - 0.1(maximum(all_y_values) - minimum(all_y_values))   # Add some padding
        y_max = maximum(all_y_values) + 0.1(maximum(all_y_values) - minimum(all_y_values))  # Add some padding
        # if y_labels[i] == "Temp"
        #     y_min = minimum(all_y_values) - 0.5 # Add some padding
        #     y_max = maximum(all_y_values) + 0.5  # Add some padding
        # end

        ylims!(ax, y_min, y_max)
        # Add legend only for the top subplot (i == 1)
        if i == 1
            fig[i, 2] = Legend(fig, ax, framevisible=false, halign=:left)
        end
    end
    linkxaxes!(axes...)
    colgap!(fig.layout, 10)
    display(fig)
    return fig, rmse
end

function train(model, θ, st, loss_fn, train_loader, val_loader, config)
    # Create optimizer from config
    opt = eval(Meta.parse(config["optimizer"]))
    tstate = Training.TrainState(model, θ, st, opt)
    n_batches = length(train_loader)
    θ_best = nothing
    best_val_metric = Inf
    counter = 0
    @info "Training started"
    for epoch in 1:config["epochs"]
        stime = time()
        train_loss = 0.0f0
        for batch in train_loader
            _, loss, kl, tstate = Training.single_train_step!(AutoZygote(), loss_fn, batch, tstate)
            train_loss += loss
        end

        θ = tstate.parameters
        st = tstate.states

        if epoch % config["log_freq"] == 0
            ttime = time() - stime
            @printf("Epoch %d/%d: \t Training loss: %.3f Time/epoch: %.3f\n",
                epoch, config["epochs"], train_loss / n_batches, ttime / config["log_freq"])
            val_metric = validate(model, θ, st, val_loader)
            @printf("Validation metric: %.3f\n", val_metric)

            if epoch % config["viz_freq"] == 0
                #viz_fn(model, θ, st, ts, first(train_loader), config["validation"]; sample_n=1)
            end

            if val_metric < best_val_metric
                @info "Saving best model!"
                best_val_metric = val_metric
                θ_best = θ
                save_state = (θ=θ_best, st=st, epoch=epoch)
                #save_object(joinpath(exp_path, "bestmodel.jld2"), save_state)
                counter = 0
            else
                if counter > config["stop_patience"]
                    @printf("No more hope training this one! Early stopping at epoch: %.f\n", epoch)
                    return θ_best
                elseif counter > config["lrdecay_patience"]
                    new_lr = config["learning_rate"] / counter
                    @printf("No improvement for %d consecutive epochs; Adjusting learning rate to: %.4f\n",
                        config["lrdecay_patience"], new_lr)
                    Optimisers.adjust!(tstate.optimizer_state, new_lr)
                end
                counter += 1
            end

        end

    end
    return θ_best
    

end

function validate(model, θ, st, val_loader)
    val_metric = 0.0
    for batch in val_loader
        val_metric += loss_fn(model, θ, st, batch)[1]
    end
    return val_metric / length(val_loader)
end




## model, training, and inference
rng = Random.MersenneTwister(123);
config_rnn = YAML.load_file("configs/RNN_config.yml");
hidden_dim = config_lsde["model"]["obs_encoder"]["hidden_size"];
latent_dim = config_lsde["model"]["latent_dim"];
n_timepoints_for = size(output_data_for, 2)

model = Chain(
    encoder=Chain(
        Recurrence(LSTMCell(dims["obs_dim"] => hidden_dim); return_sequence=true),
        Recurrence(LSTMCell(hidden_dim => latent_dim); return_sequence=true));

    decoder=Chain(
        Recurrence(LSTMCell(latent_dim => hidden_dim); return_sequence=true),
        Recurrence(LSTMCell(hidden_dim => hidden_dim); return_sequence=false),
        BranchLayer(BranchLayer(Dense(hidden_dim, n_timepoints_for), Dense(hidden_dim, n_timepoints_for, softplus)),
            BranchLayer(Dense(hidden_dim, n_timepoints_for), Dense(hidden_dim, n_timepoints_for, softplus)),
            BranchLayer(Dense(hidden_dim, n_timepoints_for), Dense(hidden_dim, n_timepoints_for, softplus)))
    )
)
# Parameter and State Variables
θ, st = Lux.setup(rng, model);
θ_trained = train(model, θ, st, loss_fn, train_loader, val_loader, config_rnn["training"]);

## forecasting
u_obs, x_obs, y_obs, masks_obs, u_forecast, x_forecast, y_forecast, masks_forecast = test_loader.data;
data_obs = (u_obs, x_obs, y_obs, masks_obs);
future_true_data = (u_forecast, x_forecast, y_forecast, masks_forecast);
t_for = timepoints_for;
t_obs = timepoints_obs;

## RNN forecast
μ, σ = forecast(model, θ_trained, st, data_obs, u_forecast, config_lsde["training"]["validation"]);
rnn_forecasted_data = (μ, σ);
fig, rmse = viz_fn_forecast(t_obs, t_for, data_obs, future_true_data, rnn_forecasted_data; sample_n=3);
#save("examples/ICU/ICU_lsde_forecast.eps", fig)
