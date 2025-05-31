function viz_fn_forecast_nde(t_obs, t_for, obs_data, future_true_data, forecasted_data; sample_n=1, plot=true)
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
        ŷ = rand.(dists)

        ŷ_mean = dropdims(mean(ŷ, dims=4), dims=4)
        ŷ_std = dropdims(std(ŷ, dims=4), dims=4)

        ŷ_mean_val = ŷ_mean[1, valid_indx_for]
        ŷ_std_val = ŷ_std[1, valid_indx_for]

        crps_ = empirical_crps(y_for[i:i, :, :], ŷ, masks_for[i:i, :, :])
        rmse_ = sqrt(MSELoss()(ŷ_mean_val, y_for_val))

        push!(crps, crps_)
        push!(rmse, rmse_)

        ŷ_ci_lower = ŷ_mean .- ŷ_std
        ŷ_ci_upper = ŷ_mean .+ ŷ_std

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

                scatter!(ax, t_for_val, ŷ_mean[1, masks_for[i, :, sample_n] .== 1, sample_n], color=:red, label="Model Predictions", markersize=10)
                lines!(ax, t_for_val, ŷ_mean[1, masks_for[i, :, sample_n] .== 1, sample_n], color=(:red, 0.4), linestyle=:dot)

                band!(ax, t_for_val, ŷ_ci_lower[1, masks_for[i, :, sample_n] .== 1, sample_n], ŷ_ci_upper[1, masks_for[i, :, sample_n] .== 1, sample_n], color=:red, alpha=0.2)

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
                    ŷ_mean[1, masks_for[i, :, sample_n] .== 1, sample_n],
                    ŷ_ci_lower[1, masks_for[i, :, sample_n] .== 1, sample_n],
                    ŷ_ci_upper[1, masks_for[i, :, sample_n] .== 1, sample_n]
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



function viz_fn_forecast_rnn(t_obs, t_for, obs_data, future_true_data, forecasted_data; sample_n=1, plot=true)
    u_obs, x_obs, y_obs, masks_obs = obs_data
    u_for, x_for, y_for, masks_for = future_true_data
    μ, σ = forecasted_data
    t_obs = t_obs * 10 
    t_for = t_for * 10 

    y_labels = ["MAP", "HR", "BT"]
    fig = Figure(size=(1200, 600), fontsize=15)
    axes = CairoMakie.Axis[]
    rmse = []

    n_features = length(y_labels)  # Assuming one label per feature

    for i in 1:n_features
        y_label = y_labels[i]
        valid_indx_obs = findall(masks_obs[i, :, :] .== 1)
        valid_indx_for = findall(masks_for[i, :, :] .== 1)

        t_obs_val = t_obs[masks_obs[i, :, sample_n] .== 1]
        t_for_val = t_for[masks_for[i, :, sample_n] .== 1]

        y_obs_val = y_obs[i, valid_indx_obs]
        y_for_val = y_for[i, valid_indx_for]

        # Generate predicted distributions (RNN format)
        dists = Normal.(μ[i], σ[i])
        ŷ = rand.(dists)

        ŷ_val = ŷ[valid_indx_for]
        rmse_ = sqrt(MSELoss()(ŷ_val, y_for_val))

        push!(rmse, rmse_)

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

                scatter!(ax, t_for_val, ŷ[ masks_for[i, :, sample_n] .== 1, sample_n], color=:red, label="Model Predictions", markersize=10)
                lines!(ax, t_for_val, ŷ[ masks_for[i, :, sample_n] .== 1, sample_n], color=(:red, 0.4), linestyle=:dot)
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
                    ŷ[ masks_for[i, :, sample_n] .== 1, sample_n],
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
        # RNN models only return RMSE, no CRPS
        return fig, rmse
    else
        # RNN models only return RMSE, no CRPS
        return rmse
    end
end