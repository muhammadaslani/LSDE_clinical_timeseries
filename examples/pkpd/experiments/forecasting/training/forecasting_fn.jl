function forecast_nde(model, θ, st, obs_data, u_forecast, time_forecast, config)
    _, covars_obs, _, y₁_obs, y₂_obs, _, _ = obs_data
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    Ex, Ey_p = predict(model, solver, vcat(covars_obs, y₁_obs, y₂_obs), u_forecast, time_forecast, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    return Ex, Ey_p
end

function forecast_lstm(model, θ, st, obs_data, u_forecast, time_forecast, config)
    _, covars_obs, _, y₁_obs, y₂_obs, _, _ = obs_data
    Ex_p, Ey_p = predict(model, vcat(covars_obs, y₁_obs, y₂_obs), u_forecast, time_forecast, θ, st, config["mcmc_samples"], model.device)

    return Ex_p, Ey_p
end
