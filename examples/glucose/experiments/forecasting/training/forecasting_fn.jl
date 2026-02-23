function forecast_nde(model, θ, st, obs_data, u_forecast, time_forecast, config)
    _, covars_obs, _, y_obs, _ = obs_data
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    Ex, Ey_p = predict(model, solver, vcat(covars_obs, y_obs), u_forecast, time_forecast, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    return Ex, Ey_p
end

function forecast_lstm(model, θ, st, obs_data, u_forecast, time_forecast, config)
    _, covars_obs, _, y_obs, _ = obs_data
    Ex_p, Ey_p = predict(model, vcat(covars_obs, y_obs), u_forecast, time_forecast, θ, st, config["mcmc_samples"], model.device)
    return Ex_p, Ey_p
end

function forecast_cde(model, θ, st, obs_data, u_forecast, time_forecast, config)
    u_obs, covars_obs, _, y_obs, _ = obs_data
    y_enc = vcat(covars_obs, y_obs, u_obs)
    # time_forecast is already (ts_obs, ts_for) tuple from kfold_trainer
    Ex_p, Ey_p = predict(model, y_enc, u_forecast, time_forecast, θ, st, config["mcmc_samples"], model.device)
    return Ex_p, Ey_p
end

