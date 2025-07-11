function forecast_nde(model, θ, st, obs_data, u_forecast, time_forecast, config)
    _, covars_obs, _, y₁_obs, y₂_obs, _, _ = obs_data
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    #Ex, Ey_p = predict(model, solver, vcat(covars_obs, y₁_obs, y₂_obs), u_forecast, time_forecast, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    Ey_p, Ex, x̂ = predict(model, vcat(covars_obs, y₁_obs, y₂_obs), u_forecast, time_forecast, θ, st, config["mcmc_samples"])
    return Ex, Ey_p
end

function forecast_rnn(model, θ, st, obs_data, u_forecast, time_forecast, config)
    u_obs, covars_obs, _, y₁_obs, y₂_obs, _, _ = obs_data
    history = vcat(covars_obs, y₁_obs, y₂_obs, u_obs)
    forecast_length = size(u_forecast, 2)
    Ey_p, st = predict_rnn(model, history, u_forecast, forecast_length, θ, st; mcmc_samples=config["mcmc_samples"])
    Ex= Ey_p
    return Ex, Ey_p
end
