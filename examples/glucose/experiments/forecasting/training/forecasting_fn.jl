function forecast(model, θ, st, obs_data, u_forecast, timepoints, config)
    u_obs, covars_obs, _, y_obs, _ = obs_data
    y_enc = vcat(covars_obs, y_obs, u_obs)

    if model isa LatentSDE || model isa LatentODE
        solver = eval(Meta.parse(config["solver"]))
        kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
        Ex, Ey_p = predict(model, solver, y_enc, u_forecast, timepoints, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    else
        Ex, Ey_p = predict(model, y_enc, u_forecast, timepoints, θ, st, config["mcmc_samples"], cpu_device())
    end
    return Ex, Ey_p
end
