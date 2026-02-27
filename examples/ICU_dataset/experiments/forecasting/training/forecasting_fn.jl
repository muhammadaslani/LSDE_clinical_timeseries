# Forecast function — matches PKPD/glucose pattern, always uses solver from config
function forecast(model, θ, st, obs_data, u_forecast, timepoints, config)
    _, x_obs, _, _ = obs_data
    solver      = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    Ex, Ey = predict(model, solver, x_obs, u_forecast, timepoints, θ, st,
                     config["mcmc_samples"], cpu_device(); kwargs_dict...)
    return Ex, Ey
end


