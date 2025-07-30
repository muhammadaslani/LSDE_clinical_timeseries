function forecast_nde(model, θ, st, obs_data, u_forecast, time_forecast, config)
    u_obs, x_obs, _, _ = obs_data    
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    _, Ey = predict(model, solver, x_obs, hcat(u_obs,u_forecast), time_forecast, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    μ = [Ey[i][1] for i in eachindex(Ey)]
    σ = [sqrt.(exp.(Ey[i][2])) for i in eachindex(Ey)]
    return μ, σ
end 
# Forecasting function for RNN models
function forecast_lstm(model, θ, st, obs_data, u_forecast, time_forecast, config)
    _, x_obs, _, _ = obs_data
    _, Ey_p = predict(model, x_obs, u_forecast, time_forecast, θ, st, config["mcmc_samples"], model.device)
    μ = [Ey_p[i][1] for i in eachindex(Ey_p)]
    σ = [sqrt.(exp.(Ey_p[i][2])) for i in eachindex(Ey_p)]
    return μ, σ
end