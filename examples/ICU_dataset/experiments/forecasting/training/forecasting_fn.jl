# functions for forecasting with different models

function forecast_nde(model, θ, st, obs_data, u_forecast, time_forecast, config)
    u_obs, x_obs, y_obs, masks_obs = obs_data    
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    _, Ey = predict(model, solver, x_obs, hcat(u_obs,u_forecast), time_forecast, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    μ = [Ey[i][1] for i in eachindex(Ey)]
    σ = [sqrt.(exp.(Ey[i][2])) for i in eachindex(Ey)]
    return μ, σ
end 
# Forecasting function for RNN models
function forecast_rnn(model, θ, st, obs_data, u_forecast, time_forecast, config)
    u_obs, x_obs, y_obs, masks_obs = obs_data
    history_data = vcat(x_obs, u_obs)
    forecast_length = size(u_forecast)[2]
    ŷ, st = model(history_data, u_forecast, forecast_length, θ, st)
    μ = [ŷ[i][1] for i in eachindex(ŷ)]
    σ = [sqrt.(exp.(ŷ[i][2])) for i in eachindex(ŷ)]
    return μ, σ
end