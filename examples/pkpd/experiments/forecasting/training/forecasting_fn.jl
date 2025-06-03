# Forecasting functions for PKPD models

function forecast_nde(model, θ, st, obs_data, u_forecast, time_forecast, config)
    u_obs, covars_obs, x_obs, y₁_obs, y₂_obs, mask₂_obs = obs_data
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    Ex, Ey_p = predict(model, solver, vcat(covars_obs, reverse(y₁_obs, dims=2), reverse(y₂_obs, dims=2)), u_forecast, time_forecast, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    return Ex, Ey_p
end

function forecast_rnn(model, θ, st, obs_data, u_forecast, time_forecast, config)
    u_obs, covars_obs, x_obs, y₁_obs, y₂_obs, masks₁_obs, masks₂_obs = obs_data
    
    # Combine inputs for RNN
    input_combined = vcat(x_obs, u_obs, covars_obs)
    
    # Forward pass through RNN
    ŷ, st = model(input_combined, θ, st)
    
    # Extract means and standard deviations
    μ = [ŷ[i][1] for i in eachindex(ŷ)]
    σ = [sqrt.(exp.(ŷ[i][2])) for i in eachindex(ŷ)]
    
    return μ, σ
end

# Alias for backward compatibility with existing PKPD code
function forecast(model, θ, st, obs_data, u_forecast, time_forecast, config)
    return forecast_nde(model, θ, st, obs_data, u_forecast, time_forecast, config)
end
