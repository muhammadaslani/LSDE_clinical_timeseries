# Forecasting functions for PKPD models

function forecast_nde(model, θ, st, obs_data, u_forecast, time_forecast, config)
    u_obs, covars_obs, x_obs, y₁_obs, y₂_obs, masks₁_obs, masks₂_obs = obs_data
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    Ex, Ey_p = predict(model, solver, vcat(covars_obs, reverse(y₁_obs, dims=2), reverse(y₂_obs, dims=2)), u_forecast, time_forecast, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    return Ex, Ey_p
end

function forecast_rnn(model, θ, st, obs_data, u_forecast, time_forecast, config)
    u_obs, covars_obs, x_obs, y₁_obs, y₂_obs, masks₁_obs, masks₂_obs = obs_data
    
    # Combine inputs: covariates + health status + cell count + control inputs + observations
    history = vcat(covars_obs, y₁_obs, y₂_obs, u_obs)
    forecast_length = size(u_forecast, 2)
    
    # Forward pass through RNN
    ŷ, st = model(history, u_forecast, forecast_length, θ, st)
    
    ŷ₁, ŷ₂ = ŷ[1], ŷ[2]  # Health status logits, Cell count rates
    Ex = nothing  # RNN doesn't have latent state trajectory like NDE models
    Ey_p = (ŷ₁, ŷ₂)  # Tuple of predictions for each output
    
    return Ex, Ey_p
end
