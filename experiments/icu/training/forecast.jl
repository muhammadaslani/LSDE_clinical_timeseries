# Forecast function — matches PKPD/glucose pattern, always uses solver from config
function forecast(model, θ, st, obs_data, u_forecast, timepoints, config)
    x_obs, _, _, _, x_masks_obs = obs_data
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])

    # Augment x_obs with observation masks (same as in loss_fn/eval_fn)
    n_static = size(x_obs, 1) - size(x_masks_obs, 1)
    static_ones = ones(Float32, n_static, size(x_obs, 2), size(x_obs, 3))
    x_mask_full = vcat(static_ones, Float32.(x_masks_obs))
    x_aug = vcat(x_obs, x_mask_full)

    Ex, Ey = predict(model, solver, x_aug, u_forecast, timepoints, θ, st,
        config["mcmc_samples"], cpu_device(); kwargs_dict...)
    return Ex, Ey
end


