function loss_fn(model, θ, st, data; β=0.01f0)
    (u_obs, covars_obs, x_obs, y_obs, mask_obs,
        u_forecast, covars_forecast, x_forecast, y_forecast, mask_forecast), ts, λ = data
    batch_size = size(x_forecast)[end]

    ts_obs, ts_for = ts
    y_enc = vcat(covars_obs, y_obs, mask_obs)
    ŷ, px₀, kl_pq = model(y_enc, u_forecast, (ts_obs, ts_for), θ, st)

    μ, log_σ² = ŷ
    forecast_loss = normal_loglikelihood(μ .* mask_forecast, log_σ² .* mask_forecast, y_forecast .* mask_forecast) / batch_size

    α = 1.0f0
    kl_init = α * kl_normal(px₀...) / batch_size
    if kl_pq === nothing
        kl_path = 0.0f0
    else
        kl_path = β * mean(kl_pq[end, :])
    end

    loss = forecast_loss + kl_init + λ * kl_path
    return loss, st, (kl_path, kl_init, forecast_loss)
end
