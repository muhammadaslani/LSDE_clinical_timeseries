function loss_fn(model, θ, st, data)
    (u_obs, covars_obs, x_obs, y_obs, mask_obs,
        u_forecast, covars_forecast, x_forecast, y_forecast, mask_forecast), ts, λ = data
    batch_size = size(x_forecast)[end]

    ts_obs, ts_for = ts
    y_enc = vcat(covars_obs, y_obs, u_obs)
    (ŷ_glucose,), px₀, kl_pq = model(y_enc, u_forecast, (ts_obs, ts_for), θ, st)

    μ, log_σ² = ŷ_glucose
    recon_loss = normal_loglikelihood(μ .* mask_forecast, log_σ² .* mask_forecast, y_forecast .* mask_forecast)

    if kl_pq === nothing
        kl_loss = kl_normal(px₀...)
    else
        kl_loss = kl_normal(px₀...) + mean(kl_pq[end, :])
    end

    train_rmse = sqrt.(mse(μ .* mask_forecast, y_forecast .* mask_forecast))
    loss = recon_loss + λ * kl_loss
    return loss, st, (kl_loss, recon_loss, recon_loss, train_rmse)
end
