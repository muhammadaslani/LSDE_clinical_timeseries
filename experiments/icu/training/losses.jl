function loss_fn(model, θ, st, data; β=0.01f0)
    (x_obs, u_obs, y_obs, y_masks_obs, x_masks_obs,
        x_fut, u_fut, y_fut, y_masks_fut, x_fut_masks), ts, λ = data
    batch_size = size(y_fut)[end]

    # Augment x_obs with its observation mask so the encoder can distinguish
    # real measurements from mean-imputed values.
    n_static = size(x_obs, 1) - size(x_masks_obs, 1)
    static_ones = ones(Float32, n_static, size(x_obs, 2), batch_size)
    x_mask_full = vcat(static_ones, Float32.(x_masks_obs))
    x_aug = vcat(x_obs, x_mask_full)

    # Forecast: encoder sees obs, dynamics driven by forecast-period controls
    ŷ, px₀, kl_pq = model(x_aug, u_fut, ts, θ, st)

    forecast_losses = map(eachindex(ŷ)) do i
        μ, log_σ² = ŷ[i][1], ŷ[i][2]
        normal_loglikelihood(
            μ[1, :, :],
            log_σ²[1, :, :],
            y_fut[i, :, :],
            y_masks_fut[i, :, :]
        ) / batch_size
    end
    forecast_loss = sum(forecast_losses)

    kl_init = kl_normal(px₀...) / batch_size
    if kl_pq === nothing
        kl_path = 0.0f0
    else
        kl_path = β * mean(kl_pq[end, :])
    end

    loss = forecast_loss + kl_init + λ * kl_path
    return loss, st, (kl_path, kl_init, forecast_losses...)
end
