function loss_fn(model, θ, st, data; β=1.0f0)
    (u_obs, covars_obs, x_obs, y_obs, mask_obs,
        u_forecast, covars_forecast, x_forecast, y_forecast, mask_forecast), ts, λ = data
    batch_size = size(x_forecast)[end]

    ts_obs, ts_for = ts
    y_enc = vcat(covars_obs, y_obs, u_obs, mask_obs)
    ŷ, px₀, kl_pq = model(y_enc, u_obs, (ts_obs, ts_for), θ, st)

    μ, log_σ² = ŷ
    recon_loss = normal_loglikelihood(μ .* mask_obs, log_σ² .* mask_obs, y_obs .* mask_obs)

    kl_init = kl_normal(px₀...)
    if kl_pq === nothing
        kl_path = 0.0f0
        kl_loss = kl_init
    else
        kl_path = mean(kl_pq[end, :])
        kl_loss = kl_init + β *  kl_path
    end

    loss = recon_loss + λ * kl_loss
    return loss, st, (kl_path, kl_init, recon_loss)
end
