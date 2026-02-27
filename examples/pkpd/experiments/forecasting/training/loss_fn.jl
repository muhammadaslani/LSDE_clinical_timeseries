function loss_fn(model, θ, st, data; β=1.0f0)
    (u_obs, covars_obs, _, y₁_obs, y₂_obs, mask₁_obs, mask₂_obs,
        _, _, x_forecast, _, _, _, _), ts, λ = data
    batch_size = size(x_forecast)[end]

    y_enc = vcat(covars_obs, y₁_obs, log.(y₂_obs .+ 1))
    (ŷ₁, ŷ₂), px₀, kl_pq = model(y_enc, u_obs, ts, θ, st)

    recon_loss1 = CrossEntropy_Loss(ŷ₁, y₁_obs, mask₁_obs; agg=sum) / batch_size
    recon_loss2 = -0.1*poisson_loglikelihood(ŷ₂, y₂_obs, mask₂_obs) / batch_size

    kl_init = kl_normal(px₀...) / batch_size
    if kl_pq === nothing
        kl_path = 0.0f0
        kl_loss = kl_init
    else
        kl_path = mean(kl_pq[end, :])
        kl_loss = kl_init + β * kl_path
    end

    loss = recon_loss1 + recon_loss2 + λ * kl_loss
    return loss, st, (kl_path, kl_init, recon_loss1, recon_loss2)
end
