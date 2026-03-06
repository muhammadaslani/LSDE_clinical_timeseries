function loss_fn(model, θ, st, data; β=0.05f0)
    (u_obs, covars_obs, x_obs, y₁_obs, y₂_obs, mask₁_obs, mask₂_obs,
        u_for, covars_for_, x_forecast, y₁_for, y₂_for, mask₁_for, mask₂_for), ts, λ = data
    batch_size = size(x_forecast)[end]

    y_enc = vcat(covars_obs, y₁_obs, log.(y₂_obs .+ 1))
    (ŷ₁, ŷ₂), px₀, kl_pq = model(y_enc, u_for, ts, θ, st)

    recon_loss1 = CrossEntropy_Loss(ŷ₁, y₁_for, mask₁_for; agg=sum) / batch_size
    recon_loss2 = -poisson_loglikelihood(ŷ₂, y₂_for, mask₂_for) / batch_size

    kl_init = kl_normal(px₀...) / batch_size
    if kl_pq === nothing
        kl_path = 0.0f0
    else
        kl_path = β * mean(kl_pq[end, :])
    end

    # kl_init is always active (tight initial conditions); only path KL is annealed
    loss = recon_loss1 + recon_loss2 + kl_init + λ * kl_path
    return loss, st, (kl_path, kl_init, recon_loss1, recon_loss2)
end
