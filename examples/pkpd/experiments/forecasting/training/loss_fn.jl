function loss_fn_nde(model, θ, st, data)
    (u_obs, covars_obs, _, y₁_obs, y₂_obs, _, _,
        u_forecast, _, x_forecast, y₁_forecast, y₂_forecast, mask₁_forecast, mask₂_forecast), ts, λ = data
    batch_size = size(x_forecast)[end]
    (ŷ₁, ŷ₂), px₀, kl_pq = model(vcat(covars_obs, y₁_obs, y₂_obs), hcat(u_obs, u_forecast), ts, θ, st)
    recon_loss1 = CrossEntropy_Loss(ŷ₁, y₁_forecast, mask₁_forecast; agg=sum) / batch_size
    recon_loss2 = -poisson_loglikelihood(ŷ₂, y₂_forecast, mask₂_forecast) / batch_size
    recon_loss = recon_loss1 + recon_loss2
    if kl_pq === nothing
        # For ODE models, only use the initial state KL divergence
        kl_loss = kl_normal(px₀...) / batch_size
    else
        # For SDE models, include both initial state and path KL divergences
        kl_loss = kl_normal(px₀...) / batch_size + mean(kl_pq[end, :])
    end
    
    loss = recon_loss + λ * kl_loss
    return loss, st, (kl_loss, recon_loss, recon_loss1, recon_loss2)
end


function loss_fn_rnn(model, θ, st, data)
    (u_obs, covars_obs, _, y₁_obs, y₂_obs, _, _,
        u_forecast, _, _, y₁_forecast, y₂_forecast, mask₁_forecast, mask₂_forecast), ts, λ = data

    forecast_length = size(u_forecast, 2)
    batch_size = size(y₁_forecast)[end]
    history = vcat(covars_obs, y₁_obs, y₂_obs, u_obs)
    (ŷ₁, ŷ₂), st, vae_params = model(history, u_forecast, forecast_length, θ, st)
    μ, logσ² = vae_params.μ, vae_params.logσ²

    recon_loss1 = CrossEntropy_Loss(ŷ₁, y₁_forecast, mask₁_forecast; agg=sum) / batch_size
    recon_loss2 = -poisson_loglikelihood(ŷ₂, y₂_forecast, mask₂_forecast) / batch_size
    total_recon_loss = recon_loss1 + recon_loss2
    kl_loss = kl_normal(μ, exp.(logσ²)) / batch_size
    total_loss = recon_loss1 + recon_loss2 + λ * kl_loss
    return total_loss, st, (kl_loss, total_recon_loss, recon_loss1, recon_loss2)
end
