# Loss functions for PKPD forecasting models

function loss_fn_nde(model, θ, st, data)
    (u_obs, covars_obs, x_obs, y₁_obs, y₂_obs, mask₁_obs, mask₂_obs,
     u_forecast, covars_forecast, x_forecast, y₁_forecast, y₂_forecast, mask₁_forecast, mask₂_forecast ), ts, λ = data
    batch_size= size(x_forecast)[end]
    ŷ, px₀, kl_pq = model(vcat(covars_obs, y₁_obs, y₂_obs), hcat(u_obs,u_forecast), ts, θ, st)
    ŷ₁, ŷ₂ = ŷ[1], ŷ[2]
    recon_loss1 =CrossEntropy_Loss(ŷ₁, y₁_forecast, mask₁_forecast; agg=sum)/batch_size
    recon_loss2 = -poisson_loglikelihood(ŷ₂, y₂_forecast, mask₂_forecast)/batch_size
    recon_loss = recon_loss1 + recon_loss2
    kl_loss = kl_normal(px₀...)/batch_size + mean(kl_pq[end, :])
    loss = recon_loss + λ * kl_loss
    return loss, st, (kl_loss, recon_loss, recon_loss1, recon_loss2)
end

function loss_fn_rnn(model, θ, st, data)
    (u_obs, covars_obs, x_obs, y₁_obs, y₂_obs, mask₁_obs, mask₂_obs,
     u_forecast, covars_forecast, x_forecast, y₁_forecast, y₂_forecast, mask₁_forecast, mask₂_forecast ), ts, λ = data

    batch_size = size(y₁_forecast)[end]
    input_combined = vcat(covars_obs, y₁_obs, y₂_obs, u_obs)
    
    # Forward pass
    ŷ, st = model(input_combined, θ, st)
    # Calculate reconstruction loss

    recon_loss1 = CrossEntropy_Loss(ŷ[1], y₁_forecast, mask₁_forecast; agg=sum) / batch_size
    recon_loss2 = -poisson_loglikelihood(ŷ[2], y₂_forecast, mask₂_forecast) / batch_size

    total_recon_loss = recon_loss1 + recon_loss2
    kl_loss = 0.0f0  # RNN doesn't have KL divergence
    
    return total_recon_loss, st, (kl_loss, total_recon_loss, recon_loss1, recon_loss2)
end
