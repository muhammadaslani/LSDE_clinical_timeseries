function loss_fn_nde(model, θ, st, data)
    (u_obs, x_obs, _, _, u_for, _, y_for, masks_for), ts, λ = data
    batch_size= size(y_for)[end]
    ŷ, px₀, kl_pq = model(x_obs,  u_for, ts, θ, st)
    recon_loss = 0.0f0
    for i in eachindex(ŷ)
        μ, log_σ² = ŷ[i][1], ŷ[i][2]
        valid_indx= findall(masks_for[i, :, :] .== 1)
        recon_loss += normal_loglikelihood(μ[1,valid_indx], log_σ²[1,valid_indx], y_for[i, valid_indx])/batch_size
    end 
    if kl_pq === nothing
        # For ODE models, only use the initial state KL divergence
        kl_loss = kl_normal(px₀...) / batch_size
    else
        # For SDE models, include both initial state and path KL divergences
        kl_loss = kl_normal(px₀...) / batch_size + mean(kl_pq[end, :])
    end
    
    loss = recon_loss + λ * kl_loss
    return loss, st, (kl_loss, recon_loss, 0.0f0, 0.0f0)
end


function loss_fn_lstm(model, θ, st, data)
    (_, x_obs, _, _, u_for, _, y_for, masks_for), ts, λ = data
    batch_size = size(y_for)[end]

    ŷ, px₀, kl_path = model(x_obs, u_for, ts, θ, st)
    recon_loss = 0.0f0
    for i in eachindex(ŷ)
        μ, log_σ² = ŷ[i][1], ŷ[i][2]
        valid_indx = findall(masks_for[i, :, :] .== 1)
        recon_loss += normal_loglikelihood(μ[1, valid_indx], log_σ²[1, valid_indx], y_for[i, valid_indx]) / batch_size
    end
    kl_loss = kl_normal(px₀...) / batch_size
    total_loss = recon_loss + λ * kl_loss
    return total_loss, st, (kl_loss, recon_loss, 0.0f0, 0.0f0)
end
    