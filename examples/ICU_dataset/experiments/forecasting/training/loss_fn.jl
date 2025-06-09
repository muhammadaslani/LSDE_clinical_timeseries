function loss_fn_nde(model, θ, st, data)
    (u_obs, x_obs, y_obs, masks_obs, u_for, x_for, y_for, masks_for), ts, λ = data
    batch_size= size(y_for)[end]
    ŷ, px₀, kl_pq = model(x_obs, hcat(u_obs, u_for), ts, θ, st)
    recon_loss = 0.0f0
    for i in eachindex(ŷ)
        μ, log_σ² = ŷ[i][1], ŷ[i][2]
        valid_indx= findall(masks_for[i, :, :] .== 1)
        recon_loss += normal_loglikelihood(μ[1,valid_indx], log_σ²[1,valid_indx], y_for[i, valid_indx])/batch_size
    end 
    kl_loss = kl_normal(px₀...) / batch_size + mean(kl_pq[end, :]) 
    loss = recon_loss + λ * kl_loss
    return loss, st, (kl_loss, recon_loss, 0.0f0, 0.0f0)
end



# RNN-specific functions
function loss_fn_rnn(model, θ, st, data)
    (u_obs, x_obs, y_obs, masks_obs, u_for, x_for, y_for, masks_for), ts, λ = data
    batch_size = size(y_for)[end]
    history_data= vcat(x_obs, u_obs)
    forecast_length = size(y_for)[2]
    ŷ, st = model(history_data, u_for, forecast_length, θ, st)
    recon_loss = 0.0f0
    for i in eachindex(ŷ)
        μ, log_σ² = ŷ[i][1], ŷ[i][2]
        valid_indx = findall(masks_for[i, :, :] .== 1)
        recon_loss += normal_loglikelihood(μ[1, valid_indx], log_σ²[1, valid_indx], y_for[i, valid_indx]) / batch_size
    end
    kl = 0.0f0
    return recon_loss, st, (kl, recon_loss, 0.0f0, 0.0f0)
end