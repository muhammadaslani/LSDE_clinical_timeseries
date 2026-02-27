function loss_fn(model, θ, st, data; β=1.0f0)
    (u_obs, x_obs, y_obs, masks_obs, _, _, _, _), ts, λ = data
    batch_size = size(y_obs)[end]

    # Reconstruct history: encoder sees full obs, driven by obs-period inputs
    ŷ, px₀, kl_pq = model(x_obs, u_obs, ts, θ, st)

    recon_losses = map(eachindex(ŷ)) do i
        μ, log_σ² = ŷ[i][1], ŷ[i][2]
        normal_loglikelihood(
            μ[1, :, :] .* masks_obs[i, :, :],
            log_σ²[1, :, :] .* masks_obs[i, :, :],
            y_obs[i, :, :] .* masks_obs[i, :, :]
        ) / batch_size
    end
    recon_loss = sum(recon_losses)

    kl_init = kl_normal(px₀...) / batch_size
    kl_path = kl_pq === nothing ? 0.0f0 : mean(kl_pq[end, :])
    kl_loss = kl_init + β * kl_path

    loss = recon_loss + λ * kl_loss
    return loss, st, (kl_path, kl_init, recon_losses...)
end