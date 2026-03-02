function loss_fn(model, θ, st, data; β=1.0f0)
    (x_obs, u_obs, y_obs, y_masks_obs, x_masks_obs, _, _, _, _, _), ts, λ = data
    batch_size = size(y_obs)[end]

    # Augment x_obs with its observation mask so the encoder can distinguish
    # real measurements from mean-imputed values.
    # x_masks_obs covers only the dynamic rows; pad static rows with ones.
    n_static = size(x_obs, 1) - size(x_masks_obs, 1)
    static_ones = ones(Float32, n_static, size(x_obs, 2), batch_size)
    x_mask_full = vcat(static_ones, Float32.(x_masks_obs))   # [obs_dim, T, N]
    x_aug = vcat(x_obs, x_mask_full)                   # [2*obs_dim, T, N]

    # Reconstruct history: encoder sees augmented obs, driven by obs-period controls
    ŷ, px₀, kl_pq = model(x_aug, u_obs, ts, θ, st)

    recon_losses = map(eachindex(ŷ)) do i
        μ, log_σ² = ŷ[i][1], ŷ[i][2]
        normal_loglikelihood(
            μ[1, :, :],
            log_σ²[1, :, :],
            y_obs[i, :, :],
            y_masks_obs[i, :, :]
        ) / batch_size
    end
    recon_loss = sum(recon_losses)

    kl_init = kl_normal(px₀...) / batch_size
    kl_path = kl_pq === nothing ? 0.0f0 : mean(kl_pq[end, :])
    kl_loss = kl_init + β * kl_path

    loss = recon_loss + λ * kl_loss
    return loss, st, (kl_path, kl_init, recon_losses...)
end