"""
Neural CDE Encoder-Decoder Architecture

Encoder:
  1. GRU backward over history [covariates | observations | controls] → h
  2. build_control_path over history data → dXdt_enc
  3. NeuralCDE(dXdt_enc, tspan, h, saveat) → z_enc states + z_enc_final

Decoder:
  1. build_control_path over future controls → dXdt_dec
  2. NeuralCDE(dXdt_dec, tspan, z_enc_final, saveat) → z_dec states
  3. output_decoder maps z_dec → predictions ŷ

Data format (from generate_dataloader):
  batch = (U_obs, Covars_obs, X_obs, Y_obs, Masks_obs,
           U_forecast, Covars_forecast, X_forecast, Y_forecast, Masks_forecast)
"""

using Lux, LuxCore, DifferentialEquations, SciMLSensitivity
using ComponentArrays, Random, Zygote
import ChainRulesCore as CRC

# Include the NeuralCDE implementation
include("neural_cde.jl")

# -----------------------------------------------------------------------
# Encoder-Decoder wrapper using NeuralCDE
# -----------------------------------------------------------------------

struct CDEEncoderDecoder <: AbstractLuxContainerLayer{(:obs_encoder, :cde_encoder, :cde_decoder, :output_decoder)}
    obs_encoder      # GRUCell: processes history backward → h
    cde_encoder      # NeuralCDE: init_map(h) → z0, CDE forward → z_enc
    cde_decoder      # NeuralCDE: init_map(z_enc_final) → z0_dec, CDE forward → z_dec
    output_decoder   # Dense: z_dec → ŷ predictions
end

"""
    CDEEncoderDecoder(; obs_dim, covars_dim, control_dim, latent_dim, hidden_size=64, depth=1)

Flow:
  GRU(history backward) → h
  build_control_path(history + time) → dXdt_enc
  encoder CDE(dXdt_enc, h) → z_enc states + z_enc_final
  build_control_path(future controls + time) → dXdt_dec
  decoder CDE(dXdt_dec, z_enc_final) → z_dec states
  output_decoder(z_dec) → ŷ
"""
function CDEEncoderDecoder(; obs_dim::Int, covars_dim::Int, control_dim::Int, latent_dim::Int,
                            hidden_size::Int = 64, depth::Int = 1)
    encoder_path_dim = covars_dim + obs_dim + control_dim

    return CDEEncoderDecoder(
        # GRU: input = encoder_path_dim (without time), hidden = latent_dim
        GRUCell(encoder_path_dim => latent_dim),
        # Encoder CDE: path driven by history + time channel, init from GRU output (latent_dim)
        NeuralCDE(path_dim=encoder_path_dim + 1, latent_dim=latent_dim,
                  hidden_size=hidden_size, depth=depth, init_dim=latent_dim),
        # Decoder CDE: path driven by future controls + time channel, init from encoder final state
        NeuralCDE(path_dim=control_dim + 1, latent_dim=latent_dim,
                  hidden_size=hidden_size, depth=depth, init_dim=latent_dim),
        Chain(Dense(latent_dim => hidden_size, tanh), Dense(hidden_size => obs_dim)),
    )
end

# -----------------------------------------------------------------------
# Forward pass
# -----------------------------------------------------------------------

function (model::CDEEncoderDecoder)(U_obs::AbstractArray{<:Real,3},
                                     Covars_obs::AbstractArray{<:Real,3},
                                     Y_obs::AbstractArray{<:Real,3},
                                     ::AbstractArray,  # Masks_obs
                                     U_forecast::AbstractArray{<:Real,3},
                                     ::AbstractArray{<:Real,3},  # Y_forecast
                                     ::AbstractArray,  # Masks_forecast
                                     t_obs::AbstractVector,
                                     t_future::AbstractVector,
                                     ps::ComponentArray, st::NamedTuple;
                                     dt::Union{Nothing,Real} = nothing)

    obs_dim, T_obs, B = size(Y_obs)

    # =====================================================================
    # ENCODER
    # =====================================================================

    # 1. Concatenate history: [covariates | observations | controls]
    encoder_input = cat(Covars_obs, Y_obs, U_obs; dims=1)  # (encoder_path_dim, T_obs, B)

    # 2. Run GRU backward over history → summary h
    gru_hidden_dim = size(ps.obs_encoder.weight_hh, 2)  # GRU hidden size = latent_dim
    h = zeros(Float32, gru_hidden_dim, B)
    st_gru = st.obs_encoder
    for t in T_obs:-1:1
        x_t = encoder_input[:, t, :]
        (h, _), st_gru = model.obs_encoder((x_t, (h,)), ps.obs_encoder, st_gru)
    end
    # h: (latent_dim, B) — initial condition for encoder CDE

    # 3. Build control path from history data + time channel
    time_channel_obs = reshape(Float32.(t_obs), 1, T_obs, 1) .* ones(Float32, 1, 1, B)  # (1, T_obs, B)
    encoder_input_cde = cat(encoder_input, time_channel_obs; dims=1)  # (encoder_path_dim+1, T_obs, B)
    _, dXdt_enc = CRC.@ignore_derivatives build_control_path(encoder_input_cde, t_obs)

    # 4. Encoder CDE: init_map(h) → z0, integrate forward over history
    z_enc, st_enc = model.cde_encoder(dXdt_enc, (t_obs[1], t_obs[end]),
                                       h, t_obs,
                                       ps.cde_encoder, st.cde_encoder; dt=dt)
    # z_enc: (latent_dim, T_obs, B)

    # Extract final encoder state as initial condition for decoder
    z_enc_final = z_enc[:, end, :]  # (latent_dim, B)

    # =====================================================================
    # DECODER
    # =====================================================================

    # 1. Build control path from future controls + time channel
    T_for = size(U_forecast, 2)
    time_channel = reshape(Float32.(t_future), 1, T_for, 1) .* ones(Float32, 1, 1, B)  # (1, T_for, B)
    decoder_input = cat(U_forecast, time_channel; dims=1)  # (control_dim+1, T_for, B)
    _, dXdt_dec = CRC.@ignore_derivatives build_control_path(decoder_input, t_future)

    # 2. Decoder CDE: init_map(z_enc_final) → z0_dec, integrate forward over future
    z_dec, st_dec = model.cde_decoder(dXdt_dec, (t_future[1], t_future[end]),
                                       z_enc_final, t_future,
                                       ps.cde_decoder, st.cde_decoder; dt=dt)
    # z_dec: (latent_dim, T_forecast, B)

    # 3. Output head: map latent states to predictions
    T_forecast = size(z_dec, 2)
    z_dec_flat = reshape(z_dec, size(z_dec, 1), T_forecast * B)       # (latent_dim, T_forecast*B)
    ŷ_flat, st_out = model.output_decoder(z_dec_flat, ps.output_decoder, st.output_decoder)
    ŷ = reshape(ŷ_flat, size(ŷ_flat, 1), T_forecast, B)              # (obs_dim, T_forecast, B)

    st_new = (
        obs_encoder = st_gru,
        cde_encoder = st_enc,
        cde_decoder = st_dec,
        output_decoder = st_out,
    )

    return ŷ, st_new
end
