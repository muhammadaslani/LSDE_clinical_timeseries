module Rhythm

using  Lux, LinearAlgebra, SciMLSensitivity, Zygote, Distributions, Interpolations, SpecialFunctions, DifferentialEquations, Random, CairoMakie, CUDA, JLD2, FileIO, Printf, 
OptimizationOptimisers, Optimisers, Printf, Colors, ComponentArrays
import ChainRulesCore as CRC
using Parameters: @unpack, @with_kw
import LuxCore: AbstractLuxContainerLayer, AbstractLuxWrapperLayer, AbstractLuxLayer

abstract type LatentVariableModel <: AbstractLuxContainerLayer{(:obs_encoder, :ctrl_encoder, :init_map, :dynamics, :state_map, :obs_decoder, :ctrl_decoder)} end
abstract type UDE <: AbstractLuxContainerLayer{(:vector_field,)} end
abstract type DynamicalSystem <: AbstractLuxLayer end

# Core stuff
include("core/dynamics.jl")
export SDE, ODE, LSTM, sample_generative, sample_augmented
include("core/latentsde.jl")
export LatentSDE, predict, generate, filter, smooth
include("core/latentode.jl")
export LatentODE, predict, generate, filter, smooth
include("core/latent_lstm.jl")
export LatentLSTM, predict
include("core/encoders.jl")
export Encoder, Identity_Encoder, Recurrent_Encoder
include("core/decoders.jl")
export Decoder, Identity_Decoder, Linear_Decoder, MLP_Decoder, MultiHeadMLPDecoder, MultiHeadLinearDecoder
include("core/vectorfields.jl")
export MLP, SparseMLP, SparseMLP_ODE, HopfOscillators, Linear, LimitCycleOscillators, StuartLandauOscillators


const TYPE_MAP = Dict(
    "Identity_Encoder" => Identity_Encoder,
    "Recurrent_Encoder" => Recurrent_Encoder,
    "MLP_Decoder" => MLP_Decoder,
    "MultiHeadMLPDecoder" => MultiHeadMLPDecoder,
    "MultiHeadLinearDecoder" => MultiHeadLinearDecoder,
    "Identity_Decoder" => Identity_Decoder,
    "Linear_Decoder" => Linear_Decoder,
    "MLP" => MLP,
    "SparseMLP" => SparseMLP,
    "HopfOscillators" => HopfOscillators,
    "StuartLandauOscillators" => StuartLandauOscillators,
    "LimitCycleOscillators" => LimitCycleOscillators,
    "Linear" => Linear,
)


const SOLVER_MAP = Dict(
    "Euler" => Euler(),
    "EM" => EM(),
    "EulerHeun" => EulerHeun(),
    "LambaEM" => LambaEM(),
    "SOSRI" => SOSRI(),
)

# Utils
include("utils/misc.jl")
export sample_rp, interpolate!, basic_tgrad, dropmean, dropsd, pad_matrices, irregularize, split_matrix, prediction_entropy, empirical_crps
include("utils/losses.jl")
export kl_normal, poisson_loglikelihood, poisson_nll_lograte, poisson_loglikelihood_multiple_samples, normal_loglikelihood, mse, frange_cycle_linear, bits_per_spike, CrossEntropy_Loss
include("utils/config.jl")
export create_object, create_latentsde, create_latentode, create_latent_lstm
include("trainer.jl")
export train, validate, vizualize
include("utils//theme.jl")
export atom_one_dark, atom_one_dark_palette, get_atom_one_dark_colors, atom_one_dark_theme
include("utils//animations.jl")
export animate_cont, animate_spikes, animate_oscillators, animate_hand 

set_theme!(atom_one_dark_theme)

end


