module Rhythm

using Lux, ComponentArrays, LinearAlgebra, SciMLSensitivity, Zygote, Distributions, Interpolations, SpecialFunctions, DifferentialEquations, Random, CairoMakie, CUDA, JLD2, FileIO, Printf, WandbMacros,
OptimizationOptimisers, Optimisers, Printf
import ChainRulesCore as CRC
using Parameters: @unpack, @with_kw
import LuxCore: AbstractLuxContainerLayer, AbstractLuxWrapperLayer, AbstractLuxLayer

abstract type LatentVariableModel <: AbstractLuxContainerLayer{(:obs_encoder, :ctrl_encoder, :init_map, :dynamics, :state_map, :obs_decoder, :ctrl_decoder)} end

# Core stuff
include("core/dynamics.jl")
export SDE, sample_generative, sample_augmented
include("core/latentsde.jl")
export LatentSDE, predict, generate, filter, smooth
include("core/encoders.jl")
export Encoder, Identity_Encoder, Recurrent_Encoder
include("core/decoders.jl")
export Decoder, Identity_Decoder, Linear_Decoder, MLP_Decoder, MultiDecoder, MultiDecoder_linear
include("core/vectorfields.jl")
export MLP, SparseMLP, HopfOscillators, Linear, LimitCycleOscillators, StuartLandauOscillators


const TYPE_MAP = Dict(
    "Identity_Encoder" => Identity_Encoder,
    "Recurrent_Encoder" => Recurrent_Encoder,
    "MLP_Decoder" => MLP_Decoder,
    "MultiDecoder" => MultiDecoder,
    "MultiDecoder_linear" => MultiDecoder_linear,
    "Identity_Decoder" => Identity_Decoder,
    "Linear_Decoder" => Linear_Decoder,
    "MLP" => MLP,
    "SparseMLP" => SparseMLP,
    "HopfOscillators" => HopfOscillators,
    "StuartLandauOscillators" => StuartLandauOscillators,
    "LimitCycleOscillators" => LimitCycleOscillators,
    "Linear" => Linear
)


const SOLVER_MAP = Dict(
    "EM" => EM(),
    "EulerHeun" => EulerHeun(),
    "LambaEM" => LambaEM(),
    "SOSRI" => SOSRI(),
)

# Utils
include("utils/misc.jl")
export sample_rp, interpolate!, basic_tgrad, dropmean, dropsd, pad_matrices, animate_oscillators
include("utils/losses.jl")
export kl_normal, poisson_loglikelihood, normal_loglikelihood, mse, frange_cycle_linear, bits_per_spike
include("utils/config.jl")
export create_object, create_latentsde
include("trainer.jl")
export train, validate, vizualize

end


