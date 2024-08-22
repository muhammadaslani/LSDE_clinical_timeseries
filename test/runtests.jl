using Rhythm, Lux, DifferentialEquations, Random, SciMLSensitivity, ComponentArrays
using Test

@testset "Rhythm.jl" begin
    @testset "LatentSDE Forward Pass" begin
    ts = 0:0.1:9.9 |> Array{Float32}
    x = rand32(2, 100, 1)
    u = nothing 

    drift = @compact(vf = Dense(2,2)) do xu 
        x, u = xu 
        @return vf(x)
    end

    dynamics = SDE(drift, Dense(4, 2), Dense(2, 2), EulerHeun(), saveat=ts, dt=0.1)
    model = LatentSDE(dynamics=dynamics)

    rng = Random.default_rng()
    θ, st = Lux.setup(rng, model)
    θ = θ |> ComponentArray
    
    # Test that the forward pass runs without errors
    ŷ, px₀, kl_path = model(x, u, ts, θ, st)
    
    # Add more specific tests here
    @test size(ŷ) == (2, 100, 1)  # Assuming this is the expected output size
    @test !isnothing(px₀)
    @test !isnothing(kl_path)
    end
end
