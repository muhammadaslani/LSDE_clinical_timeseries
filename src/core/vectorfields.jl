"""
    MLP(Id::Vector{Int} ,Od::Int; hidden_size, depth, activation)

Constructs an MLP vector field for multiple inputs and one output.

Arguments:

- `Id`: Vector of dimensions of the inputs.
- `Od`: Dimension of the output.
- `hidden_size`: Dimension of the hidden layers.
- `depth`: Number of hidden layers.
- `activation`: Activation function.

returns: 

    - `LuxCompactLayer`
        
"""
MLP(Id::Vector{Int} ,Od::Int; hidden_size, depth, activation) = @compact(m=Chain(Dense(sum(Id) => hidden_size, activation), 
                                                                        [Dense(hidden_size, hidden_size, activation) for i in 1:depth]..., 
                                                                        Dense(hidden_size, Od, activation))) do xs
                                                                            @return m(vcat(xs...))
                                                                             end


"""
    Linear(Id::Vector{Int}, Od::Int)

Constructs a linear mapping. 

Arguments:

- `Id`: Vector of dimensions of the inputs.
- `Od`: Dimension of the output.

returns: 

    - `LuxCompactLayer`
    
"""
Linear(Id::Int, Od::Int) = @compact(m=Dense(Id, Od)) do x
    @return m(x)
end


"""

    MLP(Id::Int ,Od::Int; hidden_size, depth, activation)

Constructs an MLP decoder for one input and one output.

Arguments:

- `Id`: Dimension of the input.
- `Od`: Dimension of the output.
- `hidden_size`: Dimension of the hidden layers.
- `depth`: Number of hidden layers.
- `activation`: Activation function.

returns: 

    - `Chain`
        
"""
function MLP(Id::Int, Od::Int; hidden_size, depth, activation)
    layers = Any[Dense(Id => hidden_size, activation)]
    for i in 1:depth
        push!(layers, Dense(hidden_size => hidden_size, activation))
    end
    push!(layers, Dense(hidden_size => Od, identity))
    return Chain(layers...)
end


"""
SparseMLP(Id::Int, Od::Int; activation)

constructs a Sparsely Connected layer where only the diagonal elements are non-zero.

Arguments:

- `Id`: Dimension of the input.
- `Od`: Dimension of the output.
- `activation`: Activation function.

returns: 

    - `LuxCompactLayer`
        
"""
SparseMLP(Id::Int, Od::Int; activation) = @compact(m=Scale(Id, activation, init_weight=identity_init(gain=0.1f0))) do x
                                                            @return m(x)
                                                         end

                                       
"""
    HopfOscillators(N::Int)

Constructs a systems of N coupled Hopf Oscillators.

Arguments:

- `N`: Number of Oscillators.

returns: 

    - `LuxCompactLayer`

"""
HopfOscillators(N::Int) = @compact(σ=truncated_normal(mean=0, std=3, lo=-3, hi=5)(N),
               ω=ones32(N),
               K=zeros32(N, N),
               name="HopfOscillators (N=$N)") do xu
       
            z, u = xu
            x_ = @view z[1:N,:]
            y_ = @view z[N+1:2N, :]
                    # Compute the coupling terms
            
            K = softmax(K, dims=2)
            
            coupling_x = K * x_ - sum(K, dims=2) .* x_
            coupling_y = K * y_ - sum(K, dims=2) .* y_

            dx = (-ω.*y_ + x_.*(σ .+ 2(x_.^2 + y_.^2) - (x_.^2 + y_.^2).^2)) + coupling_x .+ 0.001f0
            dy = (ω.*x_ + y_.*(σ .+ 2(x_.^2 + y_.^2) - (x_.^2 + y_.^2).^2))  + coupling_y .+ 0.001f0

            @return vcat(dx, dy)
        end


HopfOscillators(N::Int, M::Int) = @compact(
    σ=truncated_normal(mean=0, std=3, lo=-3, hi=5)(N),
    ω=ones32(N),
    K=Chain(Dense(M, N*N)),  # Coupling strength K
    name="Controlled HopfOscillators (N=$N)") do xu

    z, u = xu
    x_ = @view z[1:N,:]
    y_ = @view z[N+1:2N, :]
            # Compute the coupling terms
    bs = size(u, 2)
    K = softmax(reshape(K(u), N, N, bs), dims=2)
    coupling_x = dropdims(sum(K .* reshape(x_, (1, N, bs)), dims=2), dims=2)
    coupling_y = dropdims(sum(K .* reshape(y_, (1, N, bs)), dims=2), dims=2)
    dx = (-ω.*y_ + x_.*(σ .+ 2(x_.^2 + y_.^2) - (x_.^2 + y_.^2).^2)) .+ coupling_x .+ 0.001f0
    dy = (ω.*x_ + y_.*(σ .+ 2(x_.^2 + y_.^2) - (x_.^2 + y_.^2).^2))  .+ coupling_y .+ 0.001f0

    @return vcat(dx, dy)
end



function StuartLandauOscillators(N::Int)
    @compact(a=truncated_normal(mean=1, std=5, lo=-0.1, hi=5)(N),               # 'a' corresponds to the growth rate, set to 1 in this case
                ω=ones32(N),               # 'ω' represents the natural frequency for each oscillator
                K=rand32(1),               # Coupling strength K
                name="StuartLandauOscillators (N=$N)") do xu
        
        z, u = xu
        x_ = @view z[1:N, :]  # Real part of z
        y_ = @view z[N+1:2N, :]  # Imaginary part of z

        # Compute the squared amplitude
        r_squared = x_.^2 + y_.^2

        # Compute the complex term (1 - |z_j|^2 + iω_j) * z_j
        real_part = (1 .- r_squared) .* x_ - ω .* y_
        imag_part = (1 .- r_squared) .* y_ + ω .* x_

        # Compute the all-to-all coupling term
        avg_x = sum(x_, dims=1) / N
        avg_y = sum(y_, dims=1) / N

        coupling_x = (K / N) .* (avg_x .- x_)
        coupling_y = (K / N) .* (avg_y .- y_)

        # Combine terms for dx and dy
        dx = real_part .+ coupling_x
        dy = imag_part .+ coupling_y

        # Return concatenated real and imaginary parts
        @return vcat(dx, dy)
    end
end

function LimitCycleOscillators(N::Int)
    @compact(ω=ones32(N),  # Natural frequency for each oscillator
             K=rand32(1),  # Coupling strength K
             name="LimitCycleOscillators (N=$N)") do xu
        
        z, u = xu
        x_ = @view z[1:N, :]       # Real part of z (first N components)
        y_ = @view z[N+1:2N, :]    # Imaginary part of z (next N components)

        # Calculate |z|^2 for all oscillators and batches
        z_squared = x_.^2 + y_.^2
        
        # Calculate coupling terms
        coupling_x = K/N .* (sum(x_, dims=1) .- N*x_)
        coupling_y = K/N .* (sum(y_, dims=1) .- N*y_)
        
        # Update derivatives
         dx_ = (1 .- z_squared) .* x_ .- ω .* y_ .+ coupling_x
         dy_ = (1 .- z_squared) .* y_ .+ ω .* x_ .+ coupling_y

        @return vcat(dx_, dy_)
    end
end


function LimitCycleOscillators(N::Int, M::Int)
    @compact(ω=ones32(N),  # Natural frequency for each oscillator
             K=Chain(Dense(M, 32, softplus), Dense(32, 1, tanh)),  # Coupling strength K
             name="LimitCycleOscillators (N=$N)") do xu
        
        z, u = xu
        x_ = @view z[1:N, :]       # Real part of z (first N components)
        y_ = @view z[N+1:2N, :]    # Imaginary part of z (next N components)

        # Calculate |z|^2 for all oscillators and batches
        z_squared = x_.^2 + y_.^2
        
        # Calculate coupling terms
        coupling_x = K(u)/N .* (sum(x_, dims=1) .- N*x_)
        coupling_y = K(u)/N .* (sum(y_, dims=1) .- N*y_)
        
        # Update derivatives
         dx_ = (1 .- z_squared) .* x_ .- ω .* y_ .+ coupling_x
         dy_ = (1 .- z_squared) .* y_ .+ ω .* x_ .+ coupling_y

        @return vcat(dx_, dy_)
    end
end