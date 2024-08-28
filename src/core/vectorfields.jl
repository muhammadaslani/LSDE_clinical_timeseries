"""
    MLP(Id::Vector{Int} ,Od::Int; hidden_size, depth, activation)

Constructs an MLP decoder for multiple inputs and one output.

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
                                                                        Dense(hidden_size, Od, tanh))) do xs
                                                                
                                                                            @return m(vcat(xs...))
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

    - `LuxCompactLayer`
        
"""
MLP(Id::Int ,Od::Int; hidden_size, depth, activation) = @compact(m=Chain(Dense(Id => hidden_size, activation), 
                                                                        [Dense(hidden_size, hidden_size, activation) for i in 1:depth]..., 
                                                                        Dense(hidden_size, Od, tanh))) do x
                    
                                                                            @return m(x)
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
SparseMLP(Id::Int, Od::Int; activation) = @compact(m=Scale(Id, activation, init_weight=identity_init(gain=1.0f0))) do x
                                                            @return m(x)
                                                         end
                                                                