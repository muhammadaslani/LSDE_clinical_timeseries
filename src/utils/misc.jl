"""
    sample_rp(x::Tuple)

Samples from a MultiVariate Normal distribution using the reparameterization trick.

Arguments:

  - `x`: Tuple of the mean and squared variance of a MultiVariate Normal distribution.

returns: 

    - The sampled value.
"""
function sample_rp(x::Tuple{AbstractArray, AbstractArray})
    return x[1] + rand!(x[1]) .* sqrt.(x[2])
end


sample_rp(x::AbstractArray) = x
sample_rp(x::AbstractFloat) = x



"""
    interp!(ts, cs, time_point)

Interpolates the control signal at a given time point.

Arguments:

  - `ts`: Array of time points.
  - `x`: Arrray to interpolate.
  - `time_point`: The time point at which to interpolate

returns: 

    - The interpolated control signal.

"""

function interp!(ts, x::AbstractMatrix, t)
   return CRC.@ignore_derivatives[linear_interpolation(ts, view(x, i, :), extrapolation_bc=Line())(t) for i in axes(x, 1)]
end


function interp!(ts, x::AbstractArray, t)
    CRC.@ignore_derivatives begin
        # Determine the actual observation times for x
        obs_times = ts[1:size(x, 2)]
        
        # Create interpolation for each feature and batch
        return [
            let interp_obj = linear_interpolation(obs_times, view(x, i, :, b), extrapolation_bc=Line())
                interp_obj(t)
            end
            for i in axes(x, 1), b in axes(x, 3)
        ]
    end
end

function interp!(ts, x::Nothing, t)
    return nothing
end

dropmean(A; dims=:) = dropdims(mean(A; dims=dims); dims=dims)
dropsd(A; dims=:) = dropdims(std(A; dims=dims); dims=dims)


basic_tgrad(u, p, t) = zero(u)



function pad_matrices(Y, T; return_timepoints = true, pad_method = :zero)
    T_max = maximum(size(y, 2) for y in Y)
    
    function pad_matrix(matrix)
        pad_size = T_max - size(matrix, 2)
        if pad_size == 0
            return matrix
        end
        
        if pad_method == :last
            return hcat(matrix, repeat(matrix[:, end], 1, pad_size))
        elseif pad_method == :mean
            return hcat(matrix, repeat(mean(matrix, dims=2), 1, pad_size))
        elseif pad_method == :linear_interpolation
            start_vals = matrix[:, end]
            end_vals = matrix[:, end] + (matrix[:, end] - matrix[:, end-1])
            interpolated = [start_vals + (end_vals - start_vals) * (i / (pad_size + 1)) for i in 1:pad_size]
            return hcat(matrix, reduce(hcat, interpolated))
        else  # Default to zero padding
            return hcat(matrix, zeros(eltype(matrix), size(matrix, 1), pad_size))
        end
    end
    
    Y_padded = [pad_matrix(matrix) for matrix in Y]
    masks = [hcat(fill(true, size(matrix, 1), size(matrix, 2)), fill(false, size(matrix, 1), T_max - size(matrix, 2))) for matrix in Y]
    Y_padded = cat(Y_padded..., dims=3)
    masks = cat(masks..., dims=3)
    @info "Padded matrices using method: $pad_method"
    
    if return_timepoints
        timepoints = T[argmax(length.(T))]
        return Y_padded, masks, timepoints
    else
        return Y_padded, masks
    end
end

# Custom vcat function for handling `nothing` values
function Base.vcat(a::AbstractArray, b::Nothing, c::AbstractArray)
    return vcat(a, c)
end

function Base.vcat(a::Nothing, b::AbstractArray, c::AbstractArray)
    return vcat(b, c)
end

function Base.vcat(a::AbstractArray, b::AbstractArray, c::Nothing)
    return vcat(a, b)
end

function Base.vcat(a::AbstractArray, b::Nothing)
    return a
end

function Base.vcat(a::Nothing, b::AbstractArray)
    return b
end