#######################
    # Data Normalization Functions
#######################

function z_normalize(z::AbstractArray{T,3}; dim::Int=1, eps::T=T(1e-8)) where T
   
    # Calculate mean and std for each slice along slice_dim
    μ = [mean(selectdim(z, dim, i)) for i in axes(z, dim)]
    σ = [std(selectdim(z, dim, i)) for i in axes(z, dim)]

    return z_normalize(z, μ, σ; dim=dim, eps=eps)
end

function z_normalize(z::AbstractArray{T,3}, μ::AbstractVector, σ::AbstractVector; dim::Int=1, eps::T=T(1e-8)) where T
    z_norm = similar(z)
    
    for i in axes(z, dim)
        slice_view = selectdim(z_norm, dim, i)
        slice_view .= (selectdim(z, dim, i) .- μ[i]) ./ (σ[i] + eps)
    end
    
    return z_norm, μ, σ
end

function z_denormalize(z_norm::AbstractArray{T,3}, μ::AbstractVector, σ::AbstractVector; dim::Int=1, eps::T=T(1e-8)) where T
    z = similar(z_norm)
    
    for i in axes(z_norm, dim)
        slice_view = selectdim(z, dim, i)
        slice_view .= selectdim(z_norm, dim, i) .* (σ[i] + eps) .+ μ[i]
    end
    
    return z
end




function min_max_normalize(z::AbstractArray{T,3}; dim::Int=1, eps::T=T(1e-8)) where T
    # Calculate min and max for each slice along dim
    min_vals = [minimum(selectdim(z, dim, i)) for i in axes(z, dim)]
    max_vals = [maximum(selectdim(z, dim, i)) for i in axes(z, dim)]

    return min_max_normalize(z, min_vals, max_vals; dim=dim, eps=eps)
end

function min_max_normalize(z::AbstractArray{T,3}, min_vals::AbstractVector, max_vals::AbstractVector; dim::Int=1, eps::T=T(1e-8)) where T
    z_norm = similar(z)

    for i in axes(z, dim)
        slice_view = selectdim(z_norm, dim, i)
        slice_view .= (selectdim(z, dim, i) .- min_vals[i]) ./ (max_vals[i] - min_vals[i] + eps)
    end
    
    return z_norm, min_vals, max_vals
end

function min_max_denormalize(z_norm::AbstractArray{T,3}, min_vals::AbstractVector, max_vals::AbstractVector; dim::Int=1, eps::T=T(1e-8)) where T
    z = similar(z_norm)

    for i in axes(z_norm, dim)
        slice_view = selectdim(z, dim, i)
        slice_view .= selectdim(z_norm, dim, i) .* (max_vals[i] - min_vals[i] ) .+ min_vals[i]
    end
    
    return z
end


"""
    split_matrix(X::Array{T,3}, obs_fraction::Float64 = 0.5) where T

Split a 3D array along the second dimension (time dimension) according to the specified observation fraction.

# Arguments
- `X::Array{T,3}`: 3D array with dimensions (n_features, n_timepoints, n_samples).
- `obs_fraction::Float64=0.5`: Fraction of timepoints to include in the observed part (default: 0.5).

# Returns
- `observed::Array{T,3}`: First part of the split array containing the observed timepoints.
- `forecast::Array{T,3}`: Second part of the split array containing the timepoints to forecast.

# Example
"""
function split_matrix(X::Array{T,3}, obs_fraction::Float64 = 0.5) where T
    n_features, n_timepoints, n_samples = size(X)
    mid = round(Int, n_timepoints * obs_fraction)  # Compute split index

    observed = X[:, 1:mid, :]
    forecast = X[:, mid+1:end, :]

    return observed, forecast
end

function split_matrix(X::Array{T,2}, obs_fraction::Float64 = 0.5) where T
    n_features, n_timepoints = size(X)
    mid = round(Int, n_timepoints * obs_fraction)  # Compute split index

    observed = X[:, 1:mid]
    forecast = X[:, mid+1:end]

    return observed, forecast
end

function split_matrix(X::Vector{T}, obs_fraction::Float64 = 0.5) where T
    n_timepoints = length(X)
    mid = round(Int, n_timepoints * obs_fraction)  # Compute split index

    observed = X[1:mid]
    forecast = X[mid+1:end]

    return observed, forecast
end

"""
    irregularize(y1::AbstractMatrix, y2::AbstractMatrix, mask1::AbstractMatrix, mask2::AbstractMatrix)

Randomly sets 20% of the values in the input matrices `y1` and `y2` to zero and updates the corresponding positions in the mask matrices `mask1` and `mask2` to indicate the irregularized entries.

# Arguments
- `y1::AbstractMatrix`: The first input matrix.
- `y2::AbstractMatrix`: The second input matrix.
- `mask1::AbstractMatrix`: The mask matrix corresponding to `y1`.
- `mask2::AbstractMatrix`: The mask matrix corresponding to `y2`.

# Returns
- `y1::AbstractMatrix`: The irregularized first input matrix.
- `y2::AbstractMatrix`: The irregularized second input matrix.
- `mask1::AbstractMatrix`: The updated mask matrix for `y1`.
- `mask2::AbstractMatrix`: The updated mask matrix for `y2`.

"""


function irregularize(y1, y2, mask1, mask2)
    mask1=copy(mask1)
    mask2=copy(mask2)
    for i in 1:size(y1)[3]
        for j in 1:size(y1)[2]
            samp=rand(0:0.1:1)<0.8
            if samp==0
                y1[:,j,i].=0
                y2[:,j,i].=0
                mask1[:,j,i].=false
                mask2[:,j,i].=false
            end

        end 
    end 
    return y1, y2, mask1, mask2
end 


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

function interp!(ts, x::AbstractMatrix, t, ::Val{:linear})
   return CRC.@ignore_derivatives[linear_interpolation(ts, view(x, i, :), extrapolation_bc=Line())(t) for i in axes(x, 1)]
end


function interp!(ts, x::AbstractArray, t, ::Val{:linear})
    CRC.@ignore_derivatives begin
        # Determine the actual observation times for x
        n_obs = min(length(ts), size(x, 2))
        obs_times = ts[1:n_obs]
        
        # If x has more time points than ts, we'll extend the time points
        if size(x, 2) > length(ts)
            if length(ts) > 1
                # Calculate the time step based on the last two points in ts
                time_step = ts[end] - ts[end-1]
            else
                # If ts has only one point, assume a unit time step
                time_step = 1
            end
            
            # Extend obs_times with consistent time scaling
            extra_times = [ts[end] + i * time_step for i in 1:(size(x, 2) - length(ts))]
            obs_times = vcat(obs_times, extra_times)
        end
        
        # Create interpolation for each feature and batch
        return [
            let interp_obj = linear_interpolation(obs_times, view(x, i, 1:length(obs_times), b), extrapolation_bc=Line())
                interp_obj(t)
            end
            for i in axes(x, 1), b in axes(x, 3)
        ]
    end
end

function interp!(ts, x::AbstractVector, t, ::Val{:binary})
    CRC.@ignore_derivatives begin
        tolerance = 1e-6
        y = zero(eltype(x))  # Use eltype(x) instead of eltype(x[1])
        # Find if t matches any timepoint in obs_times within tolerance
        for (i, time_point) in enumerate(ts)
            if abs(t - time_point) <= tolerance
                y = x[i]  # Return the value at the matching time point
                break  # Add break to exit once found
            end 
        end
        return y
    end 
end

function interp!(ts, x::AbstractArray, t, ::Val{:binary})
    CRC.@ignore_derivatives begin
        # Ensure we return a proper array of numbers, not functions
        result = Float64[]  # Initialize with proper type
        for i in axes(x, 1)
            for b in axes(x, 3)
                val = interp!(ts, view(x, i, 1:length(ts), b), t, Val(:binary))
                push!(result, val)
            end
        end
        # Reshape to match expected dimensions
        return reshape(result, size(x, 1), size(x, 3))
    end 
end


function interp!(ts, x::AbstractMatrix, t, ::Val{:BSpline})
    CRC.@ignore_derivatives begin
        n_obs      = min(length(ts), size(x,2))       # how many time samples
        obs_times  = ts[1:n_obs]                      # actual time stamps

        [ begin
              # build index-space spline
            #   itp  = interpolate(view(x, i, 1:n_obs), BSpline(Cubic(Line(OnGrid()))))
            itp  = interpolate(view(x, i, 1:n_obs), BSpline(Constant()))
              # map it to real time
              sitp = Interpolations.scale(itp, range(obs_times[1],obs_times[end], length(obs_times)))
              # evaluate, falling back to *linear extrapolation* outside the range
              extrapolate(sitp, Line())(t)
          end
          for i in axes(x,1)
        ]
    end
end



function interp!(ts, x::AbstractArray, t, ::Val{:BSpline})
    CRC.@ignore_derivatives begin
        # ───── determine real sample times ────────────────────────────────────
        n_obs     = min(length(ts), size(x,2))
        obs_times = ts[1:n_obs]

        if size(x,2) > length(ts)          # extend when x has extra columns
            Δt = length(ts) > 1 ? ts[end] - ts[end-1] : 1
            extra = ts[end] .+ Δt .* (1:(size(x,2)-length(ts)))
            obs_times = vcat(obs_times, extra)
        end

        # ───── interpolate each (feature, batch) slice ‐ safely ───────────────
        [ begin
            #   itp  = interpolate(view(x, i, 1:length(obs_times), b),
            #                      BSpline(Cubic(Line(OnGrid()))))
            itp  = interpolate(view(x, i, 1:length(obs_times), b),
                                 BSpline(Constant()))
              sitp = Interpolations.scale(itp, range(obs_times[1],obs_times[end], length(obs_times)))
              extrapolate(sitp, Line())(t)
          end
          for i in axes(x,1), b in axes(x,3)
        ]
    end
end


function interp!(ts, x::Nothing, t, ::Val)
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


function stack_seqs(x)
    return stack(x; dims=2)
end



function animate_oscillators(z)
    N = Int(size(z, 1) / 2)
    x = z[1:N, :]
    y = z[N+1:end, :]
    
    fig = Figure(size = (800, 600))
    ax = CairoMakie.Axis(fig[1, 1], 
              xlabel = "Re(z)", 
              ylabel = "Im(z)", 
              title = "Oscillator Animation")
    
    limits!(ax, -10, 10, -10, 10)

    lines_obs = [Observable(Point2f[]) for _ in 1:N]
    scatter_obs = Observable(Point2f[])

    for i in 1:N
        lines!(ax, lines_obs[i], color = :blue, linewidth = 2.5, alpha = 0.8)
    end
    scatter!(ax, scatter_obs, color = :red, markersize = 15)

    CairoMakie.record(fig, "oscillators_animation.mp4", 1:size(x, 2); framerate = 20) do frame
        for i in 1:N
            lines_obs[i][] = Point2f[(x[i, t], y[i, t]) for t in 1:frame]
        end
        scatter_obs[] = Point2f[(x[i, frame], y[i, frame]) for i in 1:N]
        ax.title = "Frame $frame"
    end
end


function animate_oscillators(z, latent_dims)
    N = Int(size(z, 1) / 2)
    x = z[1:N, :]
    y = z[N+1:end, :]
    
    fig = Figure(size = (800, 600))
    ax = CairoMakie.Axis(fig[1, 1], 
              xlabel = "Re(z)", 
              ylabel = "Im(z)", 
              title = "Oscillator Animation")
    
    limits!(ax, -10, 10, -10, 10)

    lines_obs = [Observable(Point2f[]) for _ in 1:N]
    scatter_obs = Observable(Point2f[])

    for i in 1:N
        lines!(ax, lines_obs[i], color = :blue, linewidth = 1.5, alpha = 0.8)
    end
    scatter!(ax, scatter_obs, color = :red, markersize = 10)

    CairoMakie.record(fig, "oscillators_animation.mp4", 1:size(x, 2); framerate = 30) do frame
        for i in 1:N
            lines_obs[i][] = Point2f[(x[i, t], y[i, t]) for t in 1:frame]
        end
        scatter_obs[] = Point2f[(x[i, frame], y[i, frame]) for i in 1:N]
        ax.title = "Frame $frame"
    end
end




function animate_oscillators(z, dims, group_names, ts)
    N = Int(size(z, 1) / 2)
    x = z[1:N, :]
    y = z[N+1:end, :]
    
    # Ensure dims sums up to N and matches the number of group names
    @assert sum(dims) == N "The sum of dims must equal N"
    @assert length(dims) == length(group_names) "The number of dims must match the number of group names"
    
    fig = Figure(size = (1000, 600))
    ax = CairoMakie.Axis(fig[1, 1], 
              xlabel = "Re(z)", 
              ylabel = "Im(z)", 
              title = "Oscillator Animation")
    
    limits!(ax, -5, 5, -5, 5)

    # Generate a color palette for the groups
    colors = Makie.wong_colors()

    lines_obs = []
    scatter_obs = []
    start_idx = 1

    # Create legend elements
    legend_elements = []

    for (group, (dim, name)) in enumerate(zip(dims, group_names))
        group_lines_obs = [Observable(Point2f[]) for _ in 1:dim]
        group_scatter_obs = Observable(Point2f[])
        
        for i in 1:dim
            lines!(ax, group_lines_obs[i], color = colors[group], linewidth = 2.5, alpha = 0.8)
        end
        scatter_plot = scatter!(ax, group_scatter_obs, color = colors[group], markersize = 15)
        
        push!(lines_obs, group_lines_obs)
        push!(scatter_obs, group_scatter_obs)
        
        # Add to legend elements
        push!(legend_elements, MarkerElement(color = colors[group], marker = :circle, markersize = 15))
        
        start_idx += dim
    end

    # Add legend to the figure
    # Create legend
    # Create a compact vertical legend on the right
    Legend(fig[1, 2],
           legend_elements,
           group_names,
           "Regions",
           orientation = :vertical,
           nbanks = 1,
           tellheight = false,
           tellwidth = true,
           margin = (10, 10, 10, 10),
           halign = :left,
           valign = :top)

    # Adjust the layout to give more space to the plot and less to the legend
    colsize!(fig.layout, 1, Relative(0.85))
    colsize!(fig.layout, 2, Relative(0.15))


    CairoMakie.record(fig, "oscillators_animation.mp4", 1:size(x, 2); framerate = 20) do frame
        start_idx = 1
        for (group, dim) in enumerate(dims)
            end_idx = start_idx + dim - 1
            for i in 1:dim
                lines_obs[group][i][] = Point2f[(x[start_idx+i-1, t], y[start_idx+i-1, t]) for t in 1:frame]
            end
            scatter_obs[group][] = Point2f[(x[i, frame], y[i, frame]) for i in start_idx:end_idx]
            start_idx = end_idx + 1
        end
        t = round(ts[frame], digits=2)
        ax.title = "t = $t s"
    end
end



function create_hand_tracking_animation(b, ts, b̂_m, b̂_sd; output_file="hand_tracking_animation.mp4", framerate=20)
    # Function to create confidence ellipse
    function confidence_ellipse(x, y, sd_x, sd_y, n_std=2.0)
        theta = range(0, 2π, length=100)
        ellipse_x = @. x + n_std * sd_x * cos(theta)
        ellipse_y = @. y + n_std * sd_y * sin(theta)
        return Point2f.(ellipse_x, ellipse_y)
    end

    # Ensure all input arrays have the same number of time points
    T = min(size(b, 2), length(ts), size(b̂_m, 2), size(b̂_sd, 2))

    # Set up the figure
    fig = Figure(size=(1000, 600))
    ax = CairoMakie.Axis(fig[1, 1], 
              xlabel = "X position", 
              ylabel = "Y position", 
              title = "Hand Tracking Animation",
              aspect = DataAspect())

    limits!(ax, -10, 10, -10, 10)

    # Create observables
    current_gt_obs = Observable(Point2f(b[1, 1], b[2, 1]))
    gt_trail_obs = Observable(Point2f[])

    current_prediction_obs = Observable(Point2f(b̂_m[1, 1], b̂_m[2, 1]))
    prediction_trail_obs = Observable(Point2f[])

    ellipse_obs = Observable(Point2f[]) 

    # Plot observables
    lines!(ax, gt_trail_obs, color = (:blue, 0.8), linewidth = 3)
    scatter!(ax, current_gt_obs, color = :blue, markersize = 15, label = "Ground Truth")

    lines!(ax, prediction_trail_obs, color = (:red, 0.8), linewidth = 3)
    poly!(ax, ellipse_obs, color = (:red, 0.3), strokewidth = 0)
    scatter!(ax, current_prediction_obs, color = :red, markersize = 15, label = "Prediction")

    # Add legend
    axislegend(ax)

    # Record the animation
    record(fig, output_file, 1:T; framerate = framerate) do frame
        # Update ground truth trail
        gt_trail_obs[] = [Point2f(b[1, t], b[2, t]) for t in 1:frame]
        
        # Update prediction trail
        prediction_trail_obs[] = [Point2f(b̂_m[1, t], b̂_m[2, t]) for t in 1:frame]
        
        # Update current prediction and ground truth
        current_prediction_obs[] = Point2f(b̂_m[1, frame], b̂_m[2, frame])
        current_gt_obs[] = Point2f(b[1, frame], b[2, frame])
        
        # Update confidence ellipse
        ellipse_obs[] = confidence_ellipse(b̂_m[1, frame], b̂_m[2, frame], b̂_sd[1, frame], b̂_sd[2, frame])

        
        # Update title with current time
        ax.title = "t = $(round(ts[frame], digits=2)) s"
    end

    println("Animation saved as '$output_file'")
end











############################################
# Models evaluatiion metrics 
############################################
"""
    npe(y_pred::AbstractArray, mask::AbstractArray{Bool})

Calculate negative predictive entropy for probabilistic predictions.

# Arguments
- `y_pred::AbstractArray`: Predicted probability distributions or logits
- `mask::AbstractArray{Bool}`: Boolean mask indicating valid entries

# Returns
- Mean negative predictive entropy over all valid points

# Description
Computes the negative Shannon entropy of predictions, which measures the confidence
in the model's predictions. Higher NPE values indicate higher confidence (lower uncertainty).
This is the negative of prediction_entropy, making it a reward rather than a penalty.
"""
function npe(y_pred::AbstractArray{T,4}, 
             mask::AbstractArray{Bool,3}) where T <: Number

    n_features, n_timepoints, n_samples = size(mask)
    total_npe = zero(T)
    count = 0

    for f in 1:n_features
        for i in 1:n_samples
            for t in 1:n_timepoints
                if mask[f, t, i]
                    # Get prediction distribution for this point
                    pred_dist = y_pred[f, t, i, :]
                    
                    # Calculate negative entropy
                    pred_softmax = softmax(pred_dist)
                    entropy_val = -sum(pred_softmax .* log.(pred_softmax .+ 1e-10))
                    npe_val = -entropy_val  # Negative entropy
                    
                    total_npe += npe_val
                    count += 1
                end
            end
        end
    end

    return count > 0 ? total_npe / count : zero(T)
end


"""
    npe(y_pred::AbstractArray{T,3}) where T <: Number
Calculate negative predictive entropy for probabilistic predictions without a mask.
# Arguments
- `y_pred::AbstractArray{T,3}`: Predicted probability distributions or logits, shape (n_features, n_timepoints, n_samples).
# Returns
- Mean negative predictive entropy over all points.
    """
function npe(y_pred::AbstractArray{T,3}) where T <: Number
    total_npe = zero(T)
    count = 0
    
    for i in eachindex(size(y_pred, 3))
        for j in eachindex(size(y_pred, 2))
            pred_dist = y_pred[:, j, i]
            pred_softmax = softmax(pred_dist)
            entropy_val = -sum(pred_softmax .* log.(pred_softmax .+ 1e-10))
            npe_val = -entropy_val
            total_npe += npe_val
            count += 1
        end
    end
    
    return count > 0 ? total_npe / count : zero(T)
end


"""
    npe_per_timepoint(y_pred::AbstractArray{T,4}, mask::AbstractArray{Bool,3}) where T <: Number

Calculate negative predictive entropy for each time point and each sample over MC samples.

# Arguments
- `y_pred::AbstractArray{T,4}`: Predicted probability distributions or logits, 
  shape (n_features, n_timepoints, n_samples, n_mc_samples)
- `mask::AbstractArray{Bool,3}`: Boolean mask indicating valid entries,
  shape (n_features, n_timepoints, n_samples)

# Returns
- Array of shape (n_timepoints, n_samples) containing NPE for each time point and sample

# Description
Computes the negative Shannon entropy of predictions for each time point and sample, 
averaged over features at each position. The entropy is calculated over the MC samples 
dimension for each prediction point.
"""
function npe_per_timepoint(y_pred::AbstractArray{T,4}, mask::AbstractArray{Bool,3}) where T <: Number
    n_features, n_timepoints, n_samples, n_mc_samples = size(y_pred)
    npe_per_t_s = zeros(T, n_timepoints, n_samples)

        for t in 1:n_timepoints
            for s in 1:n_samples
                if mask[1, t, s] == true
                # Get MC samples for this prediction point
                pred_dist = y_pred[:, t, s, :]
                # Calculate negative entropy over MC samples
                pred_softmax = softmax(pred_dist, dims=1)
                npe_val = -sum(pred_softmax .* log.(pred_softmax .+ 1e-10))
                
                # Store NPE for this feature, time point, and sample
                npe_per_t_s[t, s] = npe_val / n_mc_samples
                end
            end
        end

    return npe_per_t_s
end

function npe_per_timepoint(y_pred::AbstractArray{T,4}) where T <: Number
    n_features, n_timepoints, n_samples, n_mc_samples = size(y_pred)
    npe_per_t_s = zeros(T, n_timepoints, n_samples)

        for t in 1:n_timepoints
            for s in 1:n_samples
                # Get MC samples for this prediction point
                pred_dist = y_pred[:, t, s, :]
                # Calculate negative entropy over MC samples
                pred_softmax = softmax(pred_dist, dims=1)
                npe_val = -sum(pred_softmax .* log.(pred_softmax .+ 1e-10))
                
                # Store NPE for this feature, time point, and sample
                npe_per_t_s[t, s] = npe_val / n_mc_samples
            end
        end

    return npe_per_t_s
end

"""
    acc(y_true::AbstractArray{T,3}, y_pred::AbstractArray{T,3}, mask::AbstractArray{Bool,3})

Calculate classification accuracy for 3D data with 3D mask.

# Arguments
- `y_true::AbstractArray{T,3}`: Ground truth labels, shape (n_features, n_timepoints, n_samples)
- `y_pred::AbstractArray{T,3}`: Predicted labels, shape (n_features, n_timepoints, n_samples)
- `mask::AbstractArray{Bool,3}`: Boolean mask indicating valid entries

# Returns
- Classification accuracy as a float between 0 and 1

# Description
Computes the accuracy by comparing predicted and true labels only at valid positions
indicated by the mask. For 3D predictions, it directly compares the values.
"""
function acc(y_true::AbstractArray{T,3}, 
             y_pred::AbstractArray{T,3}, 
             mask::AbstractArray{Bool,3}) where T <: Number

    n_features, n_timepoints, n_samples = size(y_true)
    correct = 0
    total = 0

    for f in 1:n_features
        for i in 1:n_samples
            for t in 1:n_timepoints
                if mask[f, t, i]
                    # For 3D predictions, directly compare values
                    pred_class = round(Int, y_pred[f, t, i])
                    true_class = round(Int, y_true[f, t, i])
                    
                    if pred_class == true_class
                        correct += 1
                    end
                    total += 1
                end
            end
        end
    end

    return total > 0 ? correct / total : 0.0
end

"""
    acc(y_true::AbstractArray, y_pred::AbstractArray, mask::AbstractArray{Bool})

Calculate classification accuracy for masked data.

# Arguments
- `y_true::AbstractArray`: Ground truth labels
- `y_pred::AbstractArray`: Predicted labels or probabilities
- `mask::AbstractArray{Bool}`: Boolean mask indicating valid entries

# Returns
- Classification accuracy as a float between 0 and 1

# Description
Computes the accuracy by comparing predicted and true labels only at valid positions
indicated by the mask. If y_pred contains probabilities, it takes the argmax to get
predicted class labels.
"""
function acc(y_true::AbstractArray{T,3}, 
             y_pred::AbstractArray{T,4}, 
             mask::AbstractArray{Bool,3}) where T <: Number

    n_features, n_timepoints, n_samples = size(y_true)
    correct = 0
    total = 0

    for f in 1:n_features
        for i in 1:n_samples
            for t in 1:n_timepoints
                if mask[f, t, i]
                    # Convert predictions to class labels (argmax for probabilities)
                    if size(y_pred, 4) > 1
                        pred_class = argmax(y_pred[f, t, i, :])
                    else
                        pred_class = round(Int, y_pred[f, t, i, 1])
                    end
                    
                    true_class = round(Int, y_true[f, t, i])
                    
                    if pred_class == true_class
                        correct += 1
                    end
                    total += 1
                end
            end
        end
    end

    return total > 0 ? correct / total : 0.0
end

"""
    empirical_crps(y_true::AbstractArray, 
                  y_pred_samples::AbstractArray, 
                  mask::AbstractArray{Bool,3})

Compute the empirical Continuous Ranked Probability Score (CRPS) for probabilistic forecasts.

# Arguments
- `y_true::AbstractArray{T1,3}`: Ground truth values, shape (n_features, n_timepoints, n_samples).
- `y_pred_samples::AbstractArray{T2,4}`: Predicted samples, shape (n_features, n_timepoints, n_samples, n_draws).
- `mask::AbstractArray{Bool,3}`: Boolean mask indicating valid entries, shape (n_features, n_timepoints, n_samples).

# Returns
- Mean empirical CRPS over all valid points.
"""
function empirical_crps(y_true::AbstractArray{T1,3}, 
                       y_pred_samples::AbstractArray{T2,4}, 
                       mask::AbstractArray{Bool,3}) where {T1 <: Number, T2 <: Number}

    n_features, n_timepoints, n_samples = size(y_true)
    
    # Promote to common type to handle Float32/Float64 mismatch
    CommonType = promote_type(T1, T2)
    total_crps = zero(CommonType)
    count = 0

    for f in 1:n_features
        for i in 1:n_samples
            for t in 1:n_timepoints
                if mask[f, t, i]
                    y_obs = CommonType(y_true[f, t, i])
                    preds = CommonType.(y_pred_samples[f, t, i, :])

                    term1 = mean(abs.(preds .- y_obs))
                    term2 = 0.5 * mean(abs.(preds .- preds'))

                    total_crps += (term1 - term2)
                    count += 1
                end
            end
        end
    end

    return total_crps / count
end