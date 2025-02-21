"""

"""


function irregularize(y1,y2, mask1, mask2)
    mask1=copy(mask1)
    mask2=copy(mask2)
    for i in 1:size(y1)[3]
        for j in 1:size(y1)[2]
            samp=rand(0:1)<0.8
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

function interp!(ts, x::AbstractMatrix, t)
   return CRC.@ignore_derivatives[linear_interpolation(ts, view(x, i, :), extrapolation_bc=Line())(t) for i in axes(x, 1)]
end


function interp!(ts, x::AbstractArray, t)
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






