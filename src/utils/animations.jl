function animate_cont(x, ts, filename, color, ylabel)
    fig = Figure(size = (900, 600))
    N = size(x, 1)
    ax = CairoMakie.Axis(fig[1, 1],
    xgridvisible = false,
    ygridvisible = false, 
    ylabel = ylabel)

    # Create an array of Observables, one for each line
    lines_obs = [Observable(Point2f[]) for _ in 1:N]

    limits!(ax, extrema(ts)..., extrema(x)...)

    # Set up the animation
    framerate = 20
    total_frames = length(ts)

    # Plot all lines
    for i in 1:N
        lines!(ax, lines_obs[i], color = color, linewidth = 3)
    end

    CairoMakie.record(fig, filename, 1:total_frames; framerate = framerate) do frame
        for i in 1:N
            lines_obs[i][] = Point2f[(ts[t], x[i, t]) for t in 1:frame]
        end

        # Adjust axis limits
        # Optional: Add a title or other annotations
        ax.title = "Time: $(round(ts[frame], digits=2))"
    end
    println("Animation saved as '$filename'")
end



function animate_spikes(y_sample, ts, filename; colors_ = nothing, normalize_color = false)
    println("Creating animation...")
    fig = Figure(size = (900, 600))
    ax = CairoMakie.Axis(fig[1, 1], backgroundcolor = :transparent)
    hidedecorations!(ax)
    hidespines!(ax)

    C, T = size(y_sample)
    
    # Create an array of Observables, one for each spike train
    spike_obs = [Observable(Point2f[]) for _ in 1:C]
    color_obs = [Observable(Float64[]) for _ in 1:C]

    # Set up color scheme
    cmap = isnothing(colors_) ? cgrad([:lightgrey, :black]) : cgrad(colors_)

    # Calculate firing rates for color normalization
    if normalize_color
        firing_rates = sum(y_sample .!= 0, dims=2) ./ size(y_sample, 2)
        max_rate = maximum(firing_rates)
    end

    # Plot all spike trains
    for i in 1:C
        scatter!(ax, spike_obs[i], color = color_obs[i], colormap = cmap, markersize = 6, marker = :rect)
    end

    # Set preset axis limits
    x_min, x_max = extrema(ts)
    y_min, y_max = 0, C + 1  # Set y-axis to match channel numbers
    
    # Add a small buffer to the limits for visual appeal
    x_range = x_max - x_min
    limits!(ax, 
        x_min - 0.02 * x_range, x_max + 0.02 * x_range,
        y_min, y_max
    )

    # Set up the animation
    framerate = 20
    CairoMakie.record(fig, filename, 1:T; framerate = framerate) do frame
        for i in 1:C
            new_points = [Point2f(ts[t], i) for t in 1:frame if y_sample[i, t] != 0]
            spike_obs[i][] = new_points
            
            if normalize_color
                color_intensity = firing_rates[i] / max_rate
            else
                color_intensity = 1.0
            end
            
            color_obs[i][] = fill(color_intensity, length(new_points))
        end

        # Update title
        ax.title = "Time: $(round(ts[frame], digits=2))"
    end

    println("Animation saved as '$filename'")
end


function animate_oscillators_(z, ts, filename)
    N = Int(size(z, 1) / 2)
    x = z[1:N, :]
    y = z[N+1:end, :]
    
    fig = Figure(size = (900, 600))
    ax = CairoMakie.Axis(fig[1, 1], backgroundcolor = :transparent)
    hidedecorations!(ax)
    hidespines!(ax)
    
    limits!(ax, -2, 2, -2, 2)

    lines_obs = [Observable(Point2f[]) for _ in 1:N]
    scatter_obs = Observable(Point2f[])

    for i in 1:N
        lines!(ax, lines_obs[i], linewidth = 3.5, alpha = 0.8)
    end
    scatter!(ax, scatter_obs, markersize = 20)

    CairoMakie.record(fig, filename, 1:size(x, 2); framerate = 20) do frame
        for i in 1:N
            lines_obs[i][] = Point2f[(x[i, t], y[i, t]) for t in 1:frame]
        end
        scatter_obs[] = Point2f[(x[i, frame], y[i, frame]) for i in 1:N]
    end
end



function animate_hand(b, ts, b̂_m, b̂_sd, file_name; framerate=20)
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
    fig = Figure(size=(900, 600))
    ax = CairoMakie.Axis(fig[1, 1], 
              xlabel = "X position (cm)", 
              ylabel = "Y position (cm)", 
              aspect = DataAspect(), 
              xgridvisible = false,
              ygridvisible = false)

    limits!(ax, -10, 10, -10, 10)

    # Create observables
    current_gt_obs = Observable(Point2f(b[1, 1], b[2, 1]))
    gt_trail_obs = Observable(Point2f[])

    current_prediction_obs = Observable(Point2f(b̂_m[1, 1], b̂_m[2, 1]))
    prediction_trail_obs = Observable(Point2f[])

    ellipse_obs = Observable(Point2f[]) 

    # Plot observables
    lines!(ax, gt_trail_obs, color = (atom_one_dark[:purple], 0.8), linewidth = 3)
    scatter!(ax, current_gt_obs, color = atom_one_dark[:purple], markersize = 15, label = "Ground Truth")

    lines!(ax, prediction_trail_obs, color = (atom_one_dark[:cyan], 0.8), linewidth = 3)
    poly!(ax, ellipse_obs, color = (atom_one_dark[:cyan], 0.3), strokewidth = 0)
    scatter!(ax, current_prediction_obs, color = atom_one_dark[:cyan], markersize = 15, label = "Prediction")

    # Add legend
    axislegend(ax, backgroundcolor = :transparent)

    # Record the animation
    CairoMakie.record(fig, file_name, 1:T; framerate = framerate) do frame
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

    println("Animation saved as '$file_name'")
end