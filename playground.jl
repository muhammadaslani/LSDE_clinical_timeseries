# using Interpolations
# using CairoMakie

# # Simple B-spline example
# println("=== Simple B-spline Example ===")

# # Simple data points
# x = [1.0, 2.0, 3.0, 4.0, 5.0]  # time points
# y = [1.0, 4.0, 2.0, 5.0, 3.0]  # values

# # Create B-spline interpolation
# itp = interpolate(y, BSpline(Cubic(Line(OnGrid()))))
# # Use range instead of vector for scaling
# x_range = range(1.0, 5.0, length=5)
# scaled_itp = scale(itp, x_range)
# #scaled_itp=itp
# # Test interpolation at a point
# t_test = 2.5
# result = scaled_itp(t_test)
# println("Value at t=$t_test: $result")

# # Plot comparison
# t_fine = 1.0:0.1:5.0
# bspline_values = [scaled_itp(t) for t in t_fine]

# # Linear interpolation for comparison
# linear_itp = linear_interpolation(x, y)
# linear_values = [linear_itp(t) for t in t_fine]

# fig = Figure(size=(600, 400))
# ax = Axis(fig[1, 1], xlabel="Time", ylabel="Value", title="Simple B-spline Example")

# scatter!(ax, x, y, color=:black, markersize=12, label="Data Points")
# lines!(ax, t_fine, bspline_values, color=:blue, linewidth=2, label="B-spline")
# lines!(ax, t_fine, linear_values, color=:red, linewidth=2, label="Linear")
# axislegend(ax)

# display(fig)









# function interp!(ts, x::AbstractMatrix, t; interp_type=:BSpline)
#     return  [interpolate(view(x, i, :), BSpline(Cubic(Line(OnGrid()))))(t)  for i in axes(x, 1)]
# end

# inttt=interp!(ts_test, x_matrix, t_interp; interp_type=:BSpline)

# function interp!(ts, x::AbstractArray, t; interp_type=:BSpline)
    
#         # Determine the actual observation times for x
#         n_obs = min(length(ts), size(x, 2))
#         obs_times = ts[1:n_obs]
        
#         # If x has more time points than ts, we'll extend the time points
#         if size(x, 2) > length(ts)
#             if length(ts) > 1
#                 # Calculate the time step based on the last two points in ts
#                 time_step = ts[end] - ts[end-1]
#             else
#                 # If ts has only one point, assume a unit time step
#                 time_step = 1
#             end
            
#             # Extend obs_times with consistent time scaling
#             extra_times = [ts[end] + i * time_step for i in 1:(size(x, 2) - length(ts))]
#             obs_times = vcat(obs_times, extra_times)
#         end
        
#         # Create interpolation for each feature and batch
#         return [
#             let interp_obj = interpolate(view(x, i, 1:length(obs_times), b), BSpline(Cubic(Line(OnGrid()))))
#                 interp_obj(t)
#             end
#             for i in axes(x, 1), b in axes(x, 3)
#         ]
# end


# # Create test data - 2D matrix (2 features x 6 timepoints)
# ts_test = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
# x_matrix = [1.0 2.0 3.0 2.0 1.0 0.5;    # Feature 1
#             0.5 1.5 2.5 3.0 2.0 1.0]     # Feature 2
# x_3d= rand(2, 6, 2) # 2 features, 6 timepoints, 1 batch
# # Test interpolation at t=2.5
# t_interp = 2.5
# inttt = interp!(ts_test, x_3d, 5; interp_type=:BSpline)