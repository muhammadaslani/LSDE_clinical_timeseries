function load_data(; n_samples=512, sampling_rate=1, batch_size=32)
    time_series_dataset = load_multiple_files("/Volumes/Mine/Academic/PhD/datasets/Physionet 2012 challenge dataset/Data/set_a_data/time_series")
    outcomes_file = "/Volumes/Mine/Academic/PhD/datasets/Physionet 2012 challenge dataset/Data/set_a_data/Outcomes-a.txt"

  
    time_series_variables = ["ALP", "HR", "DiasABP", "Na", "Lactate", "NIDiasABP", "PaO2", "WBC", "pH", "Albumin", "ALT", "Glucose", "SaO2",
        "Temp", "AST", "Bilirubin", "BUN", "RespRate", "Mg", "HCT", "SysABP", "FiO2", "K", "GCS",
        "Cholesterol", "NISysABP", "TroponinT", "MAP", "TroponinI", "PaCO2", "Platelets", "Urine", "NIMAP",
        "Creatinine", "HCO3"]
    
    static_features_df,static_features_matrix =extract_static_features("/Volumes/Mine/Academic/PhD/datasets/Physionet 2012 challenge dataset/Data/set_a_data/time_series")
        
    variables_of_interest=["MAP", "HR", "RespRate", "Temp", "SaO2"]
    timeseries, masks = create_tensor(time_series_dataset, variables_of_interest)
    inputs_data, _ = create_tensor(time_series_dataset, ["MechVent"])
    obs_data=join_static_and_timeseries(static_features_matrix, timeseries)

    inputs_data = inputs_data[:, 1:sampling_rate:end, 1:n_samples] |> Array{Float32}
    obs_data = obs_data[:, 1:sampling_rate:end, 1:n_samples] |> Array{Float32}
    timeseries = timeseries[:, 1:sampling_rate:end, 1:n_samples] |> Array{Float32}
    masks= masks[:, 1:sampling_rate:end,1:n_samples] |> Array{Bool}

    data = (inputs_data, obs_data, timeseries, masks)
    train_data, test_data, val_data = splitobs(data, at=(0.5, 0.3))
    train_loader = DataLoader(train_data, batchsize=batch_size, shuffle=true)
    val_loader = DataLoader(val_data, batchsize=batch_size, shuffle=true)
    test_loader=DataLoader(test_data, batchsize=batch_size, shuffle=false)
    return  data, train_loader, val_loader, test_loader, time_series_dataset
end

function load_data(spl; n_samples=512, sampling_rate=1, batch_size=32)
    time_series_dataset = load_multiple_files("/Volumes/Mine/Academic/PhD/datasets/Physionet 2012 challenge dataset/Data/set_a_data/time_series")
    outcomes_file = "/Volumes/Mine/Academic/PhD/datasets/Physionet 2012 challenge dataset/Data/set_a_data/Outcomes-a.txt"

  
    time_series_variables = ["ALP", "HR", "DiasABP", "Na", "Lactate", "NIDiasABP", "PaO2", "WBC", "pH", "Albumin", "ALT", "Glucose", "SaO2",
        "Temp", "AST", "Bilirubin", "BUN", "RespRate", "Mg", "HCT", "SysABP", "FiO2", "K", "GCS",
        "Cholesterol", "NISysABP", "TroponinT", "MAP", "TroponinI", "PaCO2", "Platelets", "Urine", "NIMAP",
        "Creatinine", "HCO3"]
    
    static_features_df,static_features_matrix =extract_static_features("/Volumes/Mine/Academic/PhD/datasets/Physionet 2012 challenge dataset/Data/set_a_data/time_series")
    
    variables_of_interest=["MAP", "HR", "RespRate", "Temp", "SaO2"]
    timeseries, masks = create_tensor(time_series_dataset, variables_of_interest)
    inputs_data, _ = create_tensor(time_series_dataset, ["MechVent"])
    obs_data=join_static_and_timeseries(static_features_matrix, timeseries)

    inputs_data = inputs_data[:, 1:sampling_rate:end, 1:n_samples] |> Array{Float32}
    obs_data = obs_data[:, 1:sampling_rate:end, 1:n_samples] |> Array{Float32}
    timeseries = timeseries[:, 1:sampling_rate:end, 1:n_samples] |> Array{Float32}
    masks= masks[:, 1:sampling_rate:end,1:n_samples] |> Array{Bool}
    
    obs_data_hist, obs_data_fut = obs_data[:, 1:spl, :], obs_data[:, spl+1:end, :]
    masks_hist, masks_fut = masks[:, 1:spl, :], masks[:, spl+1:end, :]
    inputs_data_hist, inputs_data_fut = inputs_data[:, 1:spl, :], inputs_data[:, spl+1:end, :]
    timeseries_hist, timeseries_fut = timeseries[:, 1:spl, :], timeseries[:, spl+1:end, :]

    data = (inputs_data_hist, obs_data_hist, timeseries_hist, masks_hist,inputs_data_fut, obs_data_fut, timeseries_fut, masks_fut)
    train_data, val_data, test_data = splitobs(data, at=(0.5, 0.3))
    train_loader = DataLoader(train_data, batchsize=batch_size, shuffle=true)
    val_loader = DataLoader(val_data, batchsize=batch_size, shuffle=true)
    test_loader=DataLoader(test_data, batchsize=batch_size, shuffle=false)
    return  data, train_loader, val_loader, test_loader, time_series_dataset
end




function load_outcomes(filepath::String;)
    lines = readlines(filepath)
    outcomes_data = zeros(Float32, size(lines)[1] - 1, 6)
    for n in 2:1:size(lines)[1]
        outcomes_data[n-1, :] = [isnothing(x) ? NaN : x for x in tryparse.(Float32, split(lines[n], ","))]
    end
    outcomes_data = permutedims(outcomes_data, [2, 1])
    outcomes_data = replace!(outcomes_data, -1 => NaN)
    masks = .!isnan.(outcomes_data)
    outcomes_data = replace!(outcomes_data, NaN => 0.0)
    return outcomes_data, masks
end


function load_physionet_file(filepath::String; combine_method::Function=mean)

    lines = readlines(filepath)
    header = split(lines[1], ",")
    times = String[]
    parameters = String[]
    values = Float64[]

    for line in lines[2:end]
        time, param, val = split(line, ",")
        push!(times, time)
        push!(parameters, param)
        val_float = tryparse(Float64, val)
        push!(values, val_float === nothing ? missing : val_float)
    end

    df_long = DataFrame(
        Time=times,
        Parameter=parameters,
        Value=values
    )

    function time_to_hour(time_str)
        h, _ = parse.(Int, split(time_str, ":"))
        return h
    end

    # # Aggregate data based on the hour of the day
    df_long.Hour = time_to_hour.(df_long.Time)
    df_agg = combine(groupby(df_long, [:Hour, :Parameter]), :Value => combine_method =>  :Value)
    df_wide = DataFrames.unstack(df_agg, :Hour, :Parameter, :Value)
    sort!(df_wide, :Hour)
    # Aggregate data based on the original time points, ignoring hours
    # df_agg = combine(groupby(df_long, [:Time, :Parameter]), :Value => combine_method => :Value)
    # df_wide = DataFrames.unstack(df_agg, :Time, :Parameter, :Value)

    # sort!(df_wide, :Time)


    return df_wide
end



function load_multiple_files(directory::String, pattern::String="*.txt")
    files = Base.filter(f -> endswith(f, ".txt"), readdir(directory, join=true))
    all_data = Dict{String,DataFrame}()

    for file in files
        try
            df = load_physionet_file(file)
            record_id = basename(file)[1:end-4]
            all_data[record_id] = df
        catch e
            println("Error loading file $file: $e")
        end
    end

    return all_data
end


function create_tensor(data_dict::Dict{String,DataFrame}, features::Vector{String})
    n_samples = length(data_dict)
    n_features = length(features)
    max_timesteps = maximum(
        nrow(df) for (_, df) in data_dict
    )

    tensor = fill(NaN, (n_features, max_timesteps, n_samples))

    for (i, (record_id, df)) in enumerate(data_dict)
        for (j, feature) in enumerate(features)
            if feature in names(df)
                valid_times = findall(.!ismissing.(df[:, feature]))
                if !isempty(valid_times)
                    tensor[j, valid_times, i] = df[valid_times, feature]
                end
            end
        end
    end

    mask = .!isnan.(tensor)

    # Replace NaN values with the mean of non-NaN values in the second dimension
    for j in 1:n_features
        for i in 1:n_samples
            for t in 1:max_timesteps
                if isnan(tensor[j, t, i])
                    non_nan_values = tensor[j, .!isnan.(tensor[j, :, i]), i]
                    if !isempty(non_nan_values)
                        tensor[j, t, i] = round(mean(non_nan_values))
                    else
                        tensor[j, t, i] = 0.0  # or some other default value
                    end
                end
            end
        end
    end

    return tensor, mask
end


function extract_static_features(folder_path)
    if !isdir(folder_path)
        error("The specified folder does not exist: $folder_path")
    end
    
    files = Base.filter(f -> endswith(f, ".txt"), readdir(folder_path, join=true))
    
    if isempty(files)
        error("No .txt files found in the specified folder")
    end
        n_files = length(files)
    feature_matrix = Matrix{Union{Float64, Missing}}(undef, n_files, 5)
    
    for (i, filepath) in enumerate(files)
        try
            df = CSV.read(filepath, DataFrame, header=[:Time, :Parameter, :Value])
            
            # Extract initial parameters (they appear at Time="00:00")
            initial_records = df[df.Time .== "00:00", :]
                for (j, feature) in enumerate(["Weight", "Age", "Gender", "ICUType", "Height"])
                feature_row = initial_records[initial_records.Parameter .== feature, :]
                
                if size(feature_row, 1) > 0
                    # Try to convert value to float
                    try
                        if typeof(feature_row.Value[1]) <: AbstractString
                            feature_matrix[i, j] = parse(Float64, strip(feature_row.Value[1]))
                        else
                            feature_matrix[i, j] = Float64(feature_row.Value[1])
                        end
                    catch
                        feature_matrix[i, j] = missing
                    end
                else
                    feature_matrix[i, j] = missing
                end
            end
            
        catch e
            @warn "Error processing file: $filepath"
            println("Error: ", e)
            feature_matrix[i, :] .= missing
        end
    end
    
    # Create DataFrame with file names and features (now including height)
    features_df = DataFrame(
        file_name = basename.(files),
        weight = feature_matrix[:, 1],
        age = feature_matrix[:, 2],
        gender = feature_matrix[:, 3],
        icu_type = feature_matrix[:, 4],
        height = feature_matrix[:, 5]
    )
    return features_df, feature_matrix
end




function join_static_and_timeseries(static_matrix, timeseries_matrix)
    n_static_features = size(static_matrix, 2)
    n_timeseries_features, n_timepoints, n_samples = size(timeseries_matrix)
    
    combined_matrix = Array{Union{Float64, Missing}}(undef, 
                                                    n_static_features + n_timeseries_features,
                                                    n_timepoints,
                                                    n_samples)
    
    for sample in 1:n_samples
        for feature in 1:n_static_features
            combined_matrix[feature, :, sample] .= static_matrix[sample, feature]
        end
    end
    
    combined_matrix[n_static_features+1:end, :, :] = timeseries_matrix
    
    return combined_matrix
end



"""
    smooth_data(input_tensor; window_size=5)

Smooths the input tensor along the second dimension using a moving average filter.

# Arguments
- `input_tensor`: A 3-dimensional array to be smoothed.
- `window_size`: An optional integer specifying the size of the moving window. Default is 5.

# Returns
- A 3-dimensional array of the same size as `input_tensor`, where each element is the mean of the elements within the moving window along the second dimension.
"""

function smooth_data(input_tensor; window_size=5)
    smoothed_tensor = similar(input_tensor)

    for i in axes(input_tensor)[1]
        for k in axes(input_tensor)[3]
            for j in axes(input_tensor)[2]
                start_idx = max(1, j - window_size + 1)
                end_idx = min(size(input_tensor, 2), j + window_size - 1)
                smoothed_tensor[i, j, k] = mean(input_tensor[i, start_idx:end_idx, k])
            end
        end
    end
    return smoothed_tensor
end


"""
    z_normalize(X::AbstractArray{<:Real})

Z-normalizes the input array `X` along the last dimension.

For each element `x` in `X`, the z-normalized value `z` is computed as:
`z = (x - μ) / σ`, where `μ` is the mean and `σ` is the standard deviation
of the elements along the last dimension of `X`.

# Arguments
- `X::AbstractArray{<:Real}`: Input array of real numbers.

# Returns
- `AbstractArray{Float64}`: Z-normalized array with the same dimensions as `X`.

"""

function z_normalize(input_tensor)
    normalized_tensor = similar(input_tensor)
    for i in axes(input_tensor)[1]
                μ=mean(input_tensor[i,:,:])
                σ=std(input_tensor[i,:,:])
                normalized_tensor[i, :,:] = (input_tensor[i,:,:] .- μ) ./ σ
    end
    return normalized_tensor
end