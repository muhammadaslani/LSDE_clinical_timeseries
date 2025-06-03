# Evaluation functions for PKPD forecasting models

function eval_fn_nde(model, θ, st, ts, data, config)
    u_obs, covars_obs, x_obs, y₁_obs, y₂_obs, mask₁_obs, mask₂_obs, 
        u_forecast, covars_forecast, x_forecast, y₁_forecast, y₂_forecast, mask₁_forecast, mask₂_forecast = data
    batch_size= size(u_forecast)[end]
    solver = eval(Meta.parse(config["solver"]))
    kwargs_dict = Dict(Symbol(k) => v for (k, v) in config["kwargs"])
    Ex, Ey = predict(model, solver, vcat(covars_obs, reverse(y₁_obs, dims=2), reverse(y₂_obs, dims=2)), u_forecast, ts, θ, st, config["mcmc_samples"], cpu_device(); kwargs_dict...)
    ŷ₁_m, ŷ₂_m = dropmean(Ey[1], dims=4), dropmean(Ey[2], dims=4)
    eval_loss1 = CrossEntropy_Loss(ŷ₁_m, y₁_forecast, mask₁_forecast; agg=sum)/batch_size
    eval_loss2 = -poisson_loglikelihood(ŷ₂_m, y₂_forecast, mask₂_forecast)/batch_size
    eval_loss = eval_loss1 + eval_loss2
    return (eval_loss, eval_loss1, eval_loss2)
end

function eval_fn_rnn(model, θ, st, ts, data, config)
    u_obs, covars_obs, x_obs, y₁_obs, y₂_obs, masks₁_obs, masks₂_obs,
    u_for, covars_for, x_for, y₁_for, y₂_for, masks₁_for, masks₂_for = data
    
    batch_size = size(y₁_for)[end]
    
    # Combine inputs for RNN
    input_combined = vcat(x_obs, u_obs, covars_obs)
    
    # Forward pass
    ŷ, st = model(input_combined, θ, st)
    
    # Calculate evaluation losses
    eval_loss1 = 0.0f0
    eval_loss2 = 0.0f0
    
    if length(ŷ) >= 1
        μ₁, log_σ²₁ = ŷ[1][1], ŷ[1][2]
        valid_indx₁ = findall(masks₁_for .== 1)
        eval_loss1 = normal_loglikelihood(μ₁[valid_indx₁], log_σ²₁[valid_indx₁], y₁_for[valid_indx₁]) / batch_size
    end
    
    if length(ŷ) >= 2
        μ₂, log_σ²₂ = ŷ[2][1], ŷ[2][2]
        valid_indx₂ = findall(masks₂_for .== 1)
        eval_loss2 = normal_loglikelihood(μ₂[valid_indx₂], log_σ²₂[valid_indx₂], y₂_for[valid_indx₂]) / batch_size
    end
    
    total_eval_loss = eval_loss1 + eval_loss2
    
    return (total_eval_loss, eval_loss1, eval_loss2)
end

# Performance assessment function for k-fold validation
function assess_model_performance(performances, variables_of_interest; model_name="Model", model_type="lsde", 
                                forecast_fn=forecast_nde, plot_sample=false, sample_n=3, viz_fn=viz_fn_forecast_pkpd, 
                                models=nothing, params=nothing, states=nothing, data=nothing, timepoints=nothing, 
                                config=nothing, best_fold_idx=nothing)
    """
    Presents PKPD model performance across k-folds by calculating and printing
    mean and standard deviation for the three evaluation metrics:
    - Health status: cross-entropy loss
    - Tumor volume: RMSE
    - Cell count: negative log-likelihood
    Optionally plots a sample forecast from the best performing model.
    
    Arguments:
    - performances: Array of (crossentropy_health, rmse_tumor, nll_count) tuples from k-fold training
    - variables_of_interest: Array of output variable names (for backward compatibility)
    - model_name: Name of the model for display purposes
    - plot_sample: Boolean flag to enable sample plotting
    - models: Array of trained models (required if plot_sample=true)
    - params: Array of trained parameters (required if plot_sample=true)
    - states: Array of model states (required if plot_sample=true)
    - data: Test data for plotting (required if plot_sample=true)
    - timepoints: Time points for plotting (required if plot_sample=true)
    - config: Model configuration (required if plot_sample=true)
    - best_fold_idx: Index of best performing fold (required if plot_sample=true)
    """
    
    println("="^60)
    println("$model_name Performance Summary")
    println("="^60)
    
    # Extract metric values based on new format
    if model_type == "rnn"
        # RNN only returns RMSE (assuming legacy format)
        rmse_values = [perf for perf in performances]
        for (i, var_name) in enumerate(variables_of_interest)
            rmse_fold_values = [rmse[i] for rmse in rmse_values]
            rmse_mean = mean(rmse_fold_values)
            rmse_std = std(rmse_fold_values)
            println("$var_name:")
            println("  RMSE: $(round(rmse_mean, digits=4)) ± $(round(rmse_std, digits=4))")
        end
    else
        # Neural DE models now return (crossentropy_health, rmse_tumor, nll_count)
        crossentropy_health_values = [perf[1] for perf in performances]
        rmse_tumor_values = [perf[2] for perf in performances]
        nll_count_values = [perf[3] for perf in performances]
        
        # Calculate means and standard deviations
        crossentropy_mean = mean(crossentropy_health_values)
        crossentropy_std = std(crossentropy_health_values)
        rmse_mean = mean(rmse_tumor_values)
        rmse_std = std(rmse_tumor_values)
        nll_mean = mean(nll_count_values)
        nll_std = std(nll_count_values)
        
        println("Health Status:")
        println("  Cross-entropy: $(round(crossentropy_mean, digits=4)) ± $(round(crossentropy_std, digits=4))")
        println("Tumor Volume:")
        println("  RMSE: $(round(rmse_mean, digits=4)) ± $(round(rmse_std, digits=4))")
        println("Cell Count:")
        println("  Negative Log-likelihood: $(round(nll_mean, digits=4)) ± $(round(nll_std, digits=4))")
    end
    
    println("="^60)
    
    # Optional sample plotting
    if plot_sample && !isnothing(models) && !isnothing(params) && !isnothing(states) && 
       !isnothing(data) && !isnothing(timepoints) && !isnothing(config) && !isnothing(best_fold_idx)
        
        println("Generating sample forecast from best performing model (Fold $best_fold_idx)...")
        
        # Use the best model for forecasting
        best_model = models[best_fold_idx]
        best_params = params[best_fold_idx]
        best_state = states[best_fold_idx]
        
        # Extract data for plotting
        u_obs, covars_obs, x_obs, y₁_obs, y₂_obs, mask₁_obs, mask₂_obs,
        u_forecast, covars_forecast, x_forecast, y₁_forecast, y₂_forecast, mask₁_forecast, mask₂_forecast = data
        data_obs = (u_obs, covars_obs, x_obs, y₁_obs, y₂_obs, mask₁_obs, mask₂_obs)
        data_forecast = (u_forecast, covars_forecast, x_forecast, y₁_forecast, y₂_forecast, mask₁_forecast, mask₂_forecast)
        timepoints_obs, timepoints_forecast = timepoints
        
        # Generate forecast
        forecasted_data = forecast_fn(best_model, best_params, best_state, data_obs, 
                                    u_forecast, timepoints_forecast, config)
        
        # Create visualization
        fig = viz_fn(timepoints_obs, timepoints_forecast, data_obs, data_forecast, 
                    forecasted_data; sample_n=sample_n)
        
        return fig
    end
    
    return nothing
end
