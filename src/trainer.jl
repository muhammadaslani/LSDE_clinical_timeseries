function train(model, θ, st, ts, loss_fn, eval_fn, viz_fn, train_loader, val_loader, config)
    tstate = Training.TrainState(model, θ, st, config.optimizer)
    λ_schedule = frange_cycle_linear(config.epochs+1, 0.0f0, 1.0f0, 1, 0.5f0)

    n_batches = length(train_loader)
    θ_best = nothing; best_val_metric = -Inf; counter=0; 

    for epoch in 1:config.epochs
        stime = time()
        train_loss = 0.0; kl_term = 0.0;

        for batch in train_loader
            _, loss, kl_loss, tstate = Training.single_train_step!(AutoZygote(), loss_fn, (batch..., ts,  λ_schedule[epoch]), tstate) 
            train_loss += loss
            kl_loss += kl_loss
        end

        θ = tstate.parameters; st = tstate.states

        if epoch % config.log_freq == 0
            ttime = time() - stime
            @printf("Epoch %d/%d: \t Training loss: %.3f \t Kl_term:%.3f  \t Time/epoch: %.3f\n", epoch, conif.epochs, loss/n_batches, kl_term/n_batches, ttime/config.log_freq)
        end

        val_metric = validate(model, θ, st, ts, val_loader, eval_fn, config)
        @printf("Validation metric: %.3f\n", val_metric)


        if epoch % config.viz_freq == 0
            viz_fn(model, θ, st, val_loader, epoch)
        end

        if val_metric > best_val_metric
            best_val_metric = val_metric
            θ_best = copy(θ)
            save_state = (model=model, θ=θ_best, st=st, data_loader=val_loader, epoch=epoch)
            save_object(joinpath(config.save_path, "bestmodel.jld2"), save_state)
            counter = 0
        else 
            if counter > config.stop_patience
                @printf("No more hope training this one! Early stopping at epoch: %.f\n", epoch)
                return θ_best
            elseif counter > config.lrdecay_patience
                @printf("No improvment for %d consecutive epochs; Adjusting learning rate to: %.4f\n", config.lrdecay_patience, config.lr/counter)
                tstate.optimizer_state = Optimisers.adjust(tstate.optimizer_state, config.lr/counter) # Immutable optimizer state. Find another way to adjust the learning rate
            end
            counter += 1
        end 

    end  
    return θ_best
end


function validate(model, θ, st, ts, val_loader, eval_fn, config)
    val_metric = 0.0
    for batch in val_loader
        val_metric += eval_fn(model, θ, st, ts, batch, config)
    end
    return val_metric/length(val_loader)
end




function vizualize(model, data_loader, ts, θ, st, dev, sample_n, ch; kwargs...)
    u, y, x, ts_pred = first(data_loader)
    x = @view x[:,:, sample_n]

    fig = Figure(size = (800, 600), backgroundcolor = :transparent)
    ax1 = CairoMakie.Axis(fig[1,1], xlabel = "time(ms)", ylabel = "Cell count", backgroundcolor = :transparent, limits = (nothing, (0, 1.3)))
    ax2 = CairoMakie.Axis(fig[1,2], xlabel = "time(ms)", ylabel = "tumor size", backgroundcolor = :transparent, limits = (nothing, (0, 1.3)))

    ch = ch == 0 ? rand(1:size(y,1)) : ch
    θ = θ |> cpu_device() 
    Ey, Ex = predict(model, EM(), y, ts_pred, u, ps, st, n_samples, dev; kwargs...)
    ŷₘ = selectdim(dropmean(Ey, dims=4), 3, sample_n); ŷₛ = selectdim(dropmean(std(Ey, dims=4), dims=4), 3, sample_n)
    x̂ₘ = selectdim(dropmean(Ex, dims=4), 3, sample_n); x̂ₛ = selectdim(dropmean(std(Ex, dims=4), dims=4), 3, sample_n)
    dist = Poisson.(ŷₘ)
    pred_count = rand.(dist)

    lines!(ax1, ts, y[ch,:], color = :red, linewidth = 3, label = "gt counts")
    lines!(ax1, ts, pred_count, linewidth = 2, color = (:dodgerblue2, 0.5), label = "predicted count")


    lines!(ax2, ts, x[ch,:], color = :red, linewidth = 3, label = "gt tumor size")
    lines!(ax2, ts, ŷₘ[ch,:], linewidth = 2, color = (:dodgerblue2, 0.5))
    band!(ax2, t, ŷₘ[ch,:] .-   sqrt.(ŷₛ[ch,:]) , ŷₘ[ch,:] .+ sqrt.(ŷₛ[ch,:]), color= (:dodgerblue2, 0.5),  label = "predicted tumor size")

    return fig
end