
function train(model, θ, st, ts, loss_fn, eval_fn, viz_fn, train_loader, val_loader, config, exp_path)
    # Create optimizer from config
    opt = eval(Meta.parse(config["optimizer"]))
    tstate = Training.TrainState(model, θ, st, opt)
    
    λ_schedule = frange_cycle_linear(config["epochs"]+1, 0.0f0, 5.0f0, 1, 0.3f0)

    n_batches = length(train_loader)
    θ_best = nothing
    best_val_metric = -Inf
    counter = 0
    @info "Training started"
    for epoch in 1:config["epochs"]
        stime = time()
        train_loss = 0.f0
        kl_term = 0.f0

        for batch in train_loader
            _, loss, kl_loss, tstate = Training.single_train_step!(AutoZygote(), loss_fn, (batch, ts, λ_schedule[epoch]), tstate) 
            train_loss += loss
            kl_term += kl_loss
        end

        θ = tstate.parameters
        st = tstate.states

        if epoch % config["log_freq"] == 0
            ttime = time() - stime
            @printf("Epoch %d/%d: \t Training loss: %.3f \t Kl_term:%.3f  \t Time/epoch: %.3f\n", 
                    epoch, config["epochs"], train_loss/n_batches, kl_term/n_batches, ttime/config["log_freq"])
                    val_metric = validate(model, θ, st, ts, val_loader, eval_fn, config["validation"])
            @printf("Validation metric: %.3f\n", val_metric)

            if epoch % config["viz_freq"] == 0
                viz_fn(model, θ, st, ts, first(val_loader), config["validation"]; sample_n=1)
            end

            if val_metric > best_val_metric
                @info "Saving best model!"
                best_val_metric = val_metric
                θ_best = copy(θ)
                save_state = (θ=θ_best, st=st, epoch=epoch)
                save_object(joinpath(exp_path, "bestmodel.jld2"), save_state)
                counter = 0
            else 
                if counter > config["stop_patience"]
                    @printf("No more hope training this one! Early stopping at epoch: %.f\n", epoch)
                    return θ_best
                elseif counter > config["lrdecay_patience"]
                    new_lr = config["learning_rate"] / counter
                    @printf("No improvement for %d consecutive epochs; Adjusting learning rate to: %.4f\n", 
                            config["lrdecay_patience"], new_lr)
                    Optimisers.adjust!(tstate.optimizer_state, new_lr)
                end
                counter += 1
            end 

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

