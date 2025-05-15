function train(model, θ, st, ts, loss_fn, eval_fn, viz_fn, train_loader, val_loader, config, exp_path)
    # Create optimizer from config
    opt = eval(Meta.parse(config["optimizer"]))
    tstate = Training.TrainState(model, θ, st, opt)
    
    # Keep the lambda schedule for KL annealing
    λ_schedule = frange_cycle_linear(config["epochs"]+1, 0.0f0, 1.0f0, 10, 0.3f0)
    
    # Initialize exponential learning rate schedule
    initial_lr = config["learning_rate"]
    gamma = config["lr_decay"]["gamma"] 
    step_every = config["lr_decay"]["step_every"]  
    
    # Pre-compute learning rates for each epoch
    lr_schedule = [initial_lr * (gamma ^ (floor(Int, (epoch-1) / step_every))) for epoch in 1:config["epochs"]]

    n_batches = length(train_loader)
    θ_best = nothing
    best_val_metric = Inf
    counter = 0
    @info "Training started"
    stime = time()
    
    for epoch in 1:config["epochs"]
        # Apply learning rate for this epoch
        current_lr = lr_schedule[epoch]
        Optimisers.adjust!(tstate.optimizer_state, current_lr)
        
        train_loss = 0.f0
        kl_term = 0.f0
        recon_loss = 0.f0
        recon_loss1 = 0.f0
        recon_loss2 = 0.f0


        for batch in train_loader
            _, loss, (kl_loss, r_loss, r_loss1, r_loss2), tstate = Training.single_train_step!(AutoZygote(), loss_fn, (batch, ts, λ_schedule[epoch]), tstate) 
            train_loss += loss
            kl_term += kl_loss
            recon_loss += r_loss
            recon_loss1 += r_loss1
            recon_loss2 += r_loss2

        end

        θ = tstate.parameters
        st = tstate.states

        if epoch % config["log_freq"] == 0
            
            @printf("Epoch %d/%d: \t Training loss: %.3f \t λ: %.3f \t LR: %.6f \t Kl_term:%.3f \t recon_loss:%.3f\t recon_loss1: %.3f\t recon_loss2:%.3f  \n", 
                    epoch, config["epochs"], train_loss/n_batches, λ_schedule[epoch], current_lr, kl_term/n_batches, 
                    recon_loss/n_batches, recon_loss1/n_batches, recon_loss2/n_batches)
                    
            (val_metric, val_metric1, val_metric2) = validate(model, θ, st, ts, val_loader, eval_fn, config["validation"])
            @printf("Validation metric: %.3f\t val_metric1:%.3f\t val_metric2:%.3f\n", val_metric, val_metric1, val_metric2)

            if epoch % config["viz_freq"] == 0
                #viz_fn(model, θ, st, ts, first(train_loader), config["validation"]; sample_n=1)
            end

            if val_metric < best_val_metric
                @info "Saving best model!"
                best_val_metric = val_metric
                θ_best = copy(θ)
                save_state = (θ=θ_best, st=st, epoch=epoch)
                #save_object(joinpath(exp_path, "bestmodel.jld2"), save_state)
                counter = 0
            else 
                if counter > config["stop_patience"]
                    @printf("No more hope training this one! Early stopping at epoch: %.f\n", epoch)
                    return θ_best
                end
                counter += 1
            end 
        end 
    end
    ttime = time() - stime
    @info "Training finished in $(ttime) seconds"
    @info "Best validation metric: $(best_val_metric)"
    return θ_best
end

function validate(model, θ, st, ts, val_loader, eval_fn, config)
    val_metric = 0.0f0
    val_metric1= 0.0f0
    val_metric2= 0.0f0
    for batch in val_loader
        (val_m, val_m1, val_m2)= eval_fn(model, θ, st, ts, batch, config)
        val_metric += val_m
        val_metric1 += val_m1
        val_metric2 += val_m2
    end
    return (val_metric/length(val_loader), val_metric1/length(val_loader), val_metric2/length(val_loader))
end