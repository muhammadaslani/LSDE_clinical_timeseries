function train(model::LatentSDE, p, st, train_loader, val_loader, config, ts, dev)
    init_epoch = config.init_epoch
    L = frange_cycle_linear(init_epoch+config.epochs+1, 0.0f0, 1.0f0, 1, 0.5f0)
    θ_best = nothing; best_metric = -Inf; counter=0; count_thresh = 10;
    stime = time()
    @info "Training ...."

   function loss(p, u, y)
        u = u |> dev; y = y|>dev;
        ŷ, px₀, kl_pq = model(y, u, ts, p, st)
        batch_size = size(y)[end]
        recon_loss = -poisson_loglikelihood(ŷ, y)/batch_size
        kl_init = kl_normal(px₀...)/batch_size
        kl_path = mean(kl_pq[end,:])
        kl_loss =  kl_init + kl_path
        l =  recon_loss + L[epoch+1]*kl_loss
        return l, recon_loss, kl_loss
    end


    callback = function(opt_state, l, recon_loss , kl_loss)
        θ = opt_state.u
        if opt_state.iter % length(train_loader) == 0
            epoch += 1
            if epoch % config.log_freq == 0
                t_epoch = time() - stime
                @printf("Time/epoch %.2fs \t Current epoch: %d, \t Loss: %.2f, PoissonLL: %d, KL: %.2f\n", t_epoch/config.log_freq, epoch, l, recon_loss, kl_loss)
                val_pll = 0.f0
                for (u, y, x, ts_pred) in val_loader
                    u = u |> dev; y = y|>dev; x = x|>dev
                    Ey, Ex = predict(model, config.solver, y, ts_pred, u, ps, st, n_samples, dev; config.kwargs...)
                    ŷₘ = dropmean(Ey, dims=4)
                    val_pll += poisson_loglikelihood(ŷₘ, y)
                end

                val_pll /= round(length(val_loader), digits=3)
                @printf("Validation PLL: %.3f\n", val_pll)
                train_loss = round(l, digits=3)
                @wandblog val_pll train_loss step=epoch
                if val_pll > best_metric
                    best_metric = val_pll
                    @wandblog best_metric step=epoch
                    θ_best = copy(θ)
                    @printf("Saving best model\n")
                    save_state = (model=model, θ=θ_best |> cpu, st=st |> cpu, data_loader=val_loader, epoch=epoch)
                    save_object(joinpath(config.save_path, "bestmodel.jld2"), save_state)
                    counter = 0
                else 
                    counter += 1
                    if counter > count_thresh
                        @printf("Early stopping at epoch: %.f\n", epoch)
                        return true
                    end
                    if counter > 3
                        Optimisers.adjust!(opt_state.original, config.lr/(counter * 2))
                        @printf("No improvment, adjusting learning rate to: %.4f\n", config.lr/(counter * 2))
                    end
                end   
                stime = time()  
            end

            if epoch % config.plot_freq == 0 
                d = vizualize(model, val_loader, ts, θ, st, dev, sample_n, ch; config.kwargs...)
                display(d)
                image_path = joinpath(config.save_path, "img_epoch=$epoch.pdf")
                savefig(image_path)
            end

        end        
        return false
    end

    adtype = Optimization.AutoZygote()
    optf = OptimizationFunction((p, _ , u, y, x) -> loss(p, u, y), adtype)
    optproblem = OptimizationProblem(optf, p)
    result = Optimization.solve(optproblem, ADAMW(config.lr), ncycle(train_loader, config.epochs); callback)
    return model, θ_best
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