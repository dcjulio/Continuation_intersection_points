#using Pkg
#Pkg.add(["MAT","NLsolve","BifurcationKit", "Plots", "LinearAlgebra"])
include("HenonNatalia.jl")
using MAT, NLsolve, BifurcationKit, Plots, LinearAlgebra


####################################################################
function main()
####################################################################
    # load the .mat file
    vars       = matread("xi_17_100arc.mat")
 
    # xi range for continuation and step size
    xi_min, xi_max = 1.5, 2.6 

    # find the intersection curves for each intersection index and store the results in a dictionary
    name_manif      = "Smanif_min"
    branch          = "neg"
    inter_num       = length(vars[name_manif]["inter"][branch]["idx"])
    curves_pmin_neg, pars = InterCurves(vars, name_manif, branch , inter_num, xi_min, xi_max)   

    # plot the intersection curves
    plt = Plots.plot(xlabel = "xi", ylabel = "z", legend = false)
    for inter_idx in 1:inter_num
        haskey(curves_pmin_neg, inter_idx) || continue          # skip seeds that failed
        r = curves_pmin_neg[inter_idx]
        Plots.plot!(plt, r.xi, r.z)
    end
    display(plt)

    return curves_pmin_neg, pars
end
####################################################################
####################################################################

function runContinuation(vars, name_manif, branch, inter_idx, xi_min, xi_max, fixp, pars, xi0)

    # find a guess orbit of the nth intersection with the plane
    guess = build_guess(vars, name_manif, branch, inter_idx)
    hs = norm(guess[1:3] - fixp) # initial distance along the stable manifold
    x0 = [hs; guess[:]]

    # setting up the boundary value problem function
    BVP(x) = PlaneBVP(x, pars, xi0)
    # solve the boundary value problem using Newton's method to find a solution that hits the plane
    sol = nlsolve(BVP, x0, method = :newton, ftol = 1e-12, iterations = 40)
    orbit = sol.zero # the solution orbit = [hs + orbit points]

    # set up the continuation problem to vary xi and find a family of solutions
    BifBVP(x, p) = PlaneBVP(x, pars, p.xi)
    p = (xi = xi0,)

    # only saving the z coordinate of the intersection with the plane and the initial distance hs
    prob = BifurcationProblem(BifBVP, orbit, p, (@optic _.xi), record_from_solution = (x, p; k...) -> (x_end = x[end-2], y_end = x[end-1], z_end = x[end], hs = x[1]))
    
    # continues along the parameter xi in this range [p_min p_max]
    opts = ContinuationPar(p_min = xi_min, p_max = xi_max, dsmin = 1e-5, dsmax = 5e-3, ds = 1e-3, max_steps = 1000, detect_bifurcation = 0, save_sol_every_step = 0)
    # the actual continuation process    
    br = continuation(prob, PALC(), opts; bothside = true)

    return (xi = br.branch.param, x = br.branch.x_end, y = br.branch.y_end, z = br.branch.z_end, hs = br.branch.hs)
end

function build_guess(vars, field, branch, seed_pos)

    S     = vars[field]
    preim = S["points"][branch]["idx_preimages"]

    # Seed, then follow the preimage chain back to the start
    idx = Int[Int(S["inter"][branch]["idx"][seed_pos])]
    while true
        nxt = Int(preim[idx[end]])
        nxt == 0 && break
        push!(idx, nxt)
    end
    reverse!(idx)

    # Gather coordinates
    u = S["uncom"][branch]
    guess = zeros(3, length(idx))
    for (k, i) in enumerate(idx)
        guess[1, k] = u["x"][i]
        guess[2, k] = u["y"][i]
        guess[3, k] = u["z"][i]
    end

    return guess
end

function parameters(vars, field)

    S     = vars[field]
    alpha = S["par"]["a"]
    beta  = S["par"]["b"]
    c     = S["par"]["c"]
    xi    = S["par"]["xi"]
    v0    = S["inter"]["v0"]
    pars = [alpha, beta, c, v0]

    fixp_x = S["per_orbit"]["coord_original"]["x"]
    fixp_y = S["per_orbit"]["coord_original"]["y"]
    fixp_z = S["per_orbit"]["coord_original"]["z"]
    fixp = [fixp_x, fixp_y, fixp_z]

    return fixp, pars, xi
end

function InterCurves(vars, name_manif, branch, inter_num, xi_min, xi_max)   
    
    # find the intersection curves for each intersection index and store the results in a dictionary

    fixp, pars, xi0 = parameters(vars, name_manif)

    inter_curves = Dict{Int, NamedTuple}()
    for inter_idx in 1:inter_num
        try
            inter_curves[inter_idx] = runContinuation(vars, name_manif, branch, inter_idx, xi_min, xi_max, fixp, pars, xi0)
            println("inter index $inter_idx: $(length(inter_curves[inter_idx].xi)) steps")
        catch e
            @warn "inter index $inter_idx failed" exception = e
        end
    end

    return inter_curves, pars
end
####################################################################
####################################################################


pars, inter_curves = main();