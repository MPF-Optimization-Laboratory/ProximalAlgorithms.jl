################################################################################
# Douglas-Rachford Newton-type iterator (with L-BFGS directions)

mutable struct DRNIterator{I <: Integer, R <: Real, T <: BlockArray} <: ProximalAlgorithm{I, T}
    x::T
    f
    g
    gamma::R
    maxit::I
    tol::R
    verbose::I
    verbose_freq::I
    lambda::R
    sigma::R
    tau::R
    FPR_x::T
    DRE_x::R
    H # inverse Jacobian approximation
    d::T # direction
    ref::T # to store reflections
    y::T
    z::T
    xnew::T
    FPR_xnew::T
    res1::T
    res2::T
    res3::T
end

################################################################################
# Constructor

function DRNIterator(x0::T; f=Zero(), g=Zero(), gamma::R=-1.0, maxit::I=10000, tol::R=1e-4, memory=10, verbose=1, verbose_freq=100, lambda=1.0, sigma=1e-4) where {I, R, T}
    if gamma <= 0.0 error("gamma must be positive") end
    x = blockcopy(x0)
    FPR_x = blockcopy(x)
    H = LBFGS(x, memory)
    d = blockcopy(x)
    ref = blockcopy(x)
    y = blockcopy(x)
    z = blockcopy(x)
    xnew = blockcopy(x)
    FPR_xnew = blockcopy(x)
    res1 = blockcopy(x)
    res2 = blockcopy(x)
    res3 = blockcopy(x)
    DRNIterator(x, f, g, gamma, maxit, tol, verbose, verbose_freq, lambda,
        sigma, 0.0, FPR_x, 0.0, H, d, ref, y, z, xnew, FPR_xnew,
        res1, res2, res3)
end

################################################################################
# Utility methods

maxit(sol::DRNIterator) = sol.maxit

converged(sol::DRNIterator, it) = it > 0 && blockmaxabs(sol.FPR_x)/sol.gamma <= sol.tol

verbose(sol::DRNIterator) = sol.verbose > 0
verbose(sol::DRNIterator, it) = sol.verbose > 0 && (sol.verbose == 2 ? true : (it == 1 || it%sol.verbose_freq == 0))

function display(sol::DRNIterator)
    @printf("%6s | %10s | %10s | %10s | %10s |\n ", "it", "gamma", "fpr", "tau", "DRE")
    @printf("------|------------|------------|------------|------------|\n")
end

function display(sol::DRNIterator, it)
    @printf("%6d | %7.4e | %7.4e | %7.4e | %7.4e | \n", it, sol.gamma, blockmaxabs(sol.FPR_x)/sol.gamma, sol.tau, sol.DRE_x)
end

function Base.show(io::IO, sol::DRNIterator)
    println(io, "DRN" )
    println(io, "fpr        : $(blockmaxabs(sol.FPR_x))")
    println(io, "gamma      : $(sol.gamma)")
    println(io, "lambda     : $(sol.lambda)")
    println(io, "tau        : $(sol.tau)")
    print(  io, "DRE        : $(sol.DRE_x)")
end

################################################################################
# Initialization

function initialize!(sol::DRNIterator)

    # reset L-BFGS operator (would be nice to have this option)
    # TODO add function reset!(::LBFGS) in AbstractOperators
    sol.H.currmem, sol.H.curridx = 0, 0
    sol.H.H = 1.0

    f_y = prox!(sol.y, sol.f, sol.x, sol.gamma)
    sol.ref .= 2.*sol.y .- sol.x
    g_z = prox!(sol.z, sol.g, sol.ref, sol.gamma)
    sol.FPR_x .= sol.z .- sol.y

    sol.res1 .= sol.y .- sol.x
    sol.res2 .= sol.ref .- sol.z
    sol.res3 .= sol.z .- sol.y
    sol.DRE_x = f_y + g_z + (0.5/sol.gamma)*(blockvecnorm(sol.res2)^2 - blockvecnorm(sol.res1)^2)

end

################################################################################
# Iteration

function iterate!(sol::DRNIterator{I, R, T}, it::I) where {I, R, T}

    if it > 1
        update!(sol.H, sol.x, sol.xnew, sol.FPR_xnew, sol.FPR_x)
    end
    A_mul_B!(sol.d, sol.H, sol.FPR_x)

    tau = 1.0

    for ls_it = 1:10

        sol.xnew .= sol.x
        sol.xnew .+= (sol.lambda*(1.0-tau)).*sol.FPR_x + tau.*sol.d

        f_y = prox!(sol.y, sol.f, sol.xnew, sol.gamma)
        sol.ref .= 2.*sol.y .- sol.xnew
        g_z = prox!(sol.z, sol.g, sol.ref, sol.gamma)
        sol.FPR_xnew .= sol.z .- sol.y

        sol.res1 .= sol.y .- sol.xnew
        sol.res2 .= sol.ref .- sol.z

        DRE_xnew = f_y + g_z + (0.5/sol.gamma)*(blockvecnorm(sol.res2)^2 - blockvecnorm(sol.res1)^2)

        if DRE_xnew <= sol.DRE_x - sol.sigma*vecnorm(sol.res3)^2
            sol.res3 .= sol.z .- sol.y
            sol.x, sol.xnew = sol.xnew, sol.x
            sol.FPR_x, sol.FPR_xnew = sol.FPR_xnew, sol.FPR_x
            sol.DRE_x = DRE_xnew
            break
        end

        tau = 0.5*tau
    end

    return sol.z

end

################################################################################
# Solver interface(s)

function DRN(x0; kwargs...)
    sol = DRNIterator(x0; kwargs...)
    it, point = run!(sol)
    blockset!(x0, point)
    return (it, point, sol)
end
