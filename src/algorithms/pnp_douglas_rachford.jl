
# Eckstein, Bertsekas, "On the Douglas-Rachford Splitting Method and the
# Proximal Point Algorithm for Maximal Monotone Operators",
# Mathematical Programming, vol. 55, no. 1, pp. 293-318 (1989).

# this is the plug and play version of DR

using Base.Iterators
using ProximalAlgorithms.IterationTools
using ProximalOperators: Zero
using LinearAlgebra
using Printf

Base.@kwdef struct PnpDrsIteration{R,C<:Union{R,Complex{R}},Tx<:AbstractArray{C},F<:Function,G<:Function}
    proxf!::F
    denoiser!::G
    uhat0::Tx #initial point
    gamma::R #don't think gamma is needed
end

Base.@kwdef struct PnpDrsIteration2{R,C<:Union{R,Complex{R}},Tx<:AbstractArray{C},F<:Function,G<:Function}
    proxf!::F
    denoiser!::G
    x0::Tx #initial point, usually 0
    gamma::R #don't think gamma is needed
end


Base.IteratorSize(::Type{<:PnpDrsIteration}) = Base.IsInfinite()

Base.@kwdef struct PnpDrsState{Tx}
    xhat::Tx
    yhat::Tx = similar(xhat) #empty
    rhat::Tx = similar(xhat)
    uhat::Tx = similar(xhat)
    res::Tx = similar(xhat)
end


function Base.iterate(iter::PnpDrsIteration, state::PnpDrsState = PnpDrsState(xhat=copy(iter.uhat0),uhat=copy(iter.uhat0)),res=copy(iter.uhat0))
    iter.proxf!(state.xhat,state.uhat) # xhat^{k+1} = A(uhat^{k+1})
    state.rhat .= 2 .*(state.xhat .- state.uhat)
    iter.denoiser!(state.yhat,state.rhat) # yhat^{k+1} = B(2xhat^{k+1} - uhat^k)
    state.res .= state.xhat .- state.yhat
    state.uhat .+= (state.xhat - state.yhat) #uhat^{k+1} = uhat^k + (xhat^{k+1} - yhat^{k+1})
    return state, state
end


Base.@kwdef struct PnpDrsState2{Tμ}
    μ::Tμ # 
    ν::Tμ = similar(μ)
    ψ::Tμ = similar(μ) # dual var
    res::Tμ = similar(μ)
    x_tilde::Tμ = similar(μ)
    y_tilde::Tμ = similar(μ)
    μ_prev::Tμ = similar(μ)
    ν_prev::Tμ = similar(μ)
    ψ_prev::Tμ = similar(μ)
    x_tilde_prev::Tμ = similar(μ)
    dualres::Tμ = similar(μ)
end


function Base.iterate(iter::PnpDrsIteration2, state::PnpDrsState2 = PnpDrsState2(μ=copy(iter.x0),ν=copy(iter.x0),ψ=copy(iter.x0),x_tilde=copy(iter.x0) ))
    state.μ_prev .= copy(state.μ)
    state.ν_prev .= copy(state.ν)
    state.ψ_prev .= copy(state.ψ)
    state.x_tilde_prev .= copy(state.x_tilde)


    #iter.proxf!(state.μ,state.ψ) # \mu^{k+1} = J_{\alpha \partial \tilde g} (\phi^k)
    iter.denoiser!(state.μ,state.ψ) # \mu^{k+1} = J_{\alpha \partial \tilde g} (\phi^k)

    state.res .= (2 .* state.μ) .- state.ψ # 2 \mu^{k+1} -\phi^k
    
    #iter.denoiser!(state.ν,state.res) # \nu^{k+1} = J_{\alpha \partial \tilde f} (2 \mu^{k+1} -\phi^k)
    iter.proxf!(state.ν,state.res) # \nu^{k+1} = J_{\alpha \partial \tilde f} (2 \mu^{k+1} -\phi^k)

    
    state.ψ .+= (state.ν .- state.μ) #\phi^{k+1} = \phi^k + \nu^{k+1} - \mu^{k+1}

    state.y_tilde .= state.ψ .- state.ν_prev #\tilde x^{k+1} = 
    state.x_tilde .= 2 .* state.y_tilde - state.ν .- state.ψ_prev # \tilde y^{k+1}

    state.dualres .= state.x_tilde_prev .- state.y_tilde # this quantity should go to zero
    return state, state
end