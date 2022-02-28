
# Eckstein, Bertsekas, "On the Douglas-Rachford Splitting Method and the
# Proximal Point Algorithm for Maximal Monotone Operators",
# Mathematical Programming, vol. 55, no. 1, pp. 293-318 (1989).


using Base.Iterators
using ProximalAlgorithms.IterationTools
using ProximalOperators: Zero
using LinearAlgebra
using Printf

Base.@kwdef struct PnpDrsIteration{R,C<:Union{R,Complex{R}},Tx<:AbstractArray{C},Tf,Tg}
    proxf!::Tf = Zero()
    denoiser!::Tg = Zero()
    uhat0::Tx #uhat0 takes the place of x0
    gamma::R #don't think gamma is needed
end

Base.IteratorSize(::Type{<:PnpDrsIteration}) = Base.IsInfinite()

Base.@kwdef struct PnpDrsState{Tx}
    xhat::Tx
    yhat::Tx = similar(x)
    rhat::Tx = similar(x)
    uhat::Tx = similar(x)
    #res::Tx = similar(x)
end


function Base.iterate(iter::PnpDrsIteration, state::PnpDrsState = PnpDrsState(x=copy(iter.x0)))
    #prox!(state.y, iter.f, state.x, iter.gamma)
    iter.proxf!(state.xhat,state.uhat) # xhat^{k+1} = A(uhat^{k+1})
    state.rhat .= 2 .* state.xhat .- state.uhat
    #prox!(state.z, iter.g, state.r, iter.gamma)
    iter.denoiser!(state.yhat,state.rhat) # yhat^{k+1} = B(2xhat^{k+1} - uhat^k)
    #state.res .= state.x .- state.y
    #state.x .-= state.res
    state.uhat .-= (state.xhat - state.yhat) #uhat^{k+1} = uhat^k - (xhat^{k+1} - yhat^{k+1})
    return state, state
end
