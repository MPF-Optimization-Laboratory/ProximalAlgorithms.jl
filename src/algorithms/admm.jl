
# ADMM

# made this to compare convergence with douglas rachford

# to be tested

using Base.Iterators
using ProximalAlgorithms.IterationTools
using ProximalOperators: Zero
using LinearAlgebra
using Printf

#Base.@kwdef struct AdmmIteration{R,C<:Union{R,Complex{R}},Tx<:AbstractArray{C},F<:Function,G<:Function}
    #proxf!::F
#    f::F
#    g::G
#    x0::Tx #initial point
#    gamma::R #don't think gamma is needed
#end

Base.@kwdef struct AdmmIteration{R,C<:Union{R,Complex{R}},Tx<:AbstractArray{C},Tf,Tg}
    f::Tf = Zero()
    g::Tg = Zero()
    x0::Tx
    gamma::R   
end



Base.IteratorSize(::Type{<:AdmmIteration}) = Base.IsInfinite()

Base.@kwdef struct AdmmState{Tx}
    x::Tx
    y::Tx = similar(x) #empty
    u::Tx = similar(x)
    dualres::Tx = similar(x)
end


function Base.iterate(iter::AdmmIteration, state::AdmmState = AdmmState(x=copy(iter.x0),y=copy(iter.x0),u=copy(iter.x0)),dualres=copy(iter.x0))
    prox!(state.x,iter.f,(state.y .- state.u), iter.gamma)
    prox!(state.y,iter.g, (state.x .+ state.u), iter.gamma)
    state.dualres .= state.x .- state.y
    state.u .+= state.dualres
    #iter.proxf!(state.xhat,state.uhat) # xhat^{k+1} = A(uhat^{k+1})
    #state.rhat .= 2 .*(state.xhat .- state.uhat)
    #iter.denoiser!(state.yhat,state.rhat) # yhat^{k+1} = B(2xhat^{k+1} - uhat^k)
    #state.res .= state.xhat .- state.yhat
    #state.uhat .+= (state.xhat - state.yhat) #uhat^{k+1} = uhat^k + (xhat^{k+1} - yhat^{k+1})
    return state, state
end





