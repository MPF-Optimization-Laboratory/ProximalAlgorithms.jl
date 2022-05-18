
# Eckstein, Bertsekas, "On the Douglas-Rachford Splitting Method and the
# Proximal Point Algorithm for Maximal Monotone Operators",
# Mathematical Programming, vol. 55, no. 1, pp. 293-318 (1989).

# this is the plug and play version of DR
# Definition of the DR recursion from Eckstein, Bertsekas is:
# z^{k+1} = J_{\gamma A}( (2J_{\gamma B} -I)z^k ) + (I - J_{\gamma B})z^k 

# This unravels to
# x^{k+1} = J_{\gamma B} z^k 
# y^{k+1} = J_{\gamma A}( 2x^{k+1} -z^k )
# z^{k+1} = z^k + y^{k+1} - x^{k+1}

using Base.Iterators
using ProximalAlgorithms.IterationTools
using ProximalOperators: Zero
using LinearAlgebra
using Printf

#Base.@kwdef struct PnpDrsIteration{R,C<:Union{R,Complex{R}},Tx<:AbstractArray{C},F<:Function,G<:Function}
#    proxf!::F
#    denoiser!::G
#    uhat0::Tx #initial point
#    gamma::R #don't think gamma is needed
#end

Base.@kwdef struct PnpDrsIteration{R,C<:Union{R,Complex{R}},Tx<:AbstractArray{C},F<:Function,G<:Function}
    J_A!::F
    J_B!::G
    z0::Tx #initial point
    #gamma::R #don't think gamma is needed
end

#Base.@kwdef struct PnpDrsIteration2{R,C<:Union{R,Complex{R}},Tx<:AbstractArray{C},F<:Function,G<:Function}
#    proxf!::F
#    denoiser!::G
#    x0::Tx #initial point, usually 0
#    gamma::R #don't think gamma is needed
#end


Base.IteratorSize(::Type{<:PnpDrsIteration}) = Base.IsInfinite()

#Base.@kwdef struct PnpDrsState{Tx}
#    xhat::Tx
#    yhat::Tx = similar(xhat) #empty
#    rhat::Tx = similar(xhat)
#    uhat::Tx = similar(xhat)
#    res::Tx = similar(xhat)
#end

Base.@kwdef struct PnpDrsState{Tx}
    z::Tx
    x::Tx = similar(z) #empty
    y::Tx = similar(z) 
    r::Tx = similar(z)
    res::Tx = similar(z)
end

#function Base.iterate(iter::PnpDrsIteration, state::PnpDrsState = PnpDrsState(xhat=copy(iter.uhat0),uhat=copy(iter.uhat0)),res=copy(iter.uhat0))
#    iter.proxf!(state.xhat,state.uhat) # xhat^{k+1} = A(uhat^{k+1})
#    state.rhat .= 2 .*(state.xhat .- state.uhat)
#    iter.denoiser!(state.yhat,state.rhat) # yhat^{k+1} = B(2xhat^{k+1} - uhat^k)
#    state.res .= state.xhat .- state.yhat
#    state.uhat .+= (state.xhat - state.yhat) #uhat^{k+1} = uhat^k + (xhat^{k+1} - yhat^{k+1})
#    return state, state
#end


function Base.iterate(iter::PnpDrsIteration, state::PnpDrsState = PnpDrsState(z=copy(iter.z0)))
    iter.J_B!(state.x,state.z)
    state.r .= 2 .*state.x .- state.z
    iter.J_A!(state.y,state.r)
    state.res .= state.y .- state.x
    state.z .+= state.res
    return state, state
end


#Base.@kwdef struct PnpDrsState2{Tμ}
#    μ::Tμ # 
#    ν::Tμ = similar(μ)
#    ψ::Tμ = similar(μ) # dual var
#    res::Tμ = similar(μ)
#end


#function Base.iterate(iter::PnpDrsIteration2, state::PnpDrsState2 = PnpDrsState2(μ=copy(iter.x0),ν=copy(iter.x0),ψ=copy(iter.x0),x_tilde=copy(iter.x0) ))
#    iter.denoiser!(state.μ,state.ψ) # \mu^{k+1} = J_{\alpha \partial \tilde g} (\phi^k)
#    state.res .= (2 .* state.μ) .- state.ψ # 2 \mu^{k+1} -\phi^k
#    iter.proxf!(state.ν,state.res) # \nu^{k+1} = J_{\alpha \partial \tilde f} (2 \mu^{k+1} -\phi^k)
#    state.ψ .+= (state.ν .- state.μ) #\phi^{k+1} = \phi^k + \nu^{k+1} - \mu^{k+1}
#    return state, state
#end