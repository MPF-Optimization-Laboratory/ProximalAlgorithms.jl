
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


Base.@kwdef struct PnpDrsIteration{R,C<:Union{R,Complex{R}},Tx<:AbstractArray{C},F<:Function,G<:Function}
    J_A!::F
    J_B!::G
    u0::Tx #initial point
    #gamma::R #don't think gamma is needed
end


Base.IteratorSize(::Type{<:PnpDrsIteration}) = Base.IsInfinite()

Base.@kwdef struct PnpDrsState{Tx}
    u::Tx
    x::Tx = similar(u) #empty
    y::Tx = similar(u) 
    r::Tx = similar(u)
    res::Tx = similar(u)
end




function Base.iterate(iter::PnpDrsIteration, state::PnpDrsState = PnpDrsState(u=copy(iter.u0)))
    iter.J_B!(state.x,state.u)
    @. state.r = 2 *state.x - state.u
    iter.J_A!(state.y,state.r)
    state.res .= state.y .- state.x
    state.u .+= state.res
    return state, state
end
