### A Pluto.jl notebook ###
# v0.18.1

using Markdown
using InteractiveUtils

# ╔═╡ 319ae2a8-b501-4078-b183-4832364c02fa
begin
	using Pkg
	Pkg.add("LinearAlgebra")
	Pkg.add("ProximalOperators")
	Pkg.add("FileIO")
	Pkg.add("ImageIO")
	Pkg.add("CUDA")
	Pkg.add("Flux")
	Pkg.add("Zygote")
	Pkg.add(url="https://github.com/JuliaTomo/TomoForward.jl")
	Pkg.add(url="https://github.com/JuliaTomo/DeepReconstruction.jl")
end

# ╔═╡ b94b20bc-9250-4889-83f1-eb5daab12e9b
begin
	using LinearAlgebra
	using ProximalOperators
	using Base.Iterators
	using Printf
	using CUDA, Flux, Zygote
	using TomoForward, DeepReconstruction
	using FileIO, ImageIO
end

# ╔═╡ 03d3ac83-1be7-4ab4-892f-091cbd758c95
md"""
This is a Pluto notebook for trying things out
"""

# ╔═╡ 8520fa00-9b21-11ec-08bd-a1067d39ea44
# import packages
#begin
#	using LinearAlgebra
#	using ProximalOperators
#	using Base.Iterators
#	using Printf
#end

# ╔═╡ d0ef7765-c56c-41a4-9087-88cde949fc5d
# set variables for testing
begin
	T = Float32
	A = T[
        1.0 -2.0 3.0 -4.0 5.0
        2.0 -1.0 0.0 -1.0 3.0
        -1.0 0.0 4.0 -3.0 2.0
        -1.0 -1.0 -1.0 1.0 3.0
    ]
	b = T[1.0, 2.0, 3.0, 4.0]
	m, n = size(A)
	R = real(T)
	lam = R(0.1) * norm(A' * b, Inf)
	x0 = zeros(R, n)
	y0 = similar(x0)
	f = LeastSquares(A, b)
	g = NormL1(lam)
	gamma=R(10) / opnorm(A)^2
	proxf!(xhat,uhat) = prox!(xhat,f,uhat,gamma)
	denoiser!(yhat,rhat) = prox!(yhat,g,rhat,gamma)
end

# ╔═╡ 6e85550d-349a-48eb-802c-b3809db16dd2
begin
	# define the pnp-DRS interation
	Base.@kwdef struct PnpDrsIteration{R,C<:Union{R,Complex{R}},Tx<:AbstractArray{C},F<:Function,G<:Function}
	    proxf!::F
	    denoiser!::G
	    uhat0::Tx #uhat0 takes the place of x0
	    gamma::R #don't think gamma is needed
	end
	
	Base.IteratorSize(::Type{<:PnpDrsIteration}) = Base.IsInfinite()
	
	
	# define the pnp-DRS state
	Base.@kwdef struct PnpDrsState{Tx}
	    xhat::Tx
	    yhat::Tx = similar(xhat)
	    rhat::Tx = similar(xhat)
	    uhat::Tx = similar(xhat)
	end
	
	function Base.iterate(iter::PnpDrsIteration, state::PnpDrsState = PnpDrsState(xhat=copy(iter.uhat0)))
	    iter.proxf!(state.xhat,state.uhat) # xhat^{k+1} = A(uhat^{k+1})
	    state.rhat .= 2 .* state.xhat .- state.uhat
	    iter.denoiser!(state.yhat,state.rhat) # yhat^{k+1} = B(2xhat^{k+1} - uhat^k)
	    state.uhat .-= (state.xhat - state.yhat) #uhat^{k+1} = uhat^k - (xhat^{k+1} - yhat^{k+1})
	    return state, state
	end
end

# ╔═╡ 32c39089-40cf-49ce-bb20-5efbf759a0b5
#example: create an instance
pnp_dr_iter = PnpDrsIteration(proxf! = proxf!, denoiser! = denoiser!,uhat0=x0,gamma=gamma)

# ╔═╡ 22027231-0c43-4bd1-a21a-4b9f4a333568
begin
	img = Float32.(load("2foam30.png"))
	img_gt = copy(img)
end

# ╔═╡ ab0460df-6e85-4cc4-a4b1-3a37dbc05f59
begin
	H, W = size(img)
	detcount = H
	nangles = 30
	proj_geom = ProjGeom(1.0, detcount, LinRange(0,pi,nangles+1)[1:nangles])
end

# ╔═╡ fd079dc5-403e-4a97-af12-60c489ad03f2
begin
	B = fp_op_parallel2d_line(proj_geom, size(img, 1), size(img, 2))
	p_data = Float32.(B * vec(img))
	# p_data = vec(reshape(p_data, nangles, :)') # detector count should be the first axis
	lr = 0.0001f0
end

# ╔═╡ 7607323c-a19e-48ab-9ec9-8a974c9a22e9
begin
	net = UNet(3; use_two_skip=true) # 128, 5
	opt = ADAM(lr) # Gadelha
	dresult = "../result/"
	mkpath(dresult)
	u_out, losses, u_best, errs = recon2d_dip(net, opt, p_data, Float32.(B), H, W; img_gt=img_gt)
end

# ╔═╡ c31ecec5-345a-4f3d-8653-410c908ca4bd
typeof(Float32.(B))

# ╔═╡ Cell order:
# ╟─03d3ac83-1be7-4ab4-892f-091cbd758c95
# ╠═319ae2a8-b501-4078-b183-4832364c02fa
# ╠═b94b20bc-9250-4889-83f1-eb5daab12e9b
# ╠═8520fa00-9b21-11ec-08bd-a1067d39ea44
# ╠═d0ef7765-c56c-41a4-9087-88cde949fc5d
# ╠═6e85550d-349a-48eb-802c-b3809db16dd2
# ╠═32c39089-40cf-49ce-bb20-5efbf759a0b5
# ╠═22027231-0c43-4bd1-a21a-4b9f4a333568
# ╠═ab0460df-6e85-4cc4-a4b1-3a37dbc05f59
# ╠═fd079dc5-403e-4a97-af12-60c489ad03f2
# ╠═7607323c-a19e-48ab-9ec9-8a974c9a22e9
# ╠═c31ecec5-345a-4f3d-8653-410c908ca4bd
