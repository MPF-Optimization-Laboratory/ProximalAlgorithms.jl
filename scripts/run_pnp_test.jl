# this is a temporary script for trying out pnp algorithms on images

# run from ProximalAlgorithms.jl folder, e.g.  /home/naomigra/projects/def-mpf/naomigra/2022/pnp_IR/dev/ProximalAlgorithms.jl
include("../src/utilities/denoisers.jl")
include("../src/utilities/image_tools.jl")
include("../src/algorithms/pnp_douglas_rachford.jl")

using LinearAlgebra
using ProximalOperators


function run_example()
    T = Float64
    R = real(T)
    k = 0.7 #how much we want to reduce (scale) max rank of A

    example_index = 10
    model_epoch_number = 20
    model_dir = "../src/utilities/saved_models/MNIST"
    encoder_μ, encoder_logvar, decoder = load_model(model_dir,model_epoch_number)

    function vae_denoiser!(x) 
        x = reconstruct_images(encoder_μ, encoder_logvar, decoder,x)
    end

    g = vae_denoiser!

    test_x, test_y = MNIST.testdata(Float32,example_index)
    m,n = size(test_x)
    A_full_rank = rand(Float32, (m*n, m*n))
    max_rank = Int64(floor(m*n*k))
    F = svd(A_full_rank)

    for i in max_rank:(m*n)
        F.S[i] = 0
    end

    A = F.U * Diagonal(F.S) * F.Vt

    test_x_vec = vectorize_and_flip(test_x)

    b = A*test_x_vec[:]

    f = LeastSquares(A, b)
    lam = R(0.1) * norm(A' * b, Inf)
    x0 = zeros(R, m*n)
    y = similar(x0)
    gamma = R(10) / opnorm(A)^2 #constant parameter for douglas rachford


    #forward operator
    proxf!(xhat,uhat) = prox!(xhat,f,uhat,gamma)
    #backward operator
    denoiser!(yhat,rhat) = prox!(yhat,g,rhat,gamma)

    pnp_dr_iter = PnpDrsIteration(proxf! = proxf!, denoiser! = denoiser!, uhat0 = x0, gamma = gamma)

    for dr_state in Iterators.take(pnp_dr_iter,20)
        println(norm(test_x_vec - pnp_dr_state.x)) 
    end


end