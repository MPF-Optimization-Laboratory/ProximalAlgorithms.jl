# this is a temporary script for trying out pnp algorithms on images

# run from ProximalAlgorithms.jl folder, e.g.  /home/naomigra/projects/def-mpf/naomigra/2022/pnp_IR/dev/ProximalAlgorithms.jl
include("../src/utilities/denoisers.jl")
include("../src/utilities/image_tools.jl")
include("../src/algorithms/pnp_douglas_rachford.jl")

using LinearAlgebra
using ProximalOperators


function run_example()
    println(pwd())
    T = Float32
    R = real(T)
    k = 0.7 #how much we want to reduce (scale) max rank of A

    example_index = 10
    model_epoch_number = 20
    model_dir = "./src/utilities/saved_models/MNIST"
    encoder_μ, encoder_logvar, decoder = load_model(model_dir,model_epoch_number)

    # backward operator (denoiser)
    function vae_denoiser!(x,y)
        # defining the denoiser this way so that it has the same format as prox!
        x .= reconstruct_images(encoder_μ, encoder_logvar, decoder,y)
    end

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

    convert_and_save_single_MNIST_image(test_x_vec[:],"./scripts/temp/","ground_truth")


    b = A*test_x_vec[:] # TO DO: values of b are outside (0,1) range, fix this

    # convert_and_save_single_MNIST_image(clamp.(b,0,1),"./scripts/temp/","observed")

    f = LeastSquares(A, b)

    lam = R(0.1) * norm(A' * b, Inf)
    #x0 = zeros(R, m*n)
    x0 = test_x_vec[:]
    y = similar(x0)
    gamma = R(10) / opnorm(A)^2 # constant parameter for douglas rachford
    gamma = 1.0f0

    # forward operator
    proxf!(xhat,uhat) = prox!(xhat,f,uhat,gamma)
    
    pnp_dr_iter = PnpDrsIteration(proxf! = proxf!, denoiser! = vae_denoiser!, uhat0 = x0, gamma = gamma)

    i = 0
    # TO DO: find a useful way to look at output
    for pnp_dr_state in Iterators.take(pnp_dr_iter,50)
        convert_and_save_single_MNIST_image(pnp_dr_state.xhat,"./scripts/temp/","iteration_$i")
        #println(norm(test_x_vec - pnp_dr_state.xhat))
        println(norm(pnp_dr_state.xhat))
        i+=1
    end


end