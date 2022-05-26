# this is a temporary script for trying out pnp algorithms on images

import ProximalOperators.prox!
import Random.seed!

# run from ProximalAlgorithms.jl folder, e.g.  /home/naomigra/projects/def-mpf/naomigra/2022/pnp_IR/dev/ProximalAlgorithms.jl
include("../src/utilities/denoisers.jl")
include("../src/utilities/image_tools.jl")
include("../src/algorithms/pnp_douglas_rachford.jl")
include("../src/algorithms/admm.jl")

using LinearAlgebra
using ProximalOperators
using MLDatasets
using Flux
using Zygote
using Random


# need a function to run PnP DR on a single example with various gamma, number of iterations, 

function run_example_1()
    Random.seed!(1)
    T = Float64 #double precision
    R = real(T)
    k = 0.8 # how much we want to reduce (scale) max rank of A

    example_index = 10
    model_epoch_number = 20
    model_dir = "./src/utilities/saved_models/MNIST"
    encoder_μ, encoder_logvar, decoder = load_model(model_dir,model_epoch_number)

    # backward operator (denoiser)
    # TO DO: look into what kind of noise is used for training
    function vae_denoiser!(x,y)
        # defining the denoiser this way so that it has the same format as prox!
        x .= reconstruct_images(encoder_μ, encoder_logvar, decoder,y)
    end

    test_x, test_y = MNIST.testdata(Float32,example_index)
    
    m,n = size(test_x)
    A_full_rank = randn(T, (m*n, m*n)) #randn performs better than rand
    max_rank = Int64(floor(m*n*k))
    F = svd(A_full_rank)

    for i in max_rank:(m*n)
        F.S[i] = 0
    end

    A = (F.U * Diagonal(F.S) * F.Vt)/100

    A = diagm(m*n, m*n, ones(T, m*n))

    test_x_vec = vectorize_and_flip(test_x)
    
    convert_and_save_single_MNIST_image(test_x_vec[:],"./scripts/temp/run_example1","ground_truth")

    #b = A*test_x_vec[:] # add additive noise as well?

    # add noise to x instead of Ax
    b = A*(test_x_vec[:] + randn(T, (length(test_x_vec[:]))))
    
    #b = b + randn(T,Int64(length(b))) #adding noise
    f = LeastSquares(A, b)

    #lam = R(0.1) * norm(A' * b, Inf)
    #z0 = zeros(R, m*n)
    u0 = ones(m*n)*R(0.5) 
    gamma = R(1)


    proxf!(x, u) = prox!(x, f, u, gamma)

    pnp_dr_iter = PnpDrsIteration(J_A! = proxf!, J_B! = vae_denoiser!, u0 = u0)

    # TO DO: find a useful way to look at output
    for (i, pnp_dr_state) in enumerate(Iterators.take(pnp_dr_iter,50))
        println("iteration $i")
        println(norm(pnp_dr_state.res))
        println("\n")
        #convert_and_save_single_MNIST_image(pnp_dr_state.x,"./scripts/temp/run_example1/","iteration_$i")
        resize_and_save_single_MNIST_image(pnp_dr_state.x,"./scripts/temp/run_example1/","iteration_$i")
        #println(norm(pnp_dr_state.res))
        #i+=1
    end


end



function run_example_2()
    T = Float32
    R = real(T)
    k = 0.5 # ratio of rows to columns

    example_index = 11
    model_epoch_number = 20
    model_dir = "./src/utilities/saved_models/MNIST"
    encoder_μ, encoder_logvar, decoder = load_model(model_dir,model_epoch_number)

    # backward operator (denoiser)
    function vae_denoiser!(x,y)
        # defining the denoiser this way so that it has the same format as prox!
        x .= reconstruct_images(encoder_μ, encoder_logvar, decoder,y)
    end

    test_x, test_y = MNIST.testdata(Float32,example_index)
    test_x_vec = vectorize_and_flip(test_x)
    convert_and_save_single_MNIST_image(test_x_vec[:],"./scripts/temp/run_example2","ground_truth")


    m,n = size(test_x)
    A = randn(T, Int64(floor(m*n*k)), (m*n)) # wide matrix
    # b = Ax + noise
    b = A*test_x_vec[:] + randn(T,Int64(floor(m*n*k)) ) 

    f = LeastSquares(A, b)

    #lam = R(0.1) * norm(A' * b, Inf)
    u0 = zeros(R, m*n)
    #x0 = test_x_vec[:]
    gamma = R(10) / opnorm(A)^2 # constant parameter used douglas rachford examples for scaling the l1 norm regularizer
    gamma = R(1)
    println("gamma is", gamma)

    # forward operator
    proxf!(x,u) = prox!(x,f,u,gamma)
    
    pnp_dr_iter = PnpDrsIteration(J_A! = proxf!, J_B! = vae_denoiser!, u0 = u0)

    i = 0
    # TO DO: find a useful way to look at output
    for pnp_dr_state in Iterators.take(pnp_dr_iter,10)
        #convert_and_save_single_MNIST_image(clamp.(pnp_dr_state.xhat,0.0,1.0),"./scripts/temp/run_example2/","iteration_$i")
        resize_and_save_single_MNIST_image(pnp_dr_state.x,"./scripts/temp/run_example2/","iteration_$i")
        println(norm(pnp_dr_state.res))
        #println(norm(test_x_vec[:] - pnp_dr_state.xhat))
        #println(norm(pnp_dr_state.xhat))
        i+=1
    end


end


# in this example we use the identity as a denoiser
function run_example_3()
    T = Float32
    R = real(T)
    k = 0.5 # ratio of rows to columns

    example_index = 11
    model_epoch_number = 20
    model_dir = "./src/utilities/saved_models/MNIST"
    encoder_μ, encoder_logvar, decoder = load_model(model_dir,model_epoch_number)

    # backward operator (denoiser)
    #function vae_denoiser!(x,y)
    function identity_denoiser!(x,y)
        # defining the denoiser this way so that it has the same format as prox!
        #x .= reconstruct_images(encoder_μ, encoder_logvar, decoder,y)
        x.=y
    end

    test_x, test_y = MNIST.testdata(Float32,example_index)
    test_x_vec = vectorize_and_flip(test_x)
    convert_and_save_single_MNIST_image(test_x_vec[:],"./scripts/temp/run_example3","ground_truth")


    m,n = size(test_x)
    A = randn(T, Int64(floor(m*n*k)), (m*n)) # wide matrix
    b = A*test_x_vec[:] + randn(T,Int64(floor(m*n*k)) ) # TO DO: values of b are outside (0,1) range, fix this

    # convert_and_save_single_MNIST_image(clamp.(b,0,1),"./scripts/temp/","observed")

    f = LeastSquares(A, b)

    #lam = R(0.1) * norm(A' * b, Inf)
    u0 = zeros(R, m*n)
    #gamma = R(10) / opnorm(A)^2 # constant parameter for douglas rachford using g is the l1 norm
    gamma = 1.0f0
    println("\n gamma is $gamma \n")

    # forward operator
    proxf!(x,u) = prox!(x,f,u,gamma)
    
    pnp_dr_iter = PnpDrsIteration(J_A! = proxf!, J_B! = identity_denoiser!, u0 = u0)

    i = 0
    for pnp_dr_state in Iterators.take(pnp_dr_iter,50)
        resize_and_save_single_MNIST_image(pnp_dr_state.x,"./scripts/temp/run_example3/","iteration_$i")
        i+=1
    end


end


# temporary function for trying the optimized decoder denoiser
function run_denoising_example()
    Random.seed!(1)
    example_index = 11
    test_x, test_y = MNIST.testdata(Float32,example_index)

    test_x_vec = vectorize_and_flip(test_x)[:]
    resize_and_save_single_MNIST_image(test_x_vec,"./scripts/temp/run_denoising","clean_img")

    model_epoch_number = 20
    model_dir = "./src/utilities/saved_models/MNIST"
    encoder_μ, encoder_logvar, decoder = load_model(model_dir,model_epoch_number)
    
    
    z0 = add_additive_noise(ones(20)) #initial point for gradient descent

    #z0 = zeros(20)
    #alternate starting point
    #μ = encoder_μ(test_x_vec)
    #logvar = encoder_logvar(test_x_vec)
    #z0 = μ + randn(Float32, size(logvar)) .* exp.(0.5f0 * logvar)

    y_noisy = add_additive_noise(test_x_vec,0.2) # vectorized noisy image

    #shift and rescale to have values between 0 and 1
    y_temp = (y_noisy .- min(0, minimum(y_noisy)))
    y_noisy .= y_temp / maximum(y_temp)

    #y_noisy = test_x_vec #trying without noise

    resize_and_save_single_MNIST_image(y_noisy,"./scripts/temp/run_denoising","before_denoising")


    loss_function = decoder_loss_function
    
    num_iter = 1000 #number of iterations of gradient descent
    z_out = gradient_descent(loss_function,z0,decoder,y_noisy,num_iter,0.05)
    
    x_out = decoder(z_out)
    resize_and_save_single_MNIST_image(x_out,"./scripts/temp/run_denoising","after_denoising")

end

#examples 4 and 5 are for studying convergence
function run_example_4()
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

    f = LeastSquares(A, b)
    g = NormL1(lam)

    u0 = zeros(R, n) # initial point of zero

    gamma=R(10) / opnorm(A)^2
    gamma = R(1) # gamma doesn't currently do anything in the algorithm!

    proxf!(x,z) = prox!(x,f,z,gamma) 
    denoiser!(y,r) = prox!(y,g,r,gamma)

    pnp_dr_iter = PnpDrsIteration(J_A! = proxf!, J_B! = denoiser!, u0 = u0)

    
    for pnp_dr_state in Iterators.take(pnp_dr_iter, 30)
        println(norm(pnp_dr_state.res))
    end

end


struct NegativeConjugate{T}
    c::Conjugate{T}
end


function prox!(y, nc::NegativeConjugate, x, gamma)
    prox!(y,nc.c,-x,gamma)
end



# trying ADMM in place of Douglas Rachford
function run_example_5()
    
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
    
    #lam = R(0.1) * norm(A' * b, Inf)
    lam = R(1)

    f = LeastSquares(A, b)
    g = NormL1(lam)

    fdual = NegativeConjugate(Conjugate(f))
    gdual = Conjugate(g)

    u0 = zeros(R, n)

    #gamma=R(10) / opnorm(A)^2
    #gamma= R(10) 
    gamma = R(1)

    proxf!(x,u) = prox!(x,f,u,gamma) 
    #proxf!(x,u) = prox!(x,fdual,u,gamma) #apply DRS to dual problem 

    denoiser!(x,u) = prox!(x,g,u,gamma)
    #denoiser!(x,u) = prox!(x,gdual,u,gamma) #apply DRS to dual problem

    #dr_iter = DouglasRachfordIteration(f=f, g=g, x0=x0, gamma=gamma)
    admm_iter = AdmmIteration(f = f, g = g, x0 = u0, gamma = gamma)
    pnp_dr_iter = PnpDrsIteration(J_A! = proxf!, J_B! = denoiser!, u0 = u0)
    
    for (admm_state, pnp_dr_state) in Iterators.take(zip(admm_iter, pnp_dr_iter), 30)
        #println(norm(admm_state.u)) # converges to 2.072
        println(norm(admm_state.x - admm_state.y)) #converges to zero
        println(norm(pnp_dr_state.res)) #converges to zero
        println("\n")
    end



end


function run_example_6()
    Random.seed!(2)
    T = Float64 #single precision
    R = real(T)

    # take an example image from the test data set
    example_index = 10
    test_x, test_y = MNIST.testdata(Float32,example_index)

    # vectorize image and save ground truth for comparison
    test_x_vec = vectorize_and_flip(test_x)[:]
    resize_and_save_single_MNIST_image(test_x_vec,"./scripts/temp/run_example6","ground_truth")

    # load the VAE model
    model_epoch_number = 20
    model_dir = "./src/utilities/saved_models/MNIST"
    encoder_μ, encoder_logvar, decoder = load_model(model_dir,model_epoch_number)

    test_x, test_y = MNIST.testdata(Float32,example_index)
    
    # generate a random low rank matrix
    k = 0.8 # how much we want to reduce (scale) max rank of A
    m,n = size(test_x)
    A_full_rank = randn(T, (m*n, m*n)) #randn performs better than rand
    max_rank = Int64(floor(m*n*k))
    F = svd(A_full_rank)

    for i in max_rank:(m*n)
        F.S[i] = 0
    end
    A = (F.U * Diagonal(F.S) * F.Vt)
    A = diagm(m*n, m*n, ones(T, m*n))

    # add noise to x instead of Ax
    #b = A*(test_x_vec[:] + randn(T, (length(test_x_vec[:]))))
    b = A*(test_x_vec[:])
    b = b + randn(T,Int64(length(b))) #adding noise

    f = LeastSquares(A, b)
    #z0 = zeros(R, m*n)
    u0 = ones(m*n)*R(0.5) 
    gamma = R(1)
    

    #function proxf!(x,z)
    #    prox!(x, f, z, gamma)
    #    x1 = (x .- min(0, minimum(x)))
    #    x .= x1 / maximum(x1)
    #end

    proxf!(x, u) = prox!(x, f, u, gamma)

    num_iter = 20 # number of iterations of GD in the denoiser
    denoiser!(x, u) = decoder_denoiser!(x,u,encoder_μ, encoder_logvar, decoder, num_iter)


    pnp_dr_iter = PnpDrsIteration(J_A! = proxf!, J_B! = denoiser!, u0 = u0)

    for (i, pnp_dr_state) in enumerate(Iterators.take(pnp_dr_iter,30))
        println("at PnP iteration $i")
        println("the residual value is $(norm(pnp_dr_state.res)) \n")
        #println("an intermediate value is $(norm(pnp_dr_state.r)) \n")
        #println("norm of y iterate is $(norm(pnp_dr_state.y)) \n")
        println("Saving output...")
        #convert_and_save_single_MNIST_image(pnp_dr_state.x,"./scripts/temp/run_example6/","iteration_$i")
        resize_and_save_single_MNIST_image(pnp_dr_state.x,"./scripts/temp/run_example6/","iteration_$i")
        #println(norm(pnp_dr_state.res))
        #i+=1
    end


end