# this is a temporary script for trying out pnp algorithms on images


import ProximalOperators.prox!

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


function run_example_1()
    T = Float32
    R = real(T)
    k = 0.8 # how much we want to reduce (scale) max rank of A

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

    A = (F.U * Diagonal(F.S) * F.Vt)/100
    test_x_vec = vectorize_and_flip(test_x)
    
    convert_and_save_single_MNIST_image(test_x_vec[:],"./scripts/temp/run_example1","ground_truth")

    b = A*test_x_vec[:] # add additive noise as well?
    
    b = b + randn(T,Int64(length(b))) #adding noise
    f = LeastSquares(A, b)

    #lam = R(0.1) * norm(A' * b, Inf)
    x0 = zeros(R, m*n)
    #x0 = test_x_vec[:]
    #gamma = R(10) / opnorm(A)^2 # constant parameter for douglas rachford
    #gamma = 1.0f0
    gamma = R(0.1) #trying different values of gamma
    # forward operator
    proxf!(x,z) = prox!(x,f,z,gamma)
    
    pnp_dr_iter = PnpDrsIteration(J_A! = proxf!, J_B! = vae_denoiser!, z0 = x0)

    i = 0
    # TO DO: find a useful way to look at output
    for pnp_dr_state in Iterators.take(pnp_dr_iter,200)
        println("iteration $i")
        println(norm(pnp_dr_state.res))
        println("\n")
        #convert_and_save_single_MNIST_image(pnp_dr_state.x,"./scripts/temp/run_example1/","iteration_$i")
        
        resize_and_save_single_MNIST_image(pnp_dr_state.x,"./scripts/temp/run_example1/","iteration_$i")
        #println(norm(pnp_dr_state.res))
        i+=1
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

    # convert_and_save_single_MNIST_image(clamp.(b,0,1),"./scripts/temp/","observed")

    f = LeastSquares(A, b)

    #lam = R(0.1) * norm(A' * b, Inf)
    x0 = zeros(R, m*n)
    #x0 = test_x_vec[:]
    #y = similar(x0)
    gamma = R(10) / opnorm(A)^2 # constant parameter for douglas rachford using g is the l1 norm
    #gamma = 1.0f0
    println("gamma is", gamma)

    # forward operator
    proxf!(xhat,uhat) = prox!(xhat,f,uhat,gamma)
    
    pnp_dr_iter = PnpDrsIteration(proxf! = proxf!, denoiser! = vae_denoiser!, uhat0 = x0, gamma = gamma)

    i = 0
    # TO DO: find a useful way to look at output
    for pnp_dr_state in Iterators.take(pnp_dr_iter,10)
        #convert_and_save_single_MNIST_image(clamp.(pnp_dr_state.xhat,0.0,1.0),"./scripts/temp/run_example2/","iteration_$i")
        resize_and_save_single_MNIST_image(clamp.(pnp_dr_state.xhat,0.0,1.0),"./scripts/temp/run_example2/","iteration_$i")
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
    x0 = zeros(R, m*n)
    #x0 = test_x_vec[:]
    #y = similar(x0)
    #gamma = R(10) / opnorm(A)^2 # constant parameter for douglas rachford using g is the l1 norm
    gamma = 1.0f0
    println("gamma is", gamma)

    # forward operator
    proxf!(xhat,uhat) = prox!(xhat,f,uhat,gamma)
    
    pnp_dr_iter = PnpDrsIteration(proxf! = proxf!, denoiser! = identity_denoiser!, uhat0 = x0, gamma = gamma)

    i = 0
    # TO DO: find a useful way to look at output
    for pnp_dr_state in Iterators.take(pnp_dr_iter,50)
        #convert_and_save_single_MNIST_image(clamp.(pnp_dr_state.xhat,0.0,1.0),"./scripts/temp/run_example2/","iteration_$i")
        resize_and_save_single_MNIST_image(clamp.(pnp_dr_state.xhat,0.0,1.0),"./scripts/temp/run_example3/","iteration_$i")
        println(norm(test_x_vec[:] - pnp_dr_state.xhat))
        i+=1
    end


end


# temporary function for trying the optimized decoder denoiser
function run_denoising_example()
    example_index = 11
    test_x, test_y = MNIST.testdata(Float32,example_index)

    test_x_vec = vectorize_and_flip(test_x)[:]
    resize_and_save_single_MNIST_image(test_x_vec,"./scripts/temp/run_denoising","before_denoising")


    model_epoch_number = 20
    model_dir = "./src/utilities/saved_models/MNIST"
    encoder_μ, encoder_logvar, decoder = load_model(model_dir,model_epoch_number)
    
    z0 = add_additive_noise(ones(20)) #initial point for gradient descent
    y_noisy = add_additive_noise(test_x_vec) # vectorized noisy image
    loss_function = decoder_loss_function
    
    num_iter = 40 #number of iterations of gradient descent
    z_out = gradient_descent(loss_function,z0,decoder,y_noisy,num_iter)
    
    x_out = decoder(z_out)
    resize_and_save_single_MNIST_image(x_out,"./scripts/temp/run_denoising","after_denoising_$num_iter")

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

    z0 = zeros(R, n) # initial point of zero

    gamma=R(10) / opnorm(A)^2
    gamma = R(1) # gamma doesn't currently do anything in the algorithm!

    proxf!(x,z) = prox!(x,f,z,gamma) 
    denoiser!(y,r) = prox!(y,g,r,gamma)

    #dr_iter = DouglasRachfordIteration(f=f, g=g, x0=x0, gamma=gamma)

    pnp_dr_iter = PnpDrsIteration(J_A! = proxf!, J_B! = denoiser!, z0 = z0)

    #pnp_dr_iter2 = PnpDrsIteration2(proxf! = proxf!, denoiser! = denoiser!, x0 = x0, gamma = gamma)

    
    for pnp_dr_state in Iterators.take(pnp_dr_iter, 30)
        println(norm(pnp_dr_state.res))
    end

    #for pnp_dr_state in Iterators.take(pnp_dr_iter2, 10)
    #    println(norm(pnp_dr_state.dualres)) #should go to zero, stalls around 1.4578023
    #    println(norm(pnp_dr_state.x_tilde_prev))
        #println(norm(pnp_dr_state.μ - pnp_dr_state.ν)) # this should go to zero according to BC
    #end
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
    #lam = R(0.5) #what role does lambda play?

    lam = R(1)

    f = LeastSquares(A, b)
    g = NormL1(lam)

    fdual = NegativeConjugate(Conjugate(f))

    gdual = Conjugate(g)

    x0 = zeros(R, n)
    u0 = zeros(R, n)

    #gamma=R(10) / opnorm(A)^2

    #gamma= R(10) bigger gamma seems to make dualres bigger

    gamma = R(1)

    #proxf!(xhat,uhat) = prox!(xhat,f,uhat,gamma) #apply DRS to dual problem 
    proxf!(out,in) = prox!(out,fdual,in,gamma) 

    denoiser!(out,in) = prox!(out,g,in,gamma)
    #denoiser!(out,in) = prox!(out,gdual,in,gamma) #apply DRS to dual problem

    #proxf!(xhat,uhat) = prox!(xhat,f,uhat,gamma) #in place prox updates xhat
    #denoiser!(yhat,rhat) = prox!(yhat,g,rhat,gamma) #rhat^{k+1} = 2xhat^{k+1} - uhat^{k}

    #dr_iter = DouglasRachfordIteration(f=f, g=g, x0=x0, gamma=gamma)
    admm_iter = AdmmIteration(f = f, g = g, x0 = u0, gamma = gamma)
    
    pnp_dr_iter2 = PnpDrsIteration2(proxf! = proxf!, denoiser! = denoiser!, x0 = x0, gamma = gamma)
    
    i = 0

    for (admm_state, pnp_dr_state) in Iterators.take(zip(admm_iter, pnp_dr_iter2), 1000)
    #for admm_state in Iterators.take(admm_iter, 5)
        #append!(dualres_vector,admm_state.dualres)
        #println(norm(admm_state.u)) # converges to 2.072
        #println(norm(admm_state.x - admm_state.y))
        #println("\n")
        println(norm(pnp_dr_state.μ - pnp_dr_state.ν)) #converges to zero
        #println(norm(pnp_dr_state.ψ))
        #println(norm(pnp_dr_state.ν))

        #println(norm(admm_state.x))
        #println(norm(pnp_dr_state.μ))
        #println("\n")
        #println(norm(admm_state.y))
        #println(norm(admm_state.y - admm_state.x))
        #println(norm(admm_state.dualres))
        #println(f(admm_state.x) + g(admm_state.y))
    end

    #println("\n")

    #for pnp_dr_state in Iterators.take(pnp_dr_iter2, 10)
        #println(norm(pnp_dr_state.μ)) #not quite converging, alternating between 1.04 and 1.03
        #println(norm(pnp_dr_state.ν))
        #println(norm(pnp_dr_state.ψ))
        #println(norm(pnp_dr_state.x_tilde))
        #println(norm(pnp_dr_state.y_tilde))

        #println(norm(pnp_dr_state.ψ - pnp_dr_state.ν))


        #println(norm(pnp_dr_state.dualres)) #should go to zero, stalls around 1.4578023
        #println(norm(pnp_dr_state.y_tilde))
        #println(norm(pnp_dr_state.x_tilde))
        #println(norm(pnp_dr_state.μ - pnp_dr_state.ν)) # this value tends to 0, as it should!
        #println(norm(pnp_dr_state.ψ - pnp_dr_state.ψ_prev)) #this value also tends to 0

    #end

end