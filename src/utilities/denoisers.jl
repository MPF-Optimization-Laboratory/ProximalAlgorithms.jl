#=
This is where I create denoising functions for the plug and play methods
=#

using BSON: @load
using MLDatasets: FashionMNIST
using NNlib
using Flux
using Zygote
using LinearAlgebra




#=
Here I copy Babhru's VAE denoiser
=#

# This function determines the architecture
function create_vae()
    # Define the encoder and decoder networks
    encoder_features = Chain(
        Dense(784,500, relu),
        Dense(500,500, relu)
    )
    encoder_μ = Chain(encoder_features, Dense(500, 20))
    encoder_logvar = Chain(encoder_features, Dense(500, 20))

    decoder = Chain(
        Dense(20, 500, relu, bias = false),
        Dense(500, 500, relu, bias = false),
        Dense(500, 784, sigmoid, bias = false)
    )
    return encoder_μ, encoder_logvar, decoder
end

# This function loads the parameters to the model 
# For now I'm using a pre-trained model

function load_model(load_dir::String, epoch::Int)
    print("Loading model...")
    @load joinpath(load_dir, "model-$epoch.bson") encoder_μ encoder_logvar decoder
    println("Done")
    return encoder_μ, encoder_logvar, decoder
end


function reconstruct_images(encoder_μ, encoder_logvar, decoder, x)
    # Forward propagate through mean encoder and std encoders
    μ = encoder_μ(x)
    logvar = encoder_logvar(x)
    # Apply reparameterisation trick to sample latent
    z = μ + randn(Float32, size(logvar)) .* exp.(0.5f0 * logvar)
    # Reconstruct from latent sample
    x̂ = decoder(z)
    return clamp.(x̂, 0 ,1)
end


function decoder_loss_function(z_variable, decoder, y_noisy)
    return 0.5*norm(y_noisy - decoder(z_variable))^2
end

# TO DO: rewrite as iterator
function gradient_descent(loss_function, z0, decoder, y_noisy, maxit, stepsize_numerator)
    zk = z0
    for t in 0:maxit
        loss_k = loss_function(zk, decoder, y_noisy)
        println("at iteration $t loss is $loss_k \n")
        resize_and_save_single_MNIST_image(clamp.(decoder(zk), 0.0, 1.0), "./scripts/temp/run_denoising", "during_GD_$t")
        zk .= zk - (stepsize_numerator/sqrt(t+1))*Zygote.gradient(z -> loss_function(z, decoder, y_noisy), zk)[1]
    end
    return zk
end


struct gradient_descent_state
    z_curr
    loss_value
    gradient
end

mutable struct denoising_problem
    z # optimization variable
    b # fit D(z) to observed image vector b
    loss_function # z -> 1/2||b - D(z)||_{2}^2
    loss_value # 1/2||b - D(z_curr)||_{2}^2
    decoder # D
end


function decoder_denoiser!(x,y,encoder_μ, encoder_logvar, decoder)
    #TO DO: load model outside of denoiser otherwise it's slow
    #model_epoch_number = 20
    #model_dir = "./src/utilities/saved_models/MNIST"
    #encoder_μ, encoder_logvar, decoder = load_model(model_dir,model_epoch_number)
    #resize_and_save_single_MNIST_image(y_curr,"./scripts/temp/denoising_decoder","before_denoising")
    loss_function = decoder_loss_function
    
    # as an initial point for gradient descent, take the current iterate (y) and apply encoder
    μ = encoder_μ(y)
    logvar = encoder_logvar(y)
    z0 = μ + randn(Float32, size(logvar)) .* exp.(0.5f0 * logvar)
    
    num_iter = 20 #number of iterations of gradient descent
    
    z_out = gradient_descent(loss_function,z0,decoder,y,num_iter,0.05)
    
    x = decoder(z_out)
end
