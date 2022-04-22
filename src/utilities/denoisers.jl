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


function decoder_loss_function(z_variable,decoder,y_noisy)
    return 0.5*norm(y_noisy - decoder(z_variable))^2
end

# TO DO: rewrite as iterator
function gradient_descent(loss_function,z0,decoder,y_noisy,maxit)
    zk = z0
    for t in 1:maxit
        zk .= zk + (1/sqrt(t))*Zygote.gradient(z -> loss_function(z,decoder,y_noisy), zk)[1]
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

Base.iterate