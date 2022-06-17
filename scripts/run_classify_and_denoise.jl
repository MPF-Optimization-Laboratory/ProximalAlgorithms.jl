#############################
## this code is copied from 
## https://github.com/ZhenanFanUBC/classified-compressed-sensing/blob/master/code/compressed_sensing_mnist_classifier.jl
#############################
include("../src/utilities/denoisers.jl")
include("../src/utilities/image_tools.jl")


using Flux
using MLDatasets
using Printf
using Flux: onehotbatch, crossentropy
using LinearAlgebra
using BSON: @save, @load
using Optim
using ImageCore
using FileIO
using ImageIO
#using ImageView

# load training data
@printf("load data ... ")
#trainset = MNIST(:train)
#X_train, y_train = trainset[:]
X_train, y_train = MNIST.traindata()
trainX = reshape(X_train, 784, 60000)
trainY = onehotbatch(y_train, 0:9)
@printf("done\n")

# load model
@printf("load model ... ")
@load "model_mnist.bson" model
@printf("done\n")

# loss function 
loss(x, y) = crossentropy(model(x), y)
# ground truth
xs = trainX[:, 2]; ys = trainY[:, 2]
filename = "ground_truth"
save(joinpath(".", "$filename.png"), MNIST.convert2image(xs))
# measurement
m = 392; M = randn(m, 784); b = M*xs
# objective
λ = 0.3; r(x) = λ*loss(x, ys); f(x) = norm(M*x - b)^2/2 + r(x)
# gradient
g(x) =  M'*(M*x - b) + gradient(r, x)[1]
# constraint
lower = zeros(784); upper = ones(784)
# initial point
x0 = ones(784)/2
# solve the optimization problem
@printf("solving optimization problem ... ")
result = optimize(f, g, lower, upper, x0; inplace = false)
@printf("done\n")

# show relative difference
x = Optim.minimizer(result)
@printf("relative difference = %.2f ", norm(x - xs)/norm(xs))
x = 1 .- x
# plot
#imshow(reshape(x, 28, 28))
filename = "classifier_result_before_denoise"
save(joinpath(".", "$filename.png"), MNIST.convert2image(x))

model_epoch_number = 20
model_dir = "./src/utilities/saved_models/MNIST"
encoder_μ, encoder_logvar, decoder = load_model(model_dir,model_epoch_number)

@printf("applying denoiser ... ")

x .= reconstruct_images(encoder_μ, encoder_logvar, decoder,x)

filename = "classifier_result_after_denoise"
save(joinpath(".", "$filename.png"), MNIST.convert2image(x))