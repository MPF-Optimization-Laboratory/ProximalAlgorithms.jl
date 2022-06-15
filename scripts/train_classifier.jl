# using code from https://github.com/ZhenanFanUBC/classified-compressed-sensing/blob/master/code/train_classifier_mnist.jl

using Flux
using MLDatasets
#Flux.Data.MNIST, 
#using MNIST
using Statistics, Printf
#using Flux: onehotbatch, onecold, crossentropy
using Base.Iterators: repeated

# load training images
#imgs = MNIST.images(:train)
# load training labels
#labels = MNIST.labels(:train)

imgs, labels = MNIST.traindata()
#
# pre-processing of training images
trainX = hcat(float.(reshape.(imgs, :))...)
# pre-processing of training labels
trainY = onehotbatch(labels, 0:9)

# pre-processing of validation data
validationX = hcat(float.(reshape.(MNIST.images(:validation), :))...)
validationY = onehotbatch(MNIST.labels(:validation), 0:9)
# pre-processing of test data
testX = hcat(float.(reshape.(MNIST.images(:test), :))...)
testY = onehotbatch(MNIST.labels(:test), 0:9)

# bulid two layer neural network
# first layer with activation function relu
layer1 = Dense(784, 32, relu)
# second layer
layer2 = Dense(32, 10)
# construct the model with two layers and softmax function
model = Chain(layer1, layer2, softmax)

# loss function
loss(x, y) = crossentropy(model(x), y)
# optimizer
opt = ADAM();
# epochs
num_epochs = 100
data = repeated((trainX, trainY),num_epochs)
# accuracy function
accuracy(x, y) = mean(onecold(model(x)) .== onecold(y)) 
# call back function
evalcb = function ()
    @printf("training loss=%f, validation accuracy=%f\n", loss(trainX, trainY), accuracy(validationX, validationY))
    accuracy(validationX, validationY) > 0.9 && Flux.stop()
end
# train the model 
Flux.train!(loss, params(model), data, opt, cb = evalcb)

# save the model
using BSON: @save
@save "model_mnist_trained.bson" model
