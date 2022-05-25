using FileIO
using ImageIO
using MLDatasets
using ImageCore
using LinearAlgebra


function save_MNIST_images(x_batch, save_dir::String, prefix::String)
    i = 1
    for x in eachslice(x_batch,dims = 3)
        convert_and_save_single_MNIST_image(x,save_dir,"$prefix-$i.png")
        i += 1
    end
end

# TO DO: remove clamp?
function resize_and_save_single_MNIST_image(x,save_dir::String,filename::String)
    save( joinpath(save_dir, "$filename.png"), colorview(Gray, reshape(x, 28, 28)') )
    #save(joinpath( save_dir, "$filename.png"), colorview(Gray, reshape(clamp.(x,0,1), 28,28)') )
end


function convert_and_save_single_MNIST_image(x,save_dir::String,filename::String)
    #save( joinpath(save_dir, "$filename.png"), MNIST.convert2image(clamp.(x,0,1)) )
    save(joinpath(save_dir, "$filename.png"), MNIST.convert2image(x)) 
end


function get_test_loader(batch_size,shuffle::Bool,type::String)
    # The FashionMNIST test set is made up of 10k 28 by 28 greyscale images
    if type == "FashionMNIST"
        test_x, test_y = FashionMNIST.testdata(Float32)
    elseif type == "MNIST"
        test_x, test_y = MNIST.testdata(Float32)
    end
    #test_x = 1 .- reshape(test_x, (784, :)) #reshape and flip white/black
    test_x = vectorize_and_flip(test_x)
    return DataLoader((test_x, test_y), batchsize=batch_size, shuffle=shuffle)
end



function get_test_images(start_index::Int64,end_index::Int64,type::String)
    @assert start_index <= end_index
    if type == "FashionMNIST"
        test_x, test_y = FashionMNIST.testdata(Float32,start_index:end_index)
    elseif type == "MNIST"
        test_x, test_y = MNIST.testdata(Float32,start_index:end_index)
    end
    #test_x = 1 .- reshape(test_x, (784, :)) #vectorize images, leave lables unchanged
    test_x = vectorize_and_flip(test_x)
    return (test_x, test_y)
end

# TO DO: make unit test for this
function vectorize_and_flip(x_in)
    m,n = size(x_in)
    x_out = 1 .- reshape(x_in, (m*n,:))
end

function add_additive_noise(x_in)
    x_out = x_in + randn(length(x_in))
end

function add_additive_noise(x_in,scale_factor)
    x_out = x_in + scale_factor*randn(length(x_in))
end


function rank_approx(F::SVD, k)
    U, S, V = F
    M = U[:, 1:k] * Diagonal(S[1:k]) * V[:, 1:k]'
    clamp01!(M)
end