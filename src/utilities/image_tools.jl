using FileIO
using ImageIO
using MLDatasets
using ImageCore


function save_to_images(x_batch, save_dir::String, prefix::String, num_images::Int64)
    @assert num_images <= size(x_batch)[2]
    for i=1:num_images
        save(joinpath(save_dir, "$prefix-$i.png"), colorview(Gray, reshape(x_batch[:, i], 28,28)' ))
    end
end

# TO DO : fix this function
function save_MNIST_images(x_batch, save_dir::String, prefix::String)
    i = 1
    for x in x_batch # need to figure out how to iterate over the batch
        convert_and_save_single_MNIST_image(x,save_dir,"$prefix-$i.png")
        i += 1
    end
end

function convert_and_save_single_MNIST_image(x,save_dir::String,filename::String)
    save( joinpath(save_dir, "$filename.png"), MNIST.convert2image(x) )
end