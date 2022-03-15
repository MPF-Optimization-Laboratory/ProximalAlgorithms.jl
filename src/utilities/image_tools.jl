using FileIO
using ImageIO
using MLDatasets
using ImageCore


function save_MNIST_images(x_batch, save_dir::String, prefix::String)
    i = 1
    for x in eachslice(x_batch,dims = 3)
        convert_and_save_single_MNIST_image(x,save_dir,"$prefix-$i.png")
        i += 1
    end
end

function convert_and_save_single_MNIST_image(x,save_dir::String,filename::String)
    save( joinpath(save_dir, "$filename.png"), MNIST.convert2image(x) )
end