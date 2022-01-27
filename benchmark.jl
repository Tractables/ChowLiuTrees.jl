using Revise, LogicCircuits, ChowLiuTrees, ProbabilisticCircuits
using CUDA

# datasetname = "nltcs"
datasetname = "nltcs"

train_x, valid_x, test_x = twenty_datasets(datasetname)

dataset = Matrix(train_x)
num_samples = size(dataset, 1)
num_vars = size(dataset, 2)

m1 = ChowLiuTrees.pairwise_marginals(dataset .+ 1, ones(Int32, num_samples), pseudocount=0.0);
m2 = pairwise_marginals_binary(dataset, pseudocount=0.0);

# sanity check
all(m1[:, :, 1, 1] .== Float32.(m2[:, :, 1]))
all(m1[:, :, 1, 2] .== Float32.(m2[:, :, 2]))
all(m1[:, :, 2, 1] .== Float32.(m2[:, :, 3]))
all(m1[:, :, 2, 2] .== Float32.(m2[:, :, 4]))

# marginal 
p0, p1 = marginals_binary(dataset; weights=nothing, pseudocount=0.0)
mi1 = pairwise_MI(train_x .+ 1, num_vars, 2, pseudocount=0.0)
mi1 = pairwise_MI(train_x .+ 1, num_vars, 2, pseudocount=0.0)
mi2 = pairwise_MI_binary(dataset, pseudocount=0.0)

mi1 .- mi2

# 
# TODO: pseudocount
# TODO: add simple test


# Benchmark
weights = ones(Int32, num_samples)
gpu_data = to_gpu(dataset)
gpu_weights = to_gpu(weights)

cat_data = dataset .+ 1
cat_gpu_data = gpu_data .+ 1

@btime pairwise_marginals(cat_data, weights, pseudocount=0.0);
@btime pairwise_marginals_binary(dataset, pseudocount=0.0);
@btime CUDA.@sync pairwise_marginals(cat_gpu_data, gpu_weights, pseudocount=0.0);
@btime CUDA.@sync pairwise_marginals_binary(gpu_data, pseudocount=0.0);


using MLDatasets, BenchmarkTools

train_int = transpose(reshape(MNIST.traintensor(UInt8), 28*28, :));
test_int = transpose(reshape(MNIST.testtensor(UInt8), 28*28, :));

function bitsfeatures(data_int)
    data_bits = zeros(Bool, size(data_int,1), 28*28*8)
    for ex = 1:size(data_int,1), pix = 1:size(data_int,2)
        x = data_int[ex,pix]
        for b = 0:7
            if (x & (one(UInt8) << b)) != zero(UInt8)
                data_bits[ex, (pix-1)*8+b+1] = true
            end
        end
    end
    data_bits
end

train_bits = bitsfeatures(train_int);
test_bits = bitsfeatures(train_int);

train_bits_gpu = to_gpu(train_bits);
@time CUDA.@sync MI = pairwise_MI_binary(train_bits_gpu, pseudocount=0.0);
MI = to_cpu(MI)
@time max_spanning_tree(MI)
@time topk_MST(MI)

# TODO GPU MST
