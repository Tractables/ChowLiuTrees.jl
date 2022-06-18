export pairwise_marginal, pairwise_MI

using LinearAlgebra: diagind, diag
using CUDA

xlogx(x) =
    iszero(x) ? zero(x) : x * log(x)

xlogy(x, y) =
    iszero(x) && !isnan(y) ? zero(x) : x * log(y)

#############################
# Mutual Information on Binary Data
#############################

function pairwise_marginal(data::Union{Matrix{Bool}, CuMatrix{Bool}, BitMatrix}; 
        weights::Union{Vector, CuVector, Nothing}=nothing, 
        pseudocount=0.0,
        Float=Float64)
    N = isnothing(weights) ? size(data, 1) : sum(weights)
    D = size(data, 2)
    base = N + pseudocount
    if data isa CuMatrix{Bool} && !isnothing(weights)
        weights = CuVector(weights)
    end

    not_data = Float.(.!data)
    data = Float.(data)
    pxy = similar(data, D, D, 4)
    
    if isnothing(weights)
        pxy[:, :, 4] = transpose(data) * data
        pxy[:, :, 3] = transpose(data) * not_data
        pxy[:, :, 2] = transpose(not_data) * data
    else 
        pxy[:, :, 4] = transpose(data) * (data .* weights)
        pxy[:, :, 3] = transpose(data) * (not_data .* weights)
        pxy[:, :, 2] = transpose(not_data) * (data .* weights)
    end
    pxy[:, :, 1] = N .- pxy[:, :, 4] .- pxy[:, :, 3] .- pxy[:, :, 2]
    
    joint_count = pseudocount / 4
    pxy .+= joint_count

    # diagonal is the marginal of p(x)
    diag = diagind(pxy[:, :, 1])
    @view(pxy[:,:, 1])[diag] .+= joint_count
    @view(pxy[:,:, 2])[diag] .= zero(Float)
    @view(pxy[:,:, 3])[diag] .= zero(Float) 
    @view(pxy[:,:, 4])[diag] .+= joint_count

    if data isa CuMatrix
        CUDA.unsafe_free!(data)
        CUDA.unsafe_free!(not_data)
    end
    pxy ./= base
end


function pairwise_MI(data::Union{Matrix{Bool}, CuMatrix{Bool}, BitMatrix}; 
        weights=nothing, 
        pseudocount=0.0,
        Float=Float64)
    pxy = pairwise_marginal(data; weights, pseudocount, Float)
    p0 = diag(pxy[:, :, 1])
    p1 = diag(pxy[:, :, 4])
    D = size(data, 2)
    pxpy = Float.(similar(data, D, D, 4))
    pxpy[:,:,1] = p0 * p0'
    pxpy[:,:,2] = p0 * p1'
    pxpy[:,:,3] = p1 * p0'
    pxpy[:,:,4] = p1 * p1'
    pxy_log_pxy = @. xlogx(pxy)
    pxy_log_pxpy = @. xlogy(pxy, pxpy)
    MI = dropdims(sum(pxy_log_pxy - pxy_log_pxpy, dims=3), dims=3)
    if data isa CuMatrix{Bool}
        CUDA.unsafe_free!(pxy)
        CUDA.unsafe_free!(pxpy)
        CUDA.unsafe_free!(pxy_log_pxy)
        CUDA.unsafe_free!(pxy_log_pxpy)
    end
    MI
end


#############################
# Mutual Information on Categorical Data
#############################

function pairwise_marginal(data::Matrix;
                           weights::Union{Vector, Nothing} = nothing,
                           num_cats = maximum(data) + 1,
                           pseudocount = zero(Float32),
                           Float = Float32)

    @assert minimum(data) >= 0 "Categorical data labels are assumed to be indexed starting 0"

    num_samples = size(data, 1)
    num_vars = size(data, 2)

    if weights === nothing
        weights = ones(Float32, num_samples)
    else
        pseudocount = pseudocount * sum(weights) / num_samples
    end

    num_vars = size(data, 2)
    pair_margs = Array{Float}(undef, num_vars, num_vars, num_cats, num_cats)
    Z = sum(weights) + pseudocount
    single_smooth = Float(pseudocount / num_cats)
    pair_smooth = Float(single_smooth / num_cats)
    
    for i = 1:num_vars
        # note: multithreading needs to be on inner loop for thread-safe copying across diagonal
        Threads.@threads for j = 1:num_vars
            if i<=j
                if i!=j
                    @inbounds pair_margs[i,j,:,:] .= pair_smooth
                else
                    @inbounds pair_margs[i,j,:,:] .= zero(eltype(pair_margs))
                    for l = 1:num_cats
                        @inbounds pair_margs[i,j,l,l] = single_smooth
                    end
                end
                @simd for k = 1:size(data,1) # @avx here gives incorrect results
                    @inbounds pair_margs[i,j,data[k,i]+1,data[k,j]+1] += weights[k]
                end
                @inbounds pair_margs[i,j,:,:] ./= Z
            else
                @inbounds pair_margs[i,j,:,:] .= transpose(@view pair_margs[j,i,:,:])
            end
            nothing
        end
    end

    return pair_margs
end


function balance_threads(num_items, num_examples, config; mine, maxe)
    block_threads = config.threads
    # make sure the number of example threads is a multiple of 32
    num_item_batches = cld(num_items, maxe)
    num_blocks = cld(num_item_batches * num_examples, block_threads)
    if num_blocks < config.blocks
        max_num_item_batch = cld(num_items, mine)
        max_num_blocks = cld(max_num_item_batch * num_examples, block_threads)
        num_blocks = min(config.blocks, max_num_blocks)
        num_item_batches = (num_blocks * block_threads) ÷ num_examples
    end
    item_work = cld(num_items, num_item_batches)
    @assert item_work*block_threads*num_blocks >= num_examples*num_items
    block_threads, num_blocks, num_examples, item_work
end


function pairwise_marginal_kernel(pair_margs, data, weights, num_ex_threads::Int32, vpair_work::Int32)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    vpair_batch, ex_id = fldmod1(threadid, num_ex_threads)

    num_vars = size(pair_margs, 1)
    vpair_start = one(Int32) + (vpair_batch - one(Int32)) * vpair_work
    vpair_end = min(vpair_start + vpair_work - one(Int32), num_vars ^ Int32(2))
    
    @inbounds if ex_id <= size(data, 1)
        for vpair_id = vpair_start : vpair_end
            v1, v2 = fldmod1(vpair_id, num_vars)
            if v1 <= v2
                d1 = data[ex_id, v1]
                d2 = data[ex_id, v2]
                CUDA.@atomic pair_margs[v1,v2,d1+1,d2+1] += weights[ex_id]
            end
        end
    end
    nothing
end

function pairwise_marginal(data::CuMatrix; 
                           weights::Union{CuVector, Vector, Nothing} = nothing,
                           num_cats = maximum(data) + 1,
                           pseudocount = zero(Float32),
                           Float = Float32)

    @assert eltype(data) != Bool
    @assert minimum(data) >= 0 "Categorical data labels are assumed to be indexed starting 0"

    num_examples = size(data, 1)
    num_vars = size(data, 2)
    num_var_pairs = num_vars * num_vars

    if weights === nothing
        weights = ones(Float, num_examples)
    else
        pseudocount = pseudocount * sum(weights) / num_examples
    end

    num_vars = size(data,2)
    pair_margs = zeros(Float, num_vars, num_vars, num_cats, num_cats)
    Z = sum(weights) + pseudocount
    single_smooth = Float(pseudocount / num_cats)
    pair_smooth = Float(single_smooth / num_cats)
    
    # init pair_margs
    for i = 1 : num_vars
        for j = 1 : num_vars
            if i <= j
                if i != j
                    @inbounds pair_margs[i,j,:,:] .= pair_smooth
                else
                    @inbounds pair_margs[i,j,:,:] .= zero(eltype(pair_margs))
                    for l = 1 : num_cats
                        @inbounds pair_margs[i,j,l,l] = single_smooth
                    end
                end
            end
            nothing
        end
    end
    
    pair_margs = cu(pair_margs)
    weights = cu(weights)

    dummy_args = (pair_margs, data, weights, Int32(1), Int32(1))
    kernel = @cuda name="pairwise_marginal" launch=false pairwise_marginal_kernel(dummy_args...)
    config = launch_configuration(kernel.fun)

    threads, blocks, num_example_threads, vpair_work = 
        balance_threads(num_var_pairs, num_examples, config; mine=2, maxe=32)

    args = (pair_margs, data, weights, Int32(num_example_threads), Int32(vpair_work))
    kernel(args...; threads, blocks)

    pair_margs ./= Z
    pair_margs = Array(pair_margs)
    
    for i = 1 : num_vars
        for j = 1 : num_vars
            if i > j
                @inbounds pair_margs[i,j,:,:] .= transpose(@view pair_margs[j,i,:,:])
            end
        end
    end
    
    pair_margs
end


function pairwise_MI(data::Union{Matrix,CuMatrix};
                     weights::Union{Vector, Nothing} = nothing,
                     num_cats = maximum(data) + 1,
                     pseudocount = zero(Float32),
                     Float = Float32)
    num_samples = size(data, 1)
    num_vars = size(data, 2)
    
    if weights === nothing
        weights = ones(Int32, num_samples)
    else
        pseudocount = pseudocount * sum(weights) / num_samples
    end
    
    # `joint_cont[i, j, k, w]' is the total weight of samples whose i- and j-th variable are k and w, respectively
    joint_cont = pairwise_marginal(data; weights, num_cats, pseudocount)
    
    # `marginal_cont[i, j]' is the total weight of sample whose i-th variable is j
    marginal_cont = zeros(Float, num_vars, num_cats)
    for i = 1:num_vars
        for j = 1:num_cats
            @inbounds marginal_cont[i,j] = joint_cont[i,i,j,j]
        end
    end
    
    # Compute mutual information
    MI = zeros(Float, num_vars, num_vars)
    for var1_idx = 1 : num_vars
        for var2_idx = var1_idx : num_vars
            @inbounds MI[var1_idx, var2_idx] = sum(
                xlogx.(joint_cont[var1_idx, var2_idx, :, :]) .-
                xlogy.(joint_cont[var1_idx, var2_idx, :, :], marginal_cont[var1_idx, :] .* marginal_cont[var2_idx, :]'))
        end
    end
    
    for var1_idx = 2 : num_vars
        for var2_idx = 1 : var1_idx - 1
            @inbounds MI[var1_idx, var2_idx] = MI[var2_idx, var1_idx]
        end
    end
    
    MI
end

