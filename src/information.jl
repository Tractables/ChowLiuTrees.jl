export pairwise_marginal, pairwise_MI

using LinearAlgebra: diagind, diag
using StatsFuns: xlogx, xlogy
using CUDA: CUDA, CuMatrix, CuVector, CuArray
using DataFrames: DataFrame


#############################
# Mutual Information on Binary Data
#############################

function pairwise_marginal(data::Union{Matrix{Bool}, CuMatrix{Bool}, BitMatrix}; 
        weights::Union{Vector, Nothing}=nothing, 
        pseudocount=1.0,
        Float=Float64)
    N = isnothing(weights) ? size(data, 1) : sum(weights)
    D = size(data, 2)
    base = N + pseudocount
    # not_data = .!data
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
    @view(pxy[:,:, 2])[diag] .-= joint_count
    @view(pxy[:,:, 3])[diag] .-= joint_count
    @view(pxy[:,:, 4])[diag] .+= joint_count

    pxy ./= base
end


function pairwise_MI(data::Union{Matrix{Bool}, CuMatrix{Bool}, BitMatrix}; 
        weights=nothing, 
        pseudocount=1.0,
        Float=Float64)
    pxy = pairwise_marginal(data; weights=weights, pseudocount=pseudocount)
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
    dropdims(sum(pxy_log_pxy - pxy_log_pxpy, dims=3), dims=3)
end


#############################
# Mutual Information on Categorical Data
#############################

function pairwise_marginal(data::Matrix;
        weights::Union{Vector, Nothing}=nothing,
        num_cats=maximum(data),
        pseudocount=1.0,
        Float=Float64)
    @assert minimum(data) > 0 "Categorical data labels are assumed to be indexed starting 1"

    num_samples = size(data, 1)
    num_vars = size(data, 2)

    if weights === nothing
        weights = ones(Int32, num_samples)
    else
        pseudocount = pseudocount * sum(weights) / num_samples
    end

    num_vars = size(data, 2)
    pair_margs = Array{Float}(undef, num_vars, num_vars, num_cats, num_cats)
    Z = sum(weights) + pseudocount
    single_smooth = pseudocount / num_cats
    pair_smooth = single_smooth / num_cats
    
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
                    @inbounds pair_margs[i,j,data[k,i],data[k,j]] += weights[k]
                end
                @inbounds pair_margs[i,j,:,:] ./= Z
            else
                @inbounds pair_margs[i,j,:,:] .= (@view pair_margs[j,i,:,:])' 
            end
            nothing
        end
    end

    return pair_margs
end


function pairwise_marginal(data::CuMatrix; 
        weights::Union{CuVector, Nothing}=nothing,
        num_cats=maximum(data),
        pseudocount=1.0,
        Float=Float64)

    @assert minimum(data) > 0 "Categorical data labels are assumed to be indexed starting 1"

    num_samples = size(data, 1)
    num_vars = size(data, 2)

    if weights === nothing
        weights = ones(Int32, num_samples)
    else
        pseudocount = pseudocount * sum(weights) / num_samples
    end

    num_vars = size(data,2)
    pair_margs = CuArray{Float}(undef, num_vars, num_vars, num_cats, num_cats)
    Z = sum(weights) + pseudocount
    single_smooth = pseudocount / num_cats
    pair_smooth = single_smooth / num_cats
    
    data_device = CUDA.cudaconvert(data)
    weights_device = CUDA.cudaconvert(weights)
    pair_margs_device = CUDA.cudaconvert(pair_margs)

    var_indices = CuArray(1:num_vars)
    CUDA.@sync begin
        broadcast(var_indices, var_indices') do i,j
            if i <= j
                if i!=j
                    @inbounds pair_margs_device[i,j,:,:] .= pair_smooth
                else
                    @inbounds pair_margs_device[i,j,:,:] .= zero(Float32)
                    for l = 1:num_cats
                        @inbounds pair_margs_device[i,j,l,l] = single_smooth
                    end
                end
                for k = 1:size(data_device,1)
                    pair_margs_device[i,j,data_device[k,i],data_device[k,j]] += weights_device[k]
                end
            end
            nothing
        end
        pair_margs ./= Z
        broadcast(var_indices, var_indices') do i,j
            if i > j
                for l = 1:num_cats, m = 1:num_cats
                    @inbounds pair_margs_device[i,j,l,m] = pair_margs_device[j,i,m,l] 
                end
            end
            nothing
        end
    end
    return pair_margs
end


function pairwise_MI(data::Matrix;
            weights::Union{Vector, Nothing} = nothing,
            num_cats = maximum(data),
            pseudocount = 1.0,
            Float = Float64)
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
    marginal_cont = zeros(Float64, num_vars, num_cats)
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


# TODO categorical pairwise MI on GPU