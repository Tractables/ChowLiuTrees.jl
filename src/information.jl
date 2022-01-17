export pairwise_marginals_binary, 
pairwise_marginals, 
pairwise_mutual_information

# using Statistics
# using StatsFuns: xlogx, xlogy
using CUDA: CUDA, CuMatrix, CuVector, CuArray
using DataFrames: DataFrame
# using Tables

function cache_distributions(bm, w::Union{Nothing, Vector}=nothing; α, flag=(pairwise=true, marginal=true))
    
    # parameters
    D = size(bm)[2]
    N = issomething(w) ? sum(w) : size(bm)[1]
    m = Matrix{Float64}(Tables.matrix(bm))
    notm = Matrix{Float64}(Tables.matrix(.!bm))

    dis_cache = DisCache(D)
    base = N + 4 * α
    w = isnothing(w) ? ones(Float64, N) : w

    # pairwise distribution
    if flag.pairwise
        dis_cache.pairwise[:,:,1] = (notm' * (notm .* w) .+ α) / base   # p00
        dis_cache.pairwise[:,:,2] = (notm' * (m .* w) .+ α) / base      # p01
        dis_cache.pairwise[:,:,3] = (m' * (notm .* w) .+ α) / base      # p10
        dis_cache.pairwise[:,:,4] = (m' * (m .* w) .+ α) / base         # p11
    end
    # marginal distribution

    if flag.marginal
        dis_cache.marginal[:, 1] = (sum(notm .* w, dims=1) .+ 2 * α) / base
        dis_cache.marginal[:, 2] = (sum(m .* w, dims=1).+ 2 * α) / base
    end
    dis_cache
end

function pairwise_marginals_binary(data; weights=nothing, pseudocount = 1.0)
    N = isnothing(weights) ? size(data, 1) : sum(weights)
    base = N + 4 * pseudocount
    # not_data = .!data
    not_data = Float32.(.!data)
    data = Float32.(data)
    
    if isnothing(weights)
        count11 = transpose(data) * data
        count01 = transpose(not_data) * data
        count10 = transpose(data) * not_data
        count00 = base .- count11 .- count10 .- count01
    else
        count11 = transpose(data) * (data .* weights)
        count01 = transpose(not_data) * (data .* weights)
        count10 = transpose(data) * (not_data .* weights)
        count00 = base .- count11 .- count10 .- count01
    end
    ((count00 .+ pseudocount) / base, (count01 .+ pseudocount) / base,
    (count10 .+ pseudocount) / base, (count11 .+ pseudocount) / base)
end

"Compute an array giving all pairwise marginals estimated on empirical (weighted) data"
function pairwise_marginals(data::Matrix, weights::Vector, num_cats = maximum(data); pseudocount = 1.0)
    
    @assert minimum(data) > 0 "Categorical data labels are assumed to be indexed starting 1"
    num_vars = size(data,2)
    pair_margs = Array{Float32}(undef, num_vars, num_vars, num_cats, num_cats)
    Z = sum(weights) + pseudocount
    single_smooth = pseudocount / num_cats
    pair_smooth = single_smooth / num_cats
    
    for i = 1:num_vars 
        Threads.@threads for j = 1:num_vars # note: multithreading needs to be on inner loop for thread-safe copying across diagonal
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

function pairwise_marginals(data::CuMatrix, weights::CuVector, num_cats = maximum(data); pseudocount = 1.0)
    @assert minimum(data) > 0 "Categorical data labels are assumed to be indexed starting 1"
    num_vars = size(data,2)
    pair_margs = CuArray{Float32}(undef, num_vars, num_vars, num_cats, num_cats)
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


"Cache pairwise / marginal distribution for all variables in one dataset"
mutable struct DisCache
    pairwise::Array{Float64}
    marginal::Array{Float64}
end

@inline dimension(discache::DisCache) = size(discache.marginal)[1]
DisCache(num) = DisCache(Array{Float64}(undef, num, num, 4), Array{Float64}(undef, num, 2))

#####################
# Mutual Information
#####################

function mutual_information(dis_cache::DisCache)
    D = dimension(dis_cache)
    p0 = @view dis_cache.marginal[:, 1]
    p1 = @view dis_cache.marginal[:, 2]
    pxpy = Array{Float64}(undef, D, D, 4)
    pxpy[:,:,1] = p0 * p0'
    pxpy[:,:,2] = p0 * p1'
    pxpy[:,:,3] = p1 * p0'
    pxpy[:,:,4] = p1 * p1'
    pxy_log_pxy = @. xlogx(dis_cache.pairwise)
    pxy_log_pxpy = @. xlogy(dis_cache.pairwise, pxpy)
    dropdims(sum(pxy_log_pxy - pxy_log_pxpy,dims=3), dims=3)
end

"Calculate mutual information of given bit matrix `bm`, example weights `w`, and smoothing pseudocount `α`"
function mutual_information(bm, w::Union{Nothing, Vector}=nothing; α)
    dis_cache = cache_distributions(bm, w; α=α)
    mi = mutual_information(dis_cache)
    return (dis_cache, mi)
end

"Calculate set mutual information"
function set_mutual_information(mi::Matrix, sets::AbstractVector{<:AbstractVector})::Matrix
    len = length(sets)
    if len == size(mi)[1]
        return mi
    end

    pmi = zeros(len, len)
    for i in 1 : len, j in i + 1 : len
        pmi[i, j] = pmi[j, i] = mean(mi[sets[i], sets[j]])
    end
    return pmi
end


"Compute pairwise Mutual Information given binary/categorical data."
function pairwise_MI(dataset::DataFrame, num_vars, num_cats; pseudocount = 1.0)
    categorical_dataset, weights = as_categorical_data(dataset::DataFrame, num_vars, num_cats)
    pairwise_MI(categorical_dataset, num_vars, num_cats, weights; pseudocount = pseudocount)
end


function pairwise_MI(dataset::Matrix, num_vars, num_cats, weights = nothing; pseudocount = 1.0)
    num_samples = size(dataset, 1)
    num_vars = size(dataset, 2)
    
    if weights === nothing
        weights = ones(Int32, num_samples)
    else
        pseudocount = pseudocount * sum(weights) / num_samples
    end
    
    # Sum of the weights
    sum_weights::Float64 = Float64(sum(weights) + num_cats^2 * pseudocount)
    
    # `joint_cont[i, j, k, w]' is the total weight of samples whose i- and j-th variable are k and w, respectively
    joint_cont = pairwise_marginals(dataset, weights, num_cats; pseudocount)
    
    # `marginal_cont[i, j]' is the total weight of sample whose i-th variable is j
    marginal_cont = zeros(Float64, num_vars, num_cats)
    for i = 1:num_vars
        for j = 1:num_cats
            @inbounds marginal_cont[i,j] = joint_cont[i,i,j,j]
        end
    end
    
    # Compute mutual information
    MI = zeros(Float64, num_vars, num_vars)
    for var1_idx = 1 : num_vars
        for var2_idx = var1_idx : num_vars
            @inbounds MI[var1_idx, var2_idx] = sum(joint_cont[var1_idx, var2_idx, :, :] .* (@. log(sum_weights .* joint_cont[var1_idx, var2_idx, :, :] / (marginal_cont[var1_idx, :] .* marginal_cont[var2_idx, :]')))) / sum_weights
        end
    end
    
    for var1_idx = 2 : num_vars
        for var2_idx = 1 : var1_idx - 1
            @inbounds MI[var1_idx, var2_idx] = MI[var2_idx, var1_idx]
        end
    end
    
    MI
end