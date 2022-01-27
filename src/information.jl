export pairwise_marginals_binary, marginals_binary, pairwise_MI_binary,
pairwise_marginals, 
pairwise_mutual_information,
pairwise_MI,
as_categorical_datas

# using Statistics
using StatsFuns: xlogx, xlogy
using CUDA: CUDA, CuMatrix, CuVector, CuArray
using DataFrames: DataFrame

#############################
# Mutual Information Binary
#############################

function pairwise_marginals_binary(data::Union{Matrix, CuMatrix}; weights=nothing, pseudocount = 1.0)
    N = isnothing(weights) ? size(data, 1) : sum(weights)
    D = size(data, 2)
    base = N + 4 * pseudocount
    # not_data = .!data
    not_data = Float64.(.!data)
    data = Float64.(data)
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
    pxy[:, :, 1] = base .- pxy[:, :, 4] .- pxy[:, :, 3] .- pxy[:, :, 2]

    pxy .+= pseudocount
    pxy ./= base
    pxy
end


function marginals_binary(data::Union{Matrix, CuMatrix}; weights=nothing, pseudocount = 1.0)

    N = isnothing(weights) ? size(data, 1) : sum(weights)
    base = N + 4 * pseudocount
    data = Float64.(data)
    
    if isnothing(weights)
        count1 = dropdims(sum(data, dims=1), dims=1)
    else
        count1 = dropdims(sum(data .* weights, dims=1), dims=1)
    end
    count0 = base .- count1
    

    (count0 .+ 2 * pseudocount) / base, (count1 .+ 2 * pseudocount) / base
end


function pairwise_MI_binary(data::Union{Matrix, CuMatrix}; weights=nothing, pseudocount=1.0)
    pxy = pairwise_marginals_binary(data; weights=weights, pseudocount=pseudocount)
    p0, p1 = marginals_binary(data; weights=weights, pseudocount=pseudocount)
    D = size(data, 2)
    pxpy = Float64.(similar(data, D, D, 4))
    pxpy[:,:,1] = p0 * p0'
    pxpy[:,:,2] = p0 * p1'
    pxpy[:,:,3] = p1 * p0'
    pxpy[:,:,4] = p1 * p1'
    pxy_log_pxy = @. xlogx(pxy)
    pxy_log_pxpy = @. xlogy(pxy, pxpy)
    dropdims(sum(pxy_log_pxy - pxy_log_pxpy, dims=3), dims=3)
end


#############################
# Mutual Information Categorical
#############################


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
            @inbounds MI[var1_idx, var2_idx] = sum(xlogy.(joint_cont[var1_idx, var2_idx, :, :], 
            joint_cont[var1_idx, var2_idx, :, :] ./ (marginal_cont[var1_idx, :] .* marginal_cont[var2_idx, :]')))
        end
    end
    
    for var1_idx = 2 : num_vars
        for var2_idx = 1 : var1_idx - 1
            @inbounds MI[var1_idx, var2_idx] = MI[var2_idx, var1_idx]
        end
    end
    
    MI
end

function as_categorical_data(dataset::DataFrame, num_vars, num_cats)
    
    # Convert binary data to categorical data
    if isweighted(dataset)
        dataset, weights = split_sample_weights(dataset)
        weights = convert(Vector{Float64}, weights)
    else
        weights = nothing
    end
    
    num_samples = size(dataset, 1)
    num_bits = ceil(Int,log2(num_cats))
    
    # Get categorical dataset from the binarized dataset
    categorical_dataset = Matrix{UInt32}(undef, num_samples, num_vars)
    if iscomplete(dataset)
        for sample_idx = 1 : num_samples
            for variable_idx = 1 : num_vars
                @inbounds categorical_dataset[sample_idx, variable_idx] = as_cat(dataset[sample_idx, (variable_idx - 1) * num_bits + 1 : variable_idx * num_bits]; complete = true)
            end
        end
    else # If the dataset contains missing values, we impute the missing values with the mode of each column
        for variable_idx = 1 : num_vars
            cat_counts::Array{UInt32} = zeros(UInt32, num_cats)
            for sample_idx = 1 : num_samples
                category = as_cat(dataset[sample_idx, (variable_idx - 1) * num_bits + 1 : variable_idx * num_bits]; complete = false)
                if category != typemax(UInt32)
                    cat_counts[category] += 1
                end
                @inbounds categorical_dataset[sample_idx, variable_idx] = category
            end
            cat_mode = argmax(cat_counts)
            for sample_idx = 1 : num_samples
                if categorical_dataset[sample_idx, variable_idx] == typemax(UInt32)
                    @inbounds categorical_dataset[sample_idx, variable_idx] = cat_mode
                end
            end
        end
    end
    return categorical_dataset, weights
end


isweighted(df::DataFrame) = 
    names(df)[end] == "weight"
isweighted(df::Vector{DataFrame}) = 
    all(d -> isweighted(d), df)


"Is the data complete (no missing values)?"
iscomplete(data::DataFrame) = 
    all(iscomplete_col, eachcol_unweighted(data))
iscomplete(data::Vector{DataFrame}) = all(iscomplete, data)

"Is the data column complete (no missing values)?"
iscomplete_col(::AbstractVector{Bool}) = true
iscomplete_col(::AbstractVector{<:Int}) = true
iscomplete_col(::AbstractVector{<:AbstractFloat}) = true
iscomplete_col(x::AbstractVector{Union{Bool,Missing}}) = !any(ismissing, x)
iscomplete_col(x::AbstractVector{Union{<:AbstractFloat,Missing}}) = !any(ismissing, x)


"Iterate over columns, excluding the sample weight column"
eachcol_unweighted(data::DataFrame) = 
    isweighted(data) ? eachcol(data)[1:end-1] : eachcol(data)


function as_cat(bits; complete)
    if !complete && !iscomplete(bits)
        return typemax(UInt32)
    end
    
    category::UInt32 = UInt32(0)
    for bit_idx = length(bits) : -1 : 1
        category = (category << 1) + bits[bit_idx]
    end
    
    (category == 0) ? 2^(length(bits)) : category
end