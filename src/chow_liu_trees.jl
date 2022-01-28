using DataStructures: PriorityQueue, enqueue!, dequeue!, dequeue_pair!,
IntDisjointSets, in_same_set, union!
using CUDA

export learn_chow_liu_tree


"Learn Chow Liu Tree(s). It will run on CPU/GPU based on where `train_x` is.

Arguments: 
- `train_x`: training data. If want gpu, move to gpu before calling this `CuArray(train_x)`.

Keyword arguments:
- `num_trees=1`: number of trees you want to learned.
- `dropout_prob=0.0`: drop edges with probability `dropout_prob` when learning maximum spanning tree.
- `weights=nothing`: weights of samples. Weights are all 1 if `nothing`.
- `pseudocount=0.0`: add a total of pseudo count spread out overall all categories.
- `Float=Float64`: precision. `Float32` is faster if `train_x` a large.
"
function learn_chow_liu_tree(train_x::Union{Matrix, CuMatrix, BitMatrix};
        num_trees::Integer=1, dropout_prob::Float64=0.0,
        weights::Union{Vector, CuVector, Nothing}=nothing, 
        pseudocount::Float64=1.0,
        Float=Float64)
    MI = pairwise_MI(train_x; weights, pseudocount, Float)
    MI = Array(MI) # TODO: GPU MST
    topk_MST(- MI; num_trees, dropout_prob)
end


"Top k minimum spanning trees for a complete graph with `weights` as weights
http://www.nda.ac.jp/~yamada/paper/enum-mst.pdf
"
function topk_MST(weights::Matrix; 
        num_trees::Integer=1, dropout_prob::Float64=0.0)
    
    # Priority queue that maintain candidate MSTs
    T = Vector{Tuple{Int, Int}}
    candidates = PriorityQueue{Tuple{T, T, T}, Float32}()
    
    included_edges::T = T()
    excluded_edges::T = T()
    reuse = similar(weights)
    topk_msts::Vector{T} = Vector{T}()
    
    # Initialize `candidate` with the global MST
    mst_edges, total_weight = MST(weights, included_edges, excluded_edges;
                                  reuse, dropout_prob=0.0)
    enqueue!(candidates, (mst_edges, included_edges, excluded_edges), total_weight)
    
    
    if Threads.nthreads() == 1
        
        # Sequential code
        for idx = 1 : num_trees
            if isempty(candidates)
                break
            end

            (mst_edges, included_edges, excluded_edges), total_weight = dequeue_pair!(candidates)

            # Record the current ST into `topk_msts`
            push!(topk_msts, mst_edges)
            
            if idx == num_trees
                break
            end

            edge_added = false
            for edge_idx = 1 : length(mst_edges)
                if mst_edges[edge_idx] in included_edges
                    continue
                end

                if edge_added
                    push!(included_edges, pop!(excluded_edges))
                end
                push!(excluded_edges, mst_edges[edge_idx])
                edge_added = true

                candidate_mst, total_weight = MST(weights, included_edges, excluded_edges;
                                                reuse, dropout_prob)
                if candidate_mst !== nothing
                    # A shallow copy of the vectors `included_edges` and `excluded_edges` is sufficient
                    enqueue!(candidates, (candidate_mst, copy(included_edges), copy(excluded_edges)), total_weight) 
                end
            end
        end
        
    else
        
        # Parallel code
        reuse = map(1:Threads.nthreads()) do idx
            similar(weights)
        end

        weights = map(1:Threads.nthreads()) do idx
            deepcopy(weights)
        end
        
        l = ReentrantLock()
        
        for idx = 1 : num_trees
            if isempty(candidates)
                break
            end

            (mst_edges, included_edges, excluded_edges), total_weight = dequeue_pair!(candidates)

            # Record the current ST into `topk_msts`
            push!(topk_msts, mst_edges)
            
            if idx == num_trees
                break
            end

            Threads.@threads for edge_idx = 1 : length(mst_edges)
                curr_included_edges = copy(included_edges)
                curr_excluded_edges = copy(excluded_edges)
                for edge in mst_edges[1:edge_idx-1]
                    if !(edge in included_edges)
                        push!(curr_included_edges, edge)
                    end
                end
                if !(mst_edges[edge_idx] in excluded_edges)
                    push!(curr_excluded_edges, mst_edges[edge_idx])
                end

                id = Threads.threadid()
                candidate_mst, total_weight = MST(weights[id], 
                                                curr_included_edges, 
                                                curr_excluded_edges; 
                                                reuse = reuse[id], 
                                                dropout_prob)

                lock(l)
                if candidate_mst !== nothing
                    # A shallow copy of the vectors `included_edges` and `excluded_edges` is sufficient
                    enqueue!(candidates, (candidate_mst, copy(included_edges), copy(excluded_edges)), total_weight) 
                end
                unlock(l)
            end
        end
        
    end
    
    # Post-process the top-K Spanning Trees
    topk_msts
end


"Compute the Minimum Spanning Tree (MST) of graph g with weights `weights`, with
 constraints such that `included_edges` should be included while `excluded_edges` 
 should be excluded."
function MST(weights::Matrix, 
             included_edges::Vector{Tuple{Int, Int}}, 
             excluded_edges::Vector{Tuple{Int, Int}};
             reuse::Matrix, dropout_prob=0.0)
    T = eltype(weights)
    @inbounds @views reuse[:, :] .= weights[:, :]
    
    # Dropout
    if dropout_prob > 1e-8
        dropped_mask = rand(Bernoulli(dropout_prob), size(reuse, 1), size(reuse, 2))
        @inbounds @views reuse[dropped_mask] .= Inf
    end
    
    # Add constraints
    map(included_edges) do edge
        reuse[edge[1], edge[2]] = -Inf
        reuse[edge[2], edge[1]] = -Inf
        nothing # Return nothing to save some effort
    end
    map(excluded_edges) do edge
        reuse[edge[1], edge[2]] = Inf
        reuse[edge[2], edge[1]] = Inf
        nothing # Return nothing to save some effort
    end
    
    mst_edges = kruskal_mst_complete(reuse)
    
    # Senity check
    valid_tree::Bool = true
    
    edges = Set(mst_edges)
    map(included_edges) do edge
        if !(edge in edges)
            valid_tree = false
        end
        nothing
    end
    map(excluded_edges) do edge
        if (edge in edges)
            valid_tree = false
        end
        nothing
    end
    
    if valid_tree
        # Compute the tree weight
        total_weight::T = 0.0
        map(mst_edges) do edge
            total_weight += weights[edge[1], edge[2]]
            nothing
        end
        mst_edges, total_weight
    else
        nothing, nothing
    end
end


function kruskal_mst_complete(distmx::Matrix; minimize=true)
    T = eltype(distmx)

    nv = size(distmx, 1)
    ne = Int(nv * (nv - 1) / 2)
    connected_vs = IntDisjointSets(nv)

    mst = Vector{Tuple{Int64, Int64}}()
    sizehint!(mst, nv - 1)

    weights = Vector{T}()
    sizehint!(weights, ne)
    edge_list = []
    for i in 1 : nv
        for j in i + 1 : nv
            push!(edge_list, (i, j))
        end
    end

    for e in edge_list
        push!(weights, distmx[e[1], e[2]])
    end

    for e in edge_list[sortperm(weights; rev=!minimize)]
        if !in_same_set(connected_vs, e[1], e[2])
            union!(connected_vs, e[1], e[2])
            push!(mst, e)
            (length(mst) >= nv - 1) && break
        end
    end
    return mst
end
