using DataStructures: PriorityQueue, enqueue!, dequeue!, dequeue_pair!
using Graphs: SimpleGraph, SimpleDiGraph, complete_graph, kruskal_mst, 
    bfs_tree, center, connected_components, induced_subgraph, src, dst
using SimpleWeightedGraphs: SimpleWeightedGraph, SimpleWeightedEdge
using MetaGraphs: MetaDiGraph, set_prop!, props, nv, ne, edges, add_edge!, vertices

export MST1, topk_MST, chow_liu_tree1


function chow_liu_tree1(train_x::Union{Matrix, CuMatrix}; pseudocount=0.1)
    MI = pairwise_MI_binary(train_x, pseudocount=pseudocount);
    MI = Array(MI)
    topk_MST(MI)
end

function MST1(MI; clt_root="graph_center")

    features_num = size(MI, 1)
    # maximum spanning tree/ forest
    g = SimpleWeightedGraph(complete_graph(features_num))
    mst_edges = kruskal_mst(g,- MI)
    tree = SimpleGraph(features_num)
    map(mst_edges) do edge
        add_edge!(tree, src(edge), dst(edge))
    end

    # Build rooted tree / forest
    if clt_root == "graph_center"
        clt = SimpleDiGraph(features_num)
        for c in filter(c -> (length(c) > 1), connected_components(tree))
            sg, vmap = induced_subgraph(tree, c)
            sub_root = vmap[center(sg)[1]]
            clt = union(clt, bfs_tree(tree, sub_root))
        end
    elseif clt_root == "rand"
        roots = [rand(c) for c in connected_components(tree)]
        clt = SimpleDiGraph(features_num)
        for root in roots clt = union(clt, bfs_tree(tree, root)) end
    else
        error("Cannot learn CLT with root $clt_root")
    end
    
    clt = MetaDiGraph(clt)
    parent = parent_vector(clt)
    for (c, p) in enumerate(parent)
        set_prop!(clt, c, :parent, p)
    end

    return clt
end


"Reference: Listing all the minimum spanning trees in an undirected graph
 http://www.nda.ac.jp/~yamada/paper/enum-mst.pdf"
 function topk_MST(MI; num_trees::Integer = 1, dropout_prob::Float64 = 0.0)
    num_vars = size(MI, 1)
    # Priority queue that maintain candidate MSTs
    candidates = PriorityQueue{Tuple{Vector{SimpleWeightedEdge}, Vector{SimpleWeightedEdge}, Vector{SimpleWeightedEdge}}, Float32}()
    
    # The fully connect graph and its weight
    g = SimpleWeightedGraph(complete_graph(num_vars))
    weights = -MI
    
    included_edges::Vector{SimpleWeightedEdge} = Vector{SimpleWeightedEdge}()
    excluded_edges::Vector{SimpleWeightedEdge} = Vector{SimpleWeightedEdge}()
    reuse = Matrix{Float64}(undef, num_vars, num_vars)
    topk_msts::Vector{Vector{SimpleWeightedEdge}} = Vector{Vector{SimpleWeightedEdge}}()
    
    # Initialize `candidate` with the global MST
    mst_edges, total_weight = MST(g; weights, included_edges, excluded_edges, reuse, dropout_prob = 0.0)
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

                candidate_mst, total_weight = MST(g; weights, included_edges, excluded_edges, reuse, dropout_prob)
                if candidate_mst !== nothing
                    # A shallow copy of the vectors `included_edges` and `excluded_edges` is sufficient
                    enqueue!(candidates, (candidate_mst, copy(included_edges), copy(excluded_edges)), total_weight) 
                end
            end
        end
        
    else
        
        # Parallel code
        reuse = map(1:Threads.nthreads()) do idx
            Matrix{Float64}(undef, num_vars, num_vars)
        end
        g = map(1:Threads.nthreads()) do idx
            deepcopy(g)
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
                candidate_mst, total_weight = MST(g[id], weights[id], curr_included_edges, curr_excluded_edges; reuse = reuse[id], dropout_prob)

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
    map(topk_msts) do mst_edges
        MStree = SimpleGraph(num_vars)
        map(mst_edges) do edge
            add_edge!(MStree, src(edge), dst(edge))
        end
        
        # Use the graph center of `MStree' as the root node of the CLT
        clt = SimpleDiGraph(num_vars)
        for c in filter(c -> (length(c) > 1), connected_components(MStree))
            sg, vmap = induced_subgraph(MStree, c)
            sub_root = vmap[center(sg)[1]]
            clt = union(clt, bfs_tree(MStree, sub_root))
        end

        MetaDiGraph(clt)
    end
end


"Compute the Minimum Spanning Tree (MST) of graph g with weights `weights`, with
 constraints such that `included_edges` should be included while `excluded_edges` 
 should be excluded."
function MST(g::SimpleWeightedGraph;
             weights::Matrix{<:AbstractFloat}, 
             included_edges::Vector{SimpleWeightedEdge}, 
             excluded_edges::Vector{SimpleWeightedEdge},
             reuse::Matrix{<:AbstractFloat}, dropout_prob = 0.0)
    T = eltype(weights)
    @inbounds @views reuse[:, :] .= weights[:, :]
    
    # Dropout
    if dropout_prob > 1e-8
        dropped_mask = rand(Bernoulli(dropout_prob), size(reuse, 1), size(reuse, 2))
        @inbounds @views reuse[dropped_mask] .= 10000.0
    end
    
    # Add constraints
    map(included_edges) do edge
        reuse[src(edge), dst(edge)] = -10000.0
        reuse[dst(edge), src(edge)] = -10000.0
        nothing # Return nothing to save some effort
    end
    map(excluded_edges) do edge
        reuse[src(edge), dst(edge)] = 10000.0
        reuse[dst(edge), src(edge)] = 10000.0
        nothing # Return nothing to save some effort
    end
    
    mst_edges = kruskal_mst(g, reuse)
    
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
            total_weight += weights[src(edge), dst(edge)]
            nothing
        end
        mst_edges, total_weight
    else
        nothing, nothing
    end
end

function parent_vector(tree)::Vector{Int64}
    v = zeros(Int64, nv(tree)) # parent of roots is 0
    foreach(e->v[dst(e)] = src(e), edges(tree))
    return v
end

