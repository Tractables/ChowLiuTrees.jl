using Test
using ChowLiuTrees


@testset "ChowLiuTrees Tests" begin
    mi = Float64.([0 1 3 4
          1 0 2 5
          3 2 0 6
          4 5 6 0])

    msts1 = ChowLiuTrees.topk_MST(mi, num_trees=10)
    
    msts2 = [[(1, 2), (2, 3), (1, 4)],
    [(1, 2), (1, 3), (1, 4)],
    [(1, 2), (2, 3), (2, 4)],
    [(1, 2), (1, 3), (2, 4)],
    [(1, 2), (2, 3), (3, 4)],
    [(2, 3), (1, 3), (1, 4)],
    [(1, 2), (1, 3), (3, 4)],
    [(1, 3), (2, 3), (2, 4)],
    [(2, 3), (1, 4), (2, 4)],
    [(1, 3), (2, 3), (3, 4)]]
    
    for (a, b) in zip(msts1, msts2)
        @test all(a .== b)  
    end
end