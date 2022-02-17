using Test
using CUDA
using ChowLiuTrees


@testset "Pairwise marginals and mutual information for binary data" begin
    # Binary
    x_bit = Bool.([ 0  0  0  0
                1  0  1  1
                0  1  0  1
                1  1  0  0
                1  0  1  0
                1  0  1  0
                0  1  1  1
                0  0  1  1
                1  0  0  0
                0  1  1  1])
    
    mi_bit = pairwise_MI(x_bit, pseudocount=0.0)

    @test mi_bit[1,1] ≈ 0.6931471805599453
    @test mi_bit[1,2] ≈ 0.08630462173553435
    @test mi_bit[1,3] ≈ 0.0
    @test mi_bit[2,2] ≈ 0.6730116670092565
    @test mi_bit[2,3] ≈ 0.013844293808390695
    @test mi_bit[3,3] ≈ 0.6730116670092565

    data_types = if CUDA.functional()
        [CuMatrix{Bool}, Matrix{Bool}, Matrix{Int}, CuMatrix{Int}]
    else
        [Matrix{Bool}, Matrix{Int}]
    end
   
    for pseudocount in [0.0, 1.0, 100.0]
        for weights in [nothing, ones(10), 
                       [1.0, 0.5, 1.0, 2.0, 0.7, 1.0, 0.5, 1.0, 1.3, 1.0]]
            for Float in [Float64, Float32]
                mar1 = pairwise_marginal(x_bit, 
                        weights=weights, pseudocount=pseudocount, Float=Float)
                mi1 = pairwise_MI(x_bit,
                        weights=weights, pseudocount=pseudocount, Float=Float) 
                for T in data_types
                    x2 = T(x_bit)
                    mar2 = pairwise_marginal(x2;
                            weights=weights, pseudocount=pseudocount, Float=Float)
                    mi2 = pairwise_MI(x2; 
                            weights=weights, pseudocount=pseudocount, Float=Float)
                    
                    if T == CuMatrix{Bool}
                        mar2 = Array(mar2)
                        mi2 = Array(mi2)
                    end
                    if T == Matrix{Int} || T == CuMatrix{Int}
                        @test all(mar2[:, :, 1, 1] .≈ mar1[:, :, 1])
                        @test all(mar2[:, :, 1, 2] .≈ mar1[:, :, 2])
                        @test all(mar2[:, :, 2, 1] .≈ mar1[:, :, 3])
                        @test all(mar2[:, :, 2, 2] .≈ mar1[:, :, 4])
                    else
                        @test all(mar1 .≈ mar2)
                    end
                    @test all(isapprox(mi1, mi2; atol=1e-5))
                end
            end
        end
    end
end


@testset "Pairwise marginals and mutual information for categorical data" begin
    # Binary
    x = [ 2  1  3  3
        1  3  3  2
        1  2  2  3
        3  2  2  2
        3  3  2  2
        1  3  3  2
        2  3  2  3
        1  2  3  3
        3  1  2  3
        1  2  3  1] .- 1
    
    m = pairwise_marginal(x, pseudocount=0.0)
    mi = pairwise_MI(x, pseudocount=0.0)

    @test mi[1,1] ≈ 1.0296530140645737
    @test mi[1,2] ≈ 0.2502012117690939
    @test mi[1,3] ≈ 0.30431653267886216
    @test mi[2,2] ≈ 1.054920167986144
    @test mi[2,3] ≈ 0.0
    @test mi[2,4] ≈ 0.3025260261455486

    if CUDA.functional()
        m_gpu = pairwise_marginal(cu(x), pseudocount=0.0)
        mi_gpu = pairwise_MI(cu(x), pseudocount=0.0)

        @test all(m .≈ m_gpu)
        @test all(mi .≈ mi_gpu)
    end
    
end