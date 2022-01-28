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

   
    for pseudocount in [0.0, 1.0, 100.0]
        for weights in [nothing, ones(4), [0.1, 0.2, 0.3, 0.4]]
            for Float in [Float64, Float32]
                mar1 = pairwise_marginal(x_bit, pseudocount=pseudocount)
                mi1 = pairwise_MI(x_bit, pseudocount=pseudocount) 
                for T in [CuArray{Bool}, Matrix{Bool}, Matrix{Int}, ] # CuArray{Int}
                    x2 = T(x_bit)
                    if T == Matrix{Int}
                        x2 .+= 1
                    end

                    if T == CuArray{Bool} && !isnothing(weights)
                        weights = CuArray(weights)
                    end

                    mar2 = pairwise_marginal(x2; pseudocount=pseudocount)
                    mi2 = pairwise_MI(x2; pseudocount=pseudocount)
                    
                    if T == CuArray{Bool} 
                        mar2 = Array(mar2)
                        mi2 = Array(mi2)
                    end

                    if T == Matrix{Int}
                        @test all(mar2[:, :, 1, 1] .== mar1[:, :, 1])
                        @test all(mar2[:, :, 1, 2] .== mar1[:, :, 2])
                        @test all(mar2[:, :, 2, 1] .== mar1[:, :, 3])
                        @test all(mar2[:, :, 2, 2] .== mar1[:, :, 4])
                    else
                        @test all(mar1 .== mar2)
                    end
                    @test all(mi1 .≈ mi2)
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
        1  2  3  1]
    
    m = pairwise_marginal(x, pseudocount=0.0)
    mi = pairwise_MI(x, pseudocount=0.0)

    @test mi[1,1] ≈ 1.0296530140645737
    @test mi[1,2] ≈ 0.2502012117690939
    @test mi[1,3] ≈ 0.30431653267886216
    @test mi[2,2] ≈ 1.054920167986144
    @test mi[2,3] ≈ 0.0
    @test mi[2,4] ≈ 0.3025260261455486

    # TODO: compare with GPU
end