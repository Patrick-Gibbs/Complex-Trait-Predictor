using Random
struct LinearSNPDataSet
    SNPMatrix
    SNPEffects
    noiseWeight
    noiseGenerationMethod
    noiseVector
    phenotypes
    generateNoiseVector::Function
    
     
    # Constructor -- Fills struct and returns LinearSNPDataSet object. 
    # SNP generation methods can be specfied, noise can be included as a vector 
    function LinearSNPDataSet(dimention::Vector{Int64}; SNPGeneration="defult", 
           SNPEffects ="defult", noiseGenerationMethod ="defult", SNPsNoEffectProportion =0.5, noiseWeight=0.0)

        # Makes SNP Matrix
        if SNPGeneration == "defult"
        SNPMatrix = Array{Float64}(bitrand(dimention[1], dimention[2])) #(number of data points, number of SNPs)
        end

        # Make SNP Effects
        if SNPEffects == "defult"
            SNPEffectsVector = randn(dimention[2])
            # Set fixed proportion to 0
            for i in range(1, dimention[2])
                if rand() > SNPsNoEffectProportion
                    SNPEffectsVector[i] = 0.0
                end
            end
        end
        
        # Make Noise Vector, this is done using normal distrobution by defult
        generateNoiseVector = function (size, method="normal")
            if method == "normal"
                noiseVector = randn(size)*noiseWeight
            end
            return noiseVector
        end

        if noiseGenerationMethod == "defult"
            noiseVector = generateNoiseVector(dimention[1])
        end

        # Makes the phenotypes
        phenotypes = SNPMatrix*SNPEffectsVector + noiseVector

        new(SNPMatrix, SNPEffectsVector, noiseWeight, noiseGenerationMethod, noiseVector, 
            phenotypes, generateNoiseVector)
    end
    
    # Constructs a new LinearSNPDataSet which is a subSet of another LinearSNPDataSet
    # Datapoints are choosen from a vector of indexes "dataPoints"
    function LinearSNPDataSet(perantDataSet::LinearSNPDataSet, dataPoints)
        SNPMatrix = reduce(vcat, transpose.([perantDataSet.SNPMatrix[i,:] for i in dataPoints]))
        SNPEffectsVector = [perantDataSet.SNPEffects[i] for i in dataPoints]
        noiseWeight = perantDataSet.noiseWeight
        noiseGenerationMethod = perantDataSet.noiseGenerationMethod
        noiseVector = [perantDataSet.noiseVector[i] for i in dataPoints]
        phenotypes = [perantDataSet.phenotypes[i] for i in dataPoints]
        generateNoiseVector = perantDataSet.generateNoiseVector
        new(SNPMatrix, SNPEffectsVector, noiseWeight, noiseGenerationMethod, noiseVector, 
            phenotypes, generateNoiseVector)
    end
end
    