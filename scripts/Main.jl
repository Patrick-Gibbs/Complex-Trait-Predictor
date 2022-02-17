include("src/PredictionModels.jl")
include("src/SNPDataSet.jl")
include("src/TestLinearModel.jl")
Random.seed!(1)
# Make ? A DataSet
dataSet = LinearSNPDataSet([300, 400], noiseWeight=0.1) # ([number of data points, number of SNPs])
# Perform K-FoldCrossValidation and print result
@time println(kFoldCrossValidation(dataSet, lassoModel, k=3))


