include("PredictionModels.jl")
include("SNPDataSet.jl")
include("TestLinearModel.jl")
# Make Dataset A DataSet
dataSet = LinearSNPDataSet([600, 1000])

# Perform K-kFoldCrossValidation and print result
@time println(kFoldCrossValidation(dataSet, lassoModel, k=10))