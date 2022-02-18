using StatsBase, Lasso, Random, Statistics, LinearAlgebra, DelimitedFiles
include("SNPDataSet.jl")

function lassoModel(dataSet::LinearSNPDataSet)::Vector{Float64}
    return coef(fit(LassoModel, dataSet.SNPMatrix, dataSet.phenotypes, α=1))
end

function ridgeModel(dataSet::LinearSNPDataSet)::Vector{Float64}
    return coef(fit(LassoModel, dataSet.SNPMatrix, dataSet.phenotypes, α=0))
end

function elasticNet(dataSet::LinearSNPDataSet)::Vector{Float64}
    return coef(fit(LassoModel, dataSet.SNPMatrix, dataSet.phenotypes, standardize = false))#maxncoef=1000
end


function ridgeModelManual(dataSet::LinearSNPDataSet; λ = 1)::Vector{Float64}
    return inv(((dataSet.SNPMatrix'*dataSet.SNPMatrix) + I * λ)
        * (dataSet.SNPMatrix' * dataSet.phenotypes))
end

function leastSquares(dataSet::LinearSNPDataSet)::Vector{Float64}
    # Least Squares requires a square matrix
    @assert size(LinearSNPDataSet.SNPMatrix)[1] == size(LinearSNPDataSet.SNPMatrix)[2]

    return (inv( tr(dataSet.SNPMatrix) * dataSet.SNPMatrix)
        * (tr(dataSet.SNPMatrix) * dataSet.phenotypes))
end



# glm net