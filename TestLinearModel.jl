include("SNPDataSet.jl")
using Statistics
using Plots
using StatsBase
using IterTools
import Base: println

# Computes the rootMeanSquare between expected and observed data
function rootMeanSquare(actualTestData::Vector{Float64}, predictedTestData::Vector{Float64})
    # Root mean squre error
    sum_squre_of_error = 0
    for i in range(1, length(actualTestData))
        sum_squre_of_error += ((actualTestData[i] - predictedTestData[i])^2)
    end
    rootMeanSquare = sqrt(sum_squre_of_error/length(actualTestData))
    return rootMeanSquare
end

# Computes pearsonsCorrelation between expected and observed data
function pearsonsCorrelation(actualTestData::Vector{Float64}, predictedTestData::Vector{Float64})
    actualTestDataMean = mean(actualTestData)
    predictedTestDataMean = mean(predictedTestData)
    upperSum = 0
    for i in range(1,length(actualTestData))
        upperSum += (actualTestData[i] - actualTestDataMean) * (predictedTestData[i] - predictedTestDataMean)
    end
        
    ∑differncesActualTestDataFromMeanSquare = sum(x -> (x - actualTestDataMean)^2, actualTestData)
    ∑differncesPredictedDataFromMeanSqure =  sum(x -> (x - predictedTestDataMean)^2, predictedTestData)

    r = upperSum/sqrt(∑differncesActualTestDataFromMeanSquare * 
        ∑differncesPredictedDataFromMeanSqure)
    
    if isnan(r)
        r = 0
    end

    
    return r
end

# Computes spearmanCorrelation between expected and observed data
function spearmanCorrelation(actualTestData::Vector{Float64}, predictedTestData::Vector{Float64})
    # could not be bothered with manual implementation
    result = corspearman(actualTestData, predictedTestData)
    if result == NaN || isnan(result)
        result = 0
    end

    return result
end

# Data structure holding multiple corrlation messures for expected and observed
struct MeasureCorrelation
    rootMeanSquare
    pearsons
    spearman
    rSquared
    function MeasureCorrelation(rootMeanSquare, pearsons, spearman, rSquared)
        new(rootMeanSquare, pearsons, spearman, rSquared)
    end
    function MeasureCorrelation(X::Vector{Float64}, Y::Vector{Float64})
        new(rootMeanSquare(X,Y), pearsonsCorrelation(X,Y), spearmanCorrelation(X,Y), 
            pearsonsCorrelation(X,Y)^2)
    end
    
end

# Prints MeasureCorrelation to be more human readable
function println(correlation::MeasureCorrelation)
    println("CORRELATION MEASURE\n-------------------")
    println("Root Mean Square:     ",correlation.rootMeanSquare)
    println("Pearsons:             ",correlation.pearsons)
    println("Spearman:             ",correlation.spearman)
    println("rSquared:             ",correlation.rSquared)
end

# Computes the average of MeasureCorrelation measures
function averageCorrelation(correlations::Vector{Any})::MeasureCorrelation
    numberOfCorrelations = length(correlations)
    println(correlations)
    rootMeanSquare = sum(x -> x.rootMeanSquare, correlations)/numberOfCorrelations
    pearsons = sum(x -> x.pearsons, correlations)/numberOfCorrelations
    spearman = sum(x -> x.spearman, correlations)/numberOfCorrelations
    rSquared = sum(x -> x.rSquared, correlations)/numberOfCorrelations
    return MeasureCorrelation(rootMeanSquare, pearsons, spearman, rSquared)
end

# Performs Kfold Cross validation given a data set and a model, returns the average
# corrolation of all k tests
function kFoldCrossValidation(dataSet::LinearSNPDataSet, model::Function; k=10)::MeasureCorrelation
    ###### Split dataset into k groups ######
    numberOfDataPoints = size(dataSet.SNPMatrix)[1]
    randomIndexes = shuffle([i for i in range(1, numberOfDataPoints)])
    indexGroups = [[] for i in range(1,k)]

    # Divide randomIndexes across indexGroups equally
    for (i, indexGroup) in enumerate(Iterators.cycle(indexGroups))
        append!(indexGroup, randomIndexes[i])
        if i == numberOfDataPoints && break
        end
    end

    # Makes k randomised disjoint sub data sets from dataSet
    subDataSets = [LinearSNPDataSet(dataSet, indexGroup) for indexGroup in indexGroups]


    # For each given (k) subdataset, train other data set and test agaist given dataset.
    predictionActualPairs = []
    for (i, testDataSet) in enumerate(subDataSets)
        for (j, trainDataSet) in enumerate(subDataSets)
            if i == j continue end
            ß = model(trainDataSet)
           
            # this is nessary as testDataSet is not mutable
            testDataSet_SNPMatrix = testDataSet.SNPMatrix

            # a column of ones must be added if lasso model is used
            if model == lassoModel
                testDataSet_SNPMatrix = hcat(ones(size(testDataSet.SNPMatrix)[1], 1),   
                    testDataSet.SNPMatrix)
            end

            # Store the predictions, and actual data in pairs to be compaired
            predictionOnTest = testDataSet_SNPMatrix * ß
            append!(predictionActualPairs, [[testDataSet.phenotypes, predictionOnTest]])
        end
    end
    
    # Create a list of correlationMeasures for every test in cross validation
    correlationMeasures = []
    for e in predictionActualPairs
        result = MeasureCorrelation(e[1], e[2]) # (actual data, prediction)
        push!(correlationMeasures, result)
    end
   
    # Average correlation measures
    avCorrelation = averageCorrelation(correlationMeasures)
    return (avCorrelation)
end






    









function TestLinearModel(dataSet::LinearSNPDataSet, ß::Vector{Float64}; testDataSetSize=false, 
    figname="recentPlot")
    # Generate test dateset of specfied size
    if !testDataSetSize
        testDataSetSize = size(dataSet.SNPMatrix)[1]
    end
    testSNPs = Array{Float64}(bitrand(testDataSetSize, size(dataSet.SNPMatrix)[2]))
    actualTestData = testSNPs * dataSet.SNPEffects + dataSet.generateNoiseVector(testDataSetSize)
    predictedTestData = testSNPs*ß
    # Ordered Scatter Plot
    tupples = sort([(actualTestData[i], predictedTestData[i]) for i in range(1, testDataSetSize)])
    testData = [tupples[i][1] for i in range(1, testDataSetSize)]
    prediction = [tupples[i][2] for i in range(1, testDataSetSize)]
    scatter([i for i in range(1, testDataSetSize)], [testData,  prediction])

    # K fold cross validation K mutually exlcuse groups

    # 10: 9 trainsing, 1 test. retrain on each of the 9 and bench on the test. this can be done 10 times
    # into a different order each time.
    savefig(figname)

end

