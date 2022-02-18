include("src/PredictionModels.jl")
include("src/SNPDataSet.jl")
include("src/TestLinearModel.jl")
Random.seed!(1)

noise = 0
numberOfSNPs = 1000
numberOfDataPoints = 800
plots = []
for i in range(1,4)
    for j in range(1,4)
        # GenerateDataSet
        dataSet = LinearSNPDataSet([numberOfDataPoints, numberOfSNPs], noiseWeight=noise)
        plot = kFoldCrossValidation(dataSet, elasticNet, k=10).predictedActualPlot
        title!(plot, "N = " * string(round(noise; digits=2)) * ", D = " 
            * string(numberOfDataPoints) * ", " * string(numberOfSNPs), plot_titlefontsize=0.1)
        push!(plots, plot)
        noise += 2
        println(i,j)
    end
    noise = 0
    numberOfDataPoints = Integer(round(numberOfDataPoints*0.7))
end
for e in plots
    xlims!(e, -30, 30)
    ylims!(e, -45, 45)
    title!("")
end
plot(plots..., layout = (4, 4), legend = false, plot_titlefontsize=0.01, markersize=2)
savefig("recentPlot")


