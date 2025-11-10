using Test
using DataTreatments

using Statistics

X = rand(200, 120)
Xmatrix = fill(X, 50, 5)
win = splitwindow(nwindows=3)

@testset "base_set features" begin
    dt = DataTreatment(Xmatrix, :aggregate;
                        vnames=Symbol.("var", 1:5),
                        win=(win,),
                        features=DataTreatments.base_set)
    
    @test size(dt, 1) == 50
    @test length(get_features(dt)) == length(DataTreatments.base_set)
    @test Set(get_features(dt)) == Set(DataTreatments.base_set)
    
    # Check all base features are present
    feature_ids = get_featureid(dt)
    feature_funcs = unique(get_feature.(feature_ids))
    @test maximum in feature_funcs
    @test minimum in feature_funcs
    @test mean in feature_funcs
    @test Statistics.std in feature_funcs
end

@testset "catch9 features" begin
    dt = DataTreatment(Xmatrix, :aggregate;
                        vnames=Symbol.("var", 1:5),
                        win=(win,),
                        features=DataTreatments.catch9)
    
    @test size(dt, 1) == 50
    @test length(get_features(dt)) == length(DataTreatments.catch9)
    
    # Check statistical features
    feature_ids = get_featureid(dt)
    feature_funcs = unique(get_feature.(feature_ids))
    @test maximum in feature_funcs
    @test minimum in feature_funcs
    @test mean in feature_funcs
    @test median in feature_funcs
    @test Statistics.std in feature_funcs
    
    # Check Catch22 features
    @test DataTreatments.stretch_high in feature_funcs
    @test DataTreatments.stretch_decreasing in feature_funcs
    @test DataTreatments.entropy_pairs in feature_funcs
    @test DataTreatments.transition_variance in feature_funcs
end

@testset "catch22_set features" begin
    dt = DataTreatment(Xmatrix, :aggregate;
                        vnames=Symbol.("var", 1:5),
                        win=(win,),
                        features=DataTreatments.catch22_set)
    
    @test size(dt, 1) == 50
    @test length(get_features(dt)) == length(DataTreatments.catch22_set)
    
    # Check a sample of Catch22 features
    feature_ids = get_featureid(dt)
    feature_funcs = unique(get_feature.(feature_ids))
    @test DataTreatments.mode_5 in feature_funcs
    @test DataTreatments.mode_10 in feature_funcs
    @test DataTreatments.embedding_dist in feature_funcs
    @test DataTreatments.acf_timescale in feature_funcs
    @test DataTreatments.periodicity in feature_funcs
    @test DataTreatments.dfa in feature_funcs
end

@testset "complete_set features" begin
    dt = DataTreatment(Xmatrix, :aggregate;
                        vnames=Symbol.("var", 1:5),
                        win=(win,),
                        features=DataTreatments.complete_set)
    
    @test size(dt, 1) == 50
    @test length(get_features(dt)) == length(DataTreatments.complete_set)
    
    # Check basic statistics
    feature_ids = get_featureid(dt)
    feature_funcs = unique(get_feature.(feature_ids))
    @test maximum in feature_funcs
    @test minimum in feature_funcs
    @test mean in feature_funcs
    @test median in feature_funcs
    @test Statistics.std in feature_funcs
    @test cov in feature_funcs
    
    # Check all catch22 features are included
    for feat in DataTreatments.catch22_set
        @test feat in feature_funcs
    end
end
