using Test
using DataTreatments

using Statistics

X = rand(100)
wfunc = splitwindow(nwindows=10)
intervals = @evalwindow X wfunc
result = applyfeat(X, intervals)

X = rand(100, 120)
wfunc = splitwindow(nwindows=3)
intervals = @evalwindow X wfunc
result = applyfeat(X, intervals; reducefunc=maximum)

X = rand(100, 120)
Xmatrix = fill(X, 100, 10)
wfunc = splitwindow(nwindows=3)
intervals = @evalwindow X wfunc
result = aggregate(Xmatrix, intervals; reducefunc=std)

X = rand(100, 120)
Xmatrix = fill(X, 100, 10)
wfunc = splitwindow(nwindows=3)
intervals = @evalwindow X wfunc
features = (mean, maximum)
result = reducesize(Xmatrix, intervals; features)