using DataTreatments
const DT = DataTreatments

using SoleData: Artifacts

# fill your Artifacts.toml file;
# Artifacts.fillartifacts()

natopsloader = Artifacts.NatopsLoader()
Xts, yts = Artifacts.load(natopsloader)

dt = DataTreatment(Xts, yts)

test1 = get_dataset(dt)
test2 = get_dataset(dt, dataframe=true)
test3 = get_dataset(
    dt,
    TreatmentGroup(aggrfunc=DT.aggregate(
        features=(mean, maximum),
        win=(DT.adaptivewindow(nwindows=5, overlap=0.4),)
        )),
    dataframe=true
)
test4 = get_dataset(
    dt,
    TreatmentGroup(aggrfunc=DT.reducesize(
        win=(adaptivewindow(nwindows=5, overlap=0.4),)
        )),
    dataframe=true
)

test5 = get_dataset(
    dt,
    TreatmentGroup(
        name_expr=["X[Hand tip r]", "Y[Hand tip r]", "Z[Hand tip r]"],
        aggrfunc=DT.aggregate(win=(adaptivewindow(nwindows=5, overlap=0.4),),),
        groupby=(:vname, :feat))
    )
