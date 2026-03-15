using DataTreatments

using SoleData: Artifacts

# fill your Artifacts.toml file;
# Artifacts.fillartifacts()

natopsloader = Artifacts.NatopsLoader()
Xts, yts = Artifacts.load(natopsloader)

dt = DataTreatment(Xts, yts)

test1 = get_dataset(dt)
get_dataset(dt, dataframe=true)