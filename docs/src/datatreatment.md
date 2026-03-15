```@meta
CurrentModule = DataTreatments
```

```@docs
DataTreatment
get_dataset(
        dt::DataTreatment,
        treatments::Base.Callable...;
        treatment_ds=true,
        leftover_ds=true,
        matrix=false,
        dataframe=false
    )
```