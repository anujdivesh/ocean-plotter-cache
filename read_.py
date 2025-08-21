import xarray as xr

# Open NetCDF file
ds = xr.open_dataset("/Users/anujdivesh/Desktop/django/production/model/regional/bom/forecast/monthly/accesss/sst/sst.forecast.anom.monthly.nc")

# Print dataset structure
print(ds)

# Check for time variable
if "time" in ds:
    times = ds["time"].values
    print("All time values:")
    for t in times:
        print(t)
else:
    print("No 'time' variable found in dataset")
