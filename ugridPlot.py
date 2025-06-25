import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# URL of the NetCDF data
url = "https://ocean-thredds01.spc.int/thredds/dodsC/POP/model/regional/noaa/hindcast/monthly/sst_anomalies/oisst-avhrr-v02r01.202501.nc"

# Open the dataset
ds = xr.open_dataset(url)

# Select the 'anom' variable and the timestamp
timestamp = np.datetime64("2025-01-16T12:00:00.000Z")
data = ds['anom'].sel(time=timestamp, method='nearest')
anom = data.squeeze().values

# Get the coordinates
lat = ds['lat'].values
lon = ds['lon'].values

# Define color scale
vmin, vmax, step = -3, 3, 1
levels = np.arange(vmin, vmax + step, step)

# Plot
plt.figure(figsize=(12, 6))
c = plt.contourf(lon, lat, anom, levels=levels, cmap='bwr', vmin=vmin, vmax=vmax, extend='both')
plt.colorbar(c, label='SST Anomaly (°C)', ticks=levels)
plt.title(f'Sea Surface Temperature Anomaly\n{str(timestamp)}')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.show()