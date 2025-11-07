import json
import requests
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from PIL import Image
from io import BytesIO
import geopandas as gpd
import sys
from datetime import datetime,timedelta
from io import BytesIO
import sys, os
#import cgi, cgitb
import io
import ssl
import warnings
import xarray as xr
import certifi
import urllib.request
from matplotlib.colors import BoundaryNorm
import pandas as pd
from owslib.wms import WebMapService
import matplotlib.colors as mcolors
import netCDF4 as nc
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from dateutil.relativedelta import relativedelta
from scipy.interpolate import NearestNDInterpolator
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter
from matplotlib import colors

#####FUNCTIONS#####
def fetch_wms_layer_data(layer_id):
    try:
        url_tmp = "https://ocean-middleware.spc.int/middleware/api/layer_web_map/{layerid}/"
        url = url_tmp.format(layerid=layer_id)
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data_dict = response.json()
        
        # Convert the dictionary to an object with attribute access
        class DataObject:
            def __init__(self, data):
                self.__dict__.update(data)
        
        data = DataObject(data_dict)
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return None


def getfromDAP(url, target_time, variable_name, adjust_lon=False, local_path=False, local_path_str=None):
    
    try:
        # Choose data source
        data_source = local_path_str if local_path and local_path_str else url

        with xr.open_dataset(data_source, engine='netcdf4', mask_and_scale=True, decode_cf=True) as ds:
            # Extract variable data
            if variable_name not in ds.variables:
                available_vars = list(ds.variables.keys())
                raise ValueError(f"Variable '{variable_name}' not found. Available variables: {available_vars}")

            # Check if "time" dimension exists
            if "time" in ds.dims or "time" in ds.coords:
                # Get available times (handle bytes if needed)
                if isinstance(ds.time.values[0], bytes):
                    time_str = [t.decode('utf-8') for t in ds.time.values]
                    time_dt = np.array([datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ") for t in time_str])
                else:
                    time_dt = [pd.to_datetime(t).to_pydatetime() for t in ds.time.values]

                # Convert target time to datetime object
                target_dt = datetime.strptime(target_time, "%Y-%m-%dT%H:%M:%SZ")
                # Find closest time index
                time_index = np.argmin([abs((t - target_dt).total_seconds()) for t in time_dt])

                data = ds[variable_name].isel(time=time_index)
            else:
                # No "time" in dims or coords, just take the first slice
                data = ds[variable_name]
                # If variable has more than 2 dimensions, select the first "frame" along the first axis
                if len(data.shape) > 2:
                    data = data.isel({data.dims[0]: 0})

            # If variable has 3 dimensions (e.g., depth), select first depth level
            if len(data.dims) == 3:
                data = data.isel({data.dims[0]: 0})  # Select first index of first dimension

            # Determine coordinate names
            coord_names = {
                'lon': ['lon', 'longitude', 'x', 'X'],
                'lat': ['lat', 'latitude', 'y', 'Y']
            }
            lon_name = next((n for n in coord_names['lon'] if n in ds.coords), None)
            lat_name = next((n for n in coord_names['lat'] if n in ds.coords), None)
            if lon_name is None:
                raise ValueError("Could not identify longitude coordinate variable")
            if lat_name is None:
                raise ValueError("Could not identify latitude coordinate variable")

            # Get coordinates
            lon = ds[lon_name].values
            lat = ds[lat_name].values

            # Adjust longitude if requested (for 180° crossing)
            if adjust_lon:
                if np.any(lon < 0):
                    lon = np.where(lon < 0, lon + 360, lon)

            # Extract and mask data values
            data_extract = np.ma.masked_invalid(data.values.squeeze())
            return lon, lat, data_extract

    except Exception as e:
        if local_path:
            raise RuntimeError(f"Error accessing local NetCDF data: {str(e)}")
        else:
            raise RuntimeError(f"Error accessing OpenDAP data: {str(e)}")

def getCountryData(country_id):
    # Fetch bounding box data from API
    region_url_prefix = "https://ocean-middleware.spc.int/middleware/api/country/"
    api_url = "%s%s/" % (region_url_prefix,str(country_id))
    response = requests.get(api_url)
    west_bound, east_bound, south_bound, north_bound,country_name = "", "", "","",""
    name = ""
    if response.status_code == 200:
        data = response.json()
        name = data['short_name']
    else:
        print(f"Failed to retrieve bounding box data. Status code: {response.status_code}")
    if name == "PAC":
        name = "PAC_EEZ_v3"
    eez_url = "https://opmgeoserver.gem.spc.int/geoserver/spc/wfs?service=WFS&version=2.0.0&request=GetFeature&typeNames=spc:{layername}&srsName=EPSG:4326&outputFormat=application/json"
    formatted_url = eez_url.format(layername=name)
    return formatted_url

def getBBox(country_id):
    region_url_prefix = "https://ocean-middleware.spc.int/middleware/api/country/"
    # Fetch bounding box data from API
    api_url = "%s%s/" % (region_url_prefix,str(country_id))
    response = requests.get(api_url)
    west_bound, east_bound, south_bound, north_bound,country_name = "", "", "","",""

    if response.status_code == 200:
        data = response.json()
        west_bound = data['west_bound_longitude']
        east_bound = data['east_bound_longitude']
        south_bound = data['south_bound_latitude']
        north_bound = data['north_bound_latitude']
        country_name = data['long_name']
        short_name = data['short_name']
    else:
        print(f"Failed to retrieve bounding box data. Status code: {response.status_code}")
    
    return west_bound, east_bound, south_bound, north_bound, country_name,short_name

def cm2inch(*tupl):
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple()

def add_z_if_needed(s):
    if len(s) == 0:
        return 'Z'  # or just return s if you want empty string to remain empty
    if s[-1] != 'Z':
        return s + 'Z'
    return s

def get_config_variables():
    config = {
        
        # Additional configuration variables
        "copyright_text": "© Pacific Community (SPC) 2025",
        "footer_text": "Climate and Ocean Support Program in the Pacific (COSPPac)",
        "app_name": "Ocean Data Viewer",
        "version": "1.2.0",
        "default_theme": "light",
        "max_upload_size": 10,  # in MB
        "supported_formats": ["geojson", "shapefile", "netcdf", "csv"]
    }
    
    # Convert the dictionary to an object with dot notation access
    class ConfigObject:
        def __init__(self, data):
            for key, value in data.items():
                if isinstance(value, dict):
                    setattr(self, key, ConfigObject(value))
                else:
                    setattr(self, key, value)
    
    return ConfigObject(config)

def demo_time(layer_map_data):
    try:
        if layer_map_data.has_specific_timestep:
            spec = layer_map_data.specific_timestemps
            specsplit = spec.split(',')
            time = specsplit[0]
        else:
            time = layer_map_data.timeIntervalEnd
    except Exception as e:
        time = layer_map_data.timeIntervalEnd
    time2 = add_z_if_needed(time)
    return time2

def get_dap_config(layer_map_data):
    dap_url = layer_map_data.url.replace("wms", "dodsC")
    dap_variable = layer_map_data.layer_name
    dapvaribsplit = dap_variable.split(',')
    #if len(dapvaribsplit) >= 1:
    #    dap_variable = dapvaribsplit[0]
    return dap_url, dap_variable

def get_title(layer_map_data,time):
    new_name = []
    week = False
    date = datetime.strptime(add_z_if_needed(time), "%Y-%m-%dT%H:%M:%SZ")
    date2 = date.strftime("%Y-%m-%dT%H%M%SZ")
    formatted_date = date.strftime("%-d %B %Y")
    orig_name = layer_map_data.get_map_names
    if "{week}" in layer_map_data.get_map_names:
        spec = layer_map_data.specific_timestemps
        specsplit = spec.split(',')
        specsplit = [s.replace(" ", "") for s in specsplit]
        interval = layer_map_data.interval_step
        intsplot = interval.split(',')

        cleaned_text = time.replace("Z", "")
        index = specsplit.index(cleaned_text)
        new_nametmp = layer_map_data.get_map_names.replace("{week}", "%s Week"%(intsplot[index]))
        new_name = new_nametmp.split('/')
        week = True

    title_suffix = "Daily Average Sea Surface Temperature Anomaly: %s" % (formatted_date)
    dataset_text = "Reynolds SST"

    if layer_map_data.get_map_names != None or layer_map_data.get_map_names != "":
        if "new-line" in layer_map_data.get_map_names:
            newliner = layer_map_data.get_map_names.split('new-line')
            layer_map_data.get_map_names = newliner[0].split('/')
            formatted_date = date.strftime(layer_map_data.get_map_names[1])
            if week:
                title_suffix = "%s: %s" % (new_name[0],formatted_date)
            else:
                title_suffix = "%s: %s" % (layer_map_data.get_map_names[0],formatted_date)
            
            dataset_text = "%s \n%s" % (layer_map_data.get_map_names[2], newliner[1])
        else:
            layer_map_data.get_map_names = layer_map_data.get_map_names.split('/')
            formatted_date = date.strftime(layer_map_data.get_map_names[1])
            if week:
                title_suffix = "%s: %s" % (new_name[0],formatted_date)
            else:
                title_suffix = "%s: %s" % (layer_map_data.get_map_names[0],formatted_date)
            dataset_text = layer_map_data.get_map_names[2]
    if "{week}" not in orig_name:
        if '{' in layer_map_data.get_map_names[0] and '}' in layer_map_data.get_map_names[0]:
            if "Anomalies" in layer_map_data.get_map_names[0]:
                cleaned = layer_map_data.get_map_names[0].replace('{', '').replace('}', '')
                formatted_date = date.strftime(layer_map_data.get_map_names[1])
                date_str = layer_map_data.get_map_names[1]
                start_date = date
                end_date = start_date + relativedelta(months=2)
                formatted_range = f"{start_date.strftime('%b %Y')} - {end_date.strftime('%b %Y')}"
                title_suffix = "%s : %s" % (cleaned, formatted_range)
            else:    
                cleaned = layer_map_data.get_map_names[0].replace('{', '').replace('}', '')
                if "Seasonal" in cleaned:
                    formatted_date = date.strftime(layer_map_data.get_map_names[1])
                    date_str = layer_map_data.get_map_names[1]
                    start_date = date - relativedelta(months=1)
                    end_date = start_date + relativedelta(months=2)
                    formatted_range = f"{start_date.strftime('%b')} - {end_date.strftime('%b %Y')}"
                    title_suffix = "%s : %s" % (cleaned, formatted_range)
                
                else:
                    formatted_date = date.strftime(layer_map_data.get_map_names[1])
                    date_str = layer_map_data.get_map_names[1]
                    start_date = date
                    end_date = start_date + relativedelta(months=2)
                    formatted_range = f"{start_date.strftime('%b')} - {end_date.strftime('%b %Y')}"
                    title_suffix = "%s : %s" % (cleaned, formatted_range)
    weekly_split = orig_name.split('/')
    if "{8week}" in weekly_split[1]:
        name = weekly_split[1].replace("{8week}", "")
        dataset_text =  weekly_split[2]
        formatted_date = date.strftime("%-d %B %Y")
        future_date = date + timedelta(days=7)
        formatted_future_date = future_date.strftime("%-d %B %Y")
        title_suffix = "%s: %s - %s" % ( weekly_split[0],formatted_date,formatted_future_date)

    return title_suffix, dataset_text

def get_plot_config(layer_map_data):
    plotter_config = layer_map_data.get_map_url.split("/")
    plot_type = plotter_config[0]
    cmap_name = plotter_config[1]
    min_color_plot = float(plotter_config[2])
    max_color_plot = float(plotter_config[3])
    steps = float(plotter_config[4])
    units = plotter_config[5]
    levels = []
    if plotter_config[6] != "null":
        levels = np.array(eval(plotter_config[6]), dtype=float)
    discrete = plotter_config[7]
    
    return cmap_name, plot_type, min_color_plot,max_color_plot,steps,units,levels,discrete

def plot_map_grid(m, south_bound, north_bound, west_bound, east_bound,region):
    """
    Draw latitude/longitude grid lines with automatic spacing:
    - Uses 2° spacing for small plots, larger spacing for bigger plots
    - Doesn't start/end exactly at the edges
    - Labels only on left (lat) and bottom (lon)
    - Clean single degree symbols
    """
    # Calculate plot dimensions
    lat_span = north_bound - south_bound
    lon_span = east_bound - west_bound
    #PACIFC - more tha 80, 
    if lat_span > 20 and lat_span <30:
        spacing = 4
    elif lat_span >80 and lat_span <120:
        spacing = 20
    else:
        spacing = 2

    """
    if int(region) == 1:
         spacing = 20
    elif int(region) == 6 or int(region) == 22:
        spacing = 4
    else:
        spacing = 2
    """
    # Calculate grid lines (not starting exactly at edges)
    # Round bounds outward to nearest spacing multiple
    def round_outward(value, step):
        return np.floor(value / step) * step if value >= 0 else np.ceil(value / step) * step
    
    first_parallel = round_outward(south_bound + 0.5, spacing)
    last_parallel = round_outward(north_bound - 0.5, spacing)
    first_meridian = round_outward(west_bound + 0.5, spacing)
    last_meridian = round_outward(east_bound - 0.5, spacing)
    
    parallels = np.arange(first_parallel, last_parallel + spacing, spacing)
    meridians = np.arange(first_meridian, last_meridian + spacing, spacing)
    
    # Draw primary grid lines
    m.drawparallels(parallels,
                    labels=[1, 0, 0, 0],
                    fmt='%.0f',  # Directly include degree symbol
                    fontsize=6,
                    color='grey',
                    linewidth=0.5,
                    dashes=[1, 1])
    
    m.drawmeridians(meridians,
                    labels=[0, 0, 0, 1],
                    fmt='%.0f',  # Directly include degree symbol
                    fontsize=6,
                    color='grey',
                    linewidth=0.5,
                    dashes=[1, 1])
    
    # Add secondary grid lines (half spacing) without labels
    if spacing > 2:  # Only add if primary spacing isn't already small
        secondary_spacing = spacing / 2
        secondary_parallels = np.arange(round_outward(south_bound, secondary_spacing),
                                      round_outward(north_bound, secondary_spacing) + secondary_spacing,
                                      secondary_spacing)
        secondary_meridians = np.arange(round_outward(west_bound, secondary_spacing),
                                       round_outward(east_bound, secondary_spacing) + secondary_spacing,
                                       secondary_spacing)
        
        m.drawparallels(secondary_parallels,
                        labels=[0, 0, 0, 0],
                        color='lightgrey',
                        linewidth=0.2,
                        dashes=[1, 1])
        
        m.drawmeridians(secondary_meridians,
                        labels=[0, 0, 0, 0],
                        color='lightgrey',
                        linewidth=0.2,
                        dashes=[1, 1])


def plot_filled_contours_no_zero_levels(is_imperial_layer,
        ax, ax_legend, lon, lat, data, 
        min_color_plot=None, max_color_plot=None, steps=None,
        cmap_name='RdBu_r', units='(°C)', levels=None, white_color=(1, 1, 1, 1)):

    base = plt.get_cmap(cmap_name)

    # DEBUG: Check what type of data we're getting
    print(f"Data type: {type(data)}")
    print(f"Data shape: {data.shape}")
    
    # Ensure data is properly masked - handle both regular arrays and masked arrays
    if hasattr(data, 'mask'):
        # Data is already a masked array
        data_masked = data
        print(f"Input data is already masked: {np.sum(data_masked.mask)} masked points")
    else:
        # Convert to masked array, properly handling NaN values
        print(f"Input data is regular array, checking for NaN values...")
        data_masked = np.ma.masked_invalid(data)
        print(f"After masking: {np.sum(data_masked.mask)} masked points")
    
    # Additional check: look for very large/small values that might be fill values
    if np.sum(data_masked.mask) == 0:
        print("Warning: No masked values found, but expecting some near coastlines")
        # Try alternative masking for very large/small values that might represent fill values
        data_range = np.nanmax(data) - np.nanmin(data)
        if data_range > 1000:  # Unusually large range might indicate fill values
            print("Large data range detected, checking for extreme values...")
            extreme_mask = (np.abs(data) > 1e10) | (data == -9999) | (data == 9999)
            if np.any(extreme_mask):
                data_masked = np.ma.masked_where(extreme_mask, data)
                print(f"Found {np.sum(extreme_mask)} extreme values to mask")

    print(f"Final masked points: {np.sum(data_masked.mask)} out of {data_masked.size}")
    print(f"Valid data range: [{np.min(data_masked):.3f}, {np.max(data_masked):.3f}]")

    # If levels not provided, derive from range (excluding zero)
    if levels is None:
        if steps is None or min_color_plot is None or max_color_plot is None:
            raise ValueError("Provide levels or all of min_color_plot, max_color_plot, and steps.")
        levels = np.arange(min_color_plot, max_color_plot, steps, dtype=float)
        levels = levels[levels != 0]

    levels = np.asarray(levels, dtype=float)

    # Basic validations
    if levels.ndim != 1 or len(levels) < 2:
        raise ValueError("levels must be a 1D array with at least two ascending values.")
    if not np.all(np.diff(levels) > 0):
        raise ValueError("levels must be strictly increasing.")

    n_intervals = len(levels) - 1

    # Detect a single "gap across zero": levels[i] < 0 < levels[i+1]
    cross_idxs = np.where((levels[:-1] < 0) & (levels[1:] > 0))[0]

    # Build interval colors
    interval_colors = None
    if cross_idxs.size == 1:
        # There is exactly one central gap across zero
        center_idx = int(cross_idxs[0])
        n_neg = center_idx                     # intervals fully below 0
        n_pos = n_intervals - center_idx - 1   # intervals fully above 0

        # Sample blue-ish side for negatives, red-ish for positives from RdBu_r
        neg_samples = np.linspace(0.05, 0.45, n_neg) if n_neg > 0 else np.array([])
        pos_samples = np.linspace(0.55, 0.95, n_pos) if n_pos > 0 else np.array([])

        colors_neg = [base(v) for v in neg_samples]
        colors_pos = [base(v) for v in pos_samples]

        # Central band white
        interval_colors = colors_neg + [white_color] + colors_pos

    else:
        # No gap across zero (all-negative or all-positive) or ambiguous.
        # Just map all intervals to one side of the palette to avoid the neutral center.
        all_pos = np.all(levels >= 0)
        all_neg = np.all(levels <= 0)

        if all_pos:
            samples = np.linspace(0.55, 0.95, n_intervals)  # reds
        elif all_neg:
            samples = np.linspace(0.05, 0.45, n_intervals)  # blues
        else:
            # If ambiguous (e.g., 0 inside levels), distribute across both sides but no white band.
            # Split at 0 index if present, otherwise fall back to full span.
            if np.any(np.isclose(levels, 0.0)):
                zero_idx = int(np.where(np.isclose(levels, 0.0))[0][0])
                n_neg = zero_idx
                n_pos = n_intervals - zero_idx
                neg_samples = np.linspace(0.05, 0.45, n_neg) if n_neg > 0 else np.array([])
                pos_samples = np.linspace(0.55, 0.95, n_pos) if n_pos > 0 else np.array([])
                interval_colors = [base(v) for v in neg_samples] + [base(v) for v in pos_samples]
            else:
                samples = np.linspace(0.05, 0.95, n_intervals)

        if interval_colors is None:
            interval_colors = [base(v) for v in samples]

    # Add under/over colors since we use extend='both'
    under_color = base(0.0)
    over_color = base(1.0)
    colors_list = [under_color] + interval_colors + [over_color]  # length == len(levels) + 1

    # Build discrete cmap/norm with correct extend
    cmap, norm = colors.from_levels_and_colors(levels, colors_list, extend='both')

    # CRITICAL: Set the bad color to transparent for masked values
    cmap.set_bad(alpha=0.0)  # Make masked values completely transparent

    # Plot with masked data - use the properly masked array
    cs = ax.contourf(
        lon, lat, data_masked,  # Use the masked data
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend='both'
    )

    # Create colorbar with Fahrenheit labels if it's an imperial layer
    if is_imperial_layer and 'C' in units:
        # Convert Celsius levels to Fahrenheit for display
        # CORRECT FORMULA: F = (C × 9/5) + 32
        def celsius_to_fahrenheit(c):
            return (c * 9/5) + 32
        
        # Create Fahrenheit tick labels
        fahrenheit_levels = [celsius_to_fahrenheit(level) for level in levels]
        
        # Format the labels with appropriate precision
        fahrenheit_labels = []
        for f_val in fahrenheit_levels:
            if f_val == 0:
                fahrenheit_labels.append("0")
            elif abs(f_val) < 1:
                fahrenheit_labels.append(f"{f_val:.2f}")
            elif abs(f_val) < 10:
                fahrenheit_labels.append(f"{f_val:.1f}")
            else:
                fahrenheit_labels.append(f"{f_val:.0f}")
        
        # Create colorbar with Fahrenheit labels
        cbar = plt.colorbar(cs, cax=ax_legend)
        cbar.set_ticks(levels)  # Keep the original Celsius levels for positioning
        cbar.set_ticklabels(fahrenheit_labels)  # But show Fahrenheit values
        
        # Update units label
        units_label = '(°F)'
        cbar.set_label(units_label, fontsize=8, rotation=0, va='center', ha='left', labelpad=1)
        
    else:
        # Regular Celsius colorbar
        cbar = plt.colorbar(cs, cax=ax_legend)
        cbar.set_ticks(levels)
        cbar.set_label(units, fontsize=8, rotation=0, va='center', ha='left', labelpad=1)

    # Style the colorbar
    cbar.ax.tick_params(labelsize=8, pad=2, direction='out', length=6, width=1)

    try:
        cbar.solids.set_edgecolor("face")
    except Exception:
        pass

    return cs, cbar

#####FUNCTIONS#####

##NEWFUNCTIONSSS

def get_layer_dataset_download_info(layer_id, time=None, root_dir=None, mapper_filename='layer_dataset_mapper.json'):
    """
    Given a layer_id, optional time, and optional root_dir, reads the mapper file for dataset_id,
    queries the dataset API, and returns:
        - path: local_directory_path (with {root-dir} replaced if root_dir is given)
        - file_name: download_file_prefix + download_file_infix + download_file_suffix
    If download_file_infix contains % (strftime), uses 'time' to fill it in.
    If layer_id does not exist in the mapping, returns 0 and does not execute further.
    """
    # Convert layer_id to string for mapping lookup
    layer_id_str = str(layer_id)
    
    # Read the mapping file from the same directory as this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mapper_path = os.path.join(current_dir, mapper_filename)
    
    with open(mapper_path, 'r') as f:
        mapping = json.load(f)
    
    # Get the dataset_id for the given layer_id
    dataset_id = mapping.get(layer_id_str)
    if not dataset_id:
        return 0  # Immediately return 0 and DO NOT execute further
    
    # Query the API for this dataset_id
    url = f"https://ocean-middleware.spc.int/middleware/api/dataset/{dataset_id}/?format=json"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()

    # Extract the required fields
    prefix = data["download_file_prefix"]
    infix = data["download_file_infix"]
    suffix = data["download_file_suffix"]
    local_directory_path = data["local_directory_path"]
    if layer_id == "26": 
        infix = "%Y%m_%Y%m"
    elif layer_id == "36":
        infix = "%Y%m_%Y%m"
        suffix = ".nc"
    elif layer_id == "37":
        infix = "decile.%Y%m"
        suffix = ".nc"
    elif layer_id == "35" or layer_id == "39":
        infix = "%Y%m"
        suffix = ".nc"
    elif layer_id == "8":
        infix = "%Y%m%d_%Y%m%d"

    

    # Prepare file name
    if "%" in infix:
        if "_" in infix and "AQUA" in infix:
            first_fmt, last_fmt = infix.split("_", 1)
            # Parse the base date
            dt = datetime.strptime(time, "%Y-%m-%dT%H:%M:%SZ")
            # First day of month
            first_day = dt.replace(day=1)
            # Last day of month: go to next month, subtract 1 day
            if dt.month == 12:
                next_month = dt.replace(year=dt.year + 1, month=1, day=1)
            else:
                next_month = dt.replace(month=dt.month + 1, day=1)
            last_day = next_month - timedelta(days=1)
            # Format
            infix_formatted = f"{first_day.strftime(first_fmt)}_{last_day.strftime(last_fmt)}"
        elif not time:
            raise ValueError("Time must be provided for infix formatting.")
        # Parse time string like "2025-10-16T12:00:00Z"
        elif layer_id == "36": 
            first_fmt, last_fmt = infix.split("_", 1)
            # Parse the base date
            # Parse the base date
            dt = datetime.strptime(time, "%Y-%m-%dT%H:%M:%SZ")
            # First day of current month
            first_day = dt.replace(day=1)

            # Calculate the first day of the month two months ahead
            if first_day.month > 10:
                # December or November
                year = first_day.year + 1
                month = (first_day.month + 2) % 12
                if month == 0: month = 12
            else:
                year = first_day.year
                month = first_day.month + 2
            next2_month = first_day.replace(year=year, month=month, day=1)
            # Last day is the last day of that month (go to next month, subtract 1 day)
            if month == 12:
                month3 = 1
                year3 = year + 1
            else:
                month3 = month + 1
                year3 = year
            month3_first = next2_month.replace(year=year3, month=month3, day=1)
            last_day = month3_first - timedelta(days=1)

            # Format
            infix_formatted = f"{first_day.strftime(first_fmt)}_{last_day.strftime(last_fmt)}"
        else:
            dt = datetime.strptime(time, "%Y-%m-%dT%H:%M:%SZ")
            infix_formatted = dt.strftime(infix)
    elif "none" in infix:
        infix_formatted = ""
        suffix = ""
    else:
        infix_formatted = infix

    file_name = f"{prefix}{infix_formatted}{suffix}"
    if not file_name.endswith('.nc'):
        file_name += '.nc'
    if layer_id == "16" or layer_id == "6" or layer_id == "27" or layer_id == "29":
        file_name = 'latest.nc'
    if layer_id == "2" or layer_id =="10" or layer_id =="11" or layer_id =="12" or layer_id =="14":
        file_name = 'latest_merged.nc'
    if layer_id == "19":
        file_name = 'latest_merged.nc'
    if layer_id == "47":
        file_name = 'sst_trend.nc'
    if layer_id == "8":
        first_fmt, last_fmt = infix.split("_", 1)
        # Parse the base date
        dt = datetime.strptime(time, "%Y-%m-%dT%H:%M:%SZ")
        # First day of month
        first_day = dt.replace(day=1)
        # Last day of month: go to next month, subtract 1 day
        if dt.month == 12:
            next_month = dt.replace(year=dt.year + 1, month=1, day=1)
        else:
            next_month = dt.replace(month=dt.month + 1, day=1)
        last_day = next_month - timedelta(days=1)
        # Format
        infix_formatted = f"{first_day.strftime(first_fmt)}_{last_day.strftime(last_fmt)}"
        file_name = f"AQUA_MODIS."+infix_formatted+".L3m.MO.CHL.chlor_a.4km.NRT.nc.dap.nc"
    if layer_id == "41":
        def get_weekly_filename(time_str):
            """
            Given a time string, returns the AQUA_MODIS 8-day composite filename
            based on the custom start date 2025-05-25.
            """
            # Reference start and end date from your first dataset
            ref_start = datetime(2025, 5, 25)
            dt = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ")
            days_since_ref = (dt - ref_start).days
            period_index = days_since_ref // 8
            # Handle dates before the reference period
            if days_since_ref < 0:
                raise ValueError("Date is before the first available dataset period.")
            start_dt = ref_start + timedelta(days=period_index * 8)
            end_dt = start_dt + timedelta(days=7)
            fname = f"AQUA_MODIS.{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.L3m.8D.CHL.chlor_a.4km.NRT.nc"
            return fname

        file_name = get_weekly_filename(time)
    if layer_id == "26":
        local_directory_path = "{root-dir}/model/regional/copernicus/hindcast/monthly/ssh"
    if layer_id == "35" or layer_id == "39":
        local_directory_path = "{root-dir}/model/regional/noaa/hindcast/monthly/sst_anomalies"
    if layer_id == "36":
        local_directory_path = "{root-dir}/model/regional/noaa/hindcast/3monthly/sst_anomalies"
    if layer_id == "37":
        local_directory_path = "{root-dir}/model/regional/noaa/hindcast/decile/sst_anomalies"
    if layer_id == "47":
        local_directory_path = "{root-dir}/model/regional/noaa/hindcast/trend"
    if layer_id == "2" or layer_id == "10" or layer_id =="11" or layer_id =="12" or layer_id =="14":
        local_directory_path = "{root-dir}/model/regional/bom/forecast/hourly/wavewatch3_latest"
    # Replace {root-dir} if root_dir is supplied
    if root_dir:
        path = local_directory_path.replace("{root-dir}", root_dir)
    else:
        path = local_directory_path

    return {
        "path": path,
        "file_name": file_name
    }

def imperial_layers(layer_id):
    vbool = False
    if layer_id == 5:
        vbool = True
    return vbool

##INIT##
config = get_config_variables()

#####PARAMETER#####
region = 2
layer_id = 5
#time= add_z_if_needed("2024-10-01T00:00:00Z")
resolution = "l"
#####PARAMETER#####

layer_map_data = fetch_wms_layer_data(layer_id)

#REMOVE DEMO
time = demo_time(layer_map_data)
time = "2025-05-25T00:00:00Z"
unit_type = "imperial"
is_imperial_layer = imperial_layers(layer_id)
if is_imperial_layer:
    if unit_type == "imperial":
        is_imperial_layer = True
    else:
        is_imperial_layer = False
#time = "2025-05-01T00:00:00Z"
#time = "2025-08-05T00:00:00Z"
#SLA Daily
#time = "2025-07-16T00:00:00Z"
#SLA MONTHLY
#time = "2025-05-01T00:00:00Z"
#BOMM
#time = "2025-09-16T00:00:00Z"
#REMOVE DEMO
##TRY TO GET DATASET
info = get_layer_dataset_download_info(str(layer_id),time,'/Users/anujdivesh/Desktop/django/production')
check_local = True
local_file_name = ""
if info == 0:
    check_local = False
else:
    local_file_name = "%s/%s" % (info['path'], info['file_name'])
    check_local = True

print(check_local)
print(local_file_name)
#sys.exit()
##

#####MAIN#####
dap_url, dap_variable = get_dap_config(layer_map_data)
title, dataset_text = get_title(layer_map_data,time)
print(dataset_text)
cmap_name, plot_type, min_color_plot, max_color_plot, steps, units, levels, discrete = get_plot_config(layer_map_data)
west_bound, east_bound, south_bound, north_bound, country_name, short_name = getBBox(region)
if short_name == "PAC":
    resolution = "h"
eez_url = getCountryData(region)

#MAPPING


figsize = cm2inch((15,13))
fig, ax = plt.subplots(figsize=figsize, dpi=300)
ax.axis('off')

ax2 = fig.add_axes([0.09, 0.2, 0.8, 0.65])
title = "%s \n %s" % (country_name,title)
ax2.set_title(title, pad=10, fontsize=8)

m = Basemap(projection='cyl', llcrnrlat=south_bound, urcrnrlat=north_bound, 
            llcrnrlon=west_bound, urcrnrlon=east_bound, resolution=resolution, ax=ax2)

plot_map_grid(m, south_bound, north_bound, west_bound, east_bound,region)

# Add colorbar to ax2
ax2_pos = ax2.get_position()
ax_legend_width = 0.03  # Width of the legend
ax_legend_gap = 0.1    # Gap between ax2 and ax_legend
if plot_type != "ugrid_9":
    ax_legend = fig.add_axes([ax2_pos.x1 +0.02, ax2_pos.y0, ax_legend_width, ax2_pos.height])

if plot_type == "contourf_nozero_levels":
    lon, lat, data_extract = getfromDAP(dap_url, time, dap_variable,adjust_lon=True,\
    local_path=check_local, local_path_str=local_file_name)

    cs, cbar = plot_filled_contours_no_zero_levels(is_imperial_layer,ax=ax2, ax_legend=ax_legend, lon=lon, lat=lat, data=data_extract,\
        min_color_plot=min_color_plot, max_color_plot=max_color_plot, steps=steps, cmap_name=cmap_name, units=units,levels=levels
    )

if cbar is not None and plot_type != "discrete":
    if is_imperial_layer:
        # For imperial layers - DON'T use FormatStrFormatter
        # Just adjust visual properties
        for t in cbar.ax.get_yticklabels():
            t.set_horizontalalignment('left')
        cbar.ax.tick_params(axis='y', pad=2, length=0)
    else:
        if layer_id == 29:
            #cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%6.1f'))
            for t in cbar.ax.get_yticklabels():
                t.set_horizontalalignment('left')
            cbar.ax.tick_params(axis='y', pad=4, length=0)
        else:
            if layer_id == 16:
                cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%6.1f'))
                for t in cbar.ax.get_yticklabels():
                    t.set_horizontalalignment('left')
                cbar.ax.tick_params(axis='y', pad=2, length=0)
            else:
                cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%6.1f'))
                for t in cbar.ax.get_yticklabels():
                    t.set_horizontalalignment('left')
                cbar.ax.tick_params(axis='y', pad=-1, length=0)


print("Plotting EEZ and Coastline...")


plt.savefig('anuj3.png', bbox_inches='tight', pad_inches=0.1,dpi=300)
