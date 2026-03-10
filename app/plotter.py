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
from typing import List, Union
from pathlib import Path
import matplotlib
from scipy.interpolate import NearestNDInterpolator
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
import re
matplotlib.use('Agg')  # Non-interactive backend
plt.switch_backend('Agg') 

class Plotter:
    @staticmethod
    def fetch_wms_layer_data(layer_id,token):
        try:
            url_tmp = "https://ocean-middleware.spc.int/middleware/api/layer_web_map/{layerid}/"
            url = url_tmp.format(layerid=layer_id)
            headers = {
                'Authorization': f'Bearer {token}',
            }
            if token != 'null':
                response = requests.get(url,headers=headers)
            else:
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

    @staticmethod
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

    @staticmethod
    def plot_coastline_from_geoserver(ax, m, filepath, style='polygon', simplify_tolerance=0.01):
        """
        Plot coastline polygons from a local shapefile/GeoJSON/zip with proper dateline handling.

        Parameters:
        - ax: matplotlib axis object
        - m: Basemap object (for coordinate transformation)
        - filepath: path to local shapefile, GeoJSON, or zipped shapefile
        - style: 'polygon' for filled polygons or 'line' for just boundaries
        - simplify_tolerance: tolerance for geometry simplification
        """
        try:
            # Detect zipped shapefile
            if filepath.endswith('.zip'):
                gdf = gpd.read_file(f"zip://{filepath}")
            else:
                gdf = gpd.read_file(filepath)

            # Ensure correct CRS (EPSG:4326)
            if gdf.crs is None:
                gdf = gdf.set_crs('EPSG:4326', allow_override=True)
            else:
                gdf = gdf.to_crs('EPSG:4326')

            # Simplify complex geometries if needed
            if simplify_tolerance and simplify_tolerance > 0:
                gdf['geometry'] = gdf['geometry'].simplify(tolerance=simplify_tolerance)

            # Plot with dateline handling
            for geom in gdf.geometry:
                if not geom.is_valid:
                    geom = geom.buffer(0)  # Fix invalid geometries

                if geom.geom_type == 'Polygon':
                    geoms = [geom]
                elif geom.geom_type == 'MultiPolygon':
                    geoms = list(geom.geoms)
                else:
                    continue

                for poly in geoms:
                    # Original and shifted version to handle dateline
                    original = gpd.GeoSeries([poly], crs='EPSG:4326')
                    shifted = original.translate(xoff=360)
                    combined = pd.concat([original, shifted])

                    for part in combined:
                        x, y = m(part.exterior.coords.xy[0], part.exterior.coords.xy[1])
                        if style == 'polygon':
                            ax.fill(x, y, color='#A9A9A9', ec='black', lw=0.5, zorder=2)
                        else:
                            ax.plot(x, y, color='black', lw=0.5, zorder=2)

        except Exception as e:
            print(f"Error plotting coastline from local file: {str(e)}")

    @staticmethod
    def plot_coastline_from_shapefile(ax, shapefile_path):
        """
        Plot coastline polygons from local shapefile with proper dateline handling
        
        Parameters:
        - ax: matplotlib axis object
        - shapefile_path: path to the shapefile (without .shp extension)
        """
        try:
            # Read shapefile using geopandas
            gdf = gpd.read_file(shapefile_path)
            
            # Ensure correct CRS (EPSG:4326)
            if gdf.crs is None:
                gdf = gdf.set_crs('EPSG:4326', allow_override=True)
            else:
                gdf = gdf.to_crs('EPSG:4326')
            
            # Simplify complex geometries (adjust tolerance as needed)
            gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.01)
            
            # Plot with dateline handling
            for geom in gdf.geometry:
                if not geom.is_valid:
                    geom = geom.buffer(0)  # Fix invalid geometries
                    
                if geom.geom_type in ['Polygon', 'MultiPolygon']:
                    # Create two versions - original and shifted by 360°
                    original = gpd.GeoSeries([geom], crs='EPSG:4326')
                    shifted = original.translate(xoff=360)
                    
                    # Combine both versions
                    combined = pd.concat([original, shifted])
                    
                    # Plot each geometry
                    for poly in combined:
                        if poly.geom_type == 'Polygon':
                            x, y = m(poly.exterior.coords.xy[0], poly.exterior.coords.xy[1])
                            ax.fill(x, y, color='#A9A9A9', ec='black', lw=0.5, zorder=2)
                        elif poly.geom_type == 'MultiPolygon':
                            for part in poly.geoms:
                                x, y = m(part.exterior.coords.xy[0], part.exterior.coords.xy[1])
                                ax.fill(x, y, color='#A9A9A9', ec='black', lw=0.5, zorder=2)
            
            
        except Exception as e:
            print(f"Error plotting coastline from shapefile: {str(e)}")

    @staticmethod
    def get_from_file(file_path, target_time, variable_name, adjust_lon=False):
        try:
            # Open dataset from local file
            with xr.open_dataset(file_path, engine='netcdf4', mask_and_scale=True, decode_cf=True) as ds:
                
                # Get available times (handle bytes if needed)
                if isinstance(ds.time.values[0], bytes):
                    time_str = [t.decode('utf-8') for t in ds.time.values]
                    time_dt = np.array([datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ") for t in time_str])
                else:
                    # Convert numpy datetime64 to datetime objects if needed
                    time_dt = [pd.to_datetime(t).to_pydatetime() for t in ds.time.values]
                
                # Convert target time to datetime object
                target_dt = datetime.strptime(target_time, "%Y-%m-%dT%H:%M:%SZ")
                
                # Find closest time by comparing timestamps
                time_index = np.argmin([abs((t - target_dt).total_seconds()) for t in time_dt])
                
                # Extract variable data
                if variable_name not in ds.variables:
                    available_vars = list(ds.variables.keys())
                    raise ValueError(f"Variable '{variable_name}' not found. Available variables: {available_vars}")
                
                data = ds[variable_name].isel(time=time_index)
                
                # If variable has 3 dimensions (e.g., depth), select first depth level
                if len(data.dims) == 3:
                    data = data.isel({data.dims[0]: 0})  # Select first index of first dimension
                
                # Determine coordinate names
                coord_names = {
                    'lon': ['lon', 'longitude', 'x', 'X'],
                    'lat': ['lat', 'latitude', 'y', 'Y']
                }
                
                # Find longitude coordinate
                lon_name = None
                for possible_name in coord_names['lon']:
                    if possible_name in ds.coords:
                        lon_name = possible_name
                        break
                if lon_name is None:
                    raise ValueError("Could not identify longitude coordinate variable")
                
                # Find latitude coordinate
                lat_name = None
                for possible_name in coord_names['lat']:
                    if possible_name in ds.coords:
                        lat_name = possible_name
                        break
                if lat_name is None:
                    raise ValueError("Could not identify latitude coordinate variable")
                
                # Get coordinates
                lon = ds[lon_name].values
                lat = ds[lat_name].values
                
                # Adjust longitude if requested (for 180° crossing)
                if adjust_lon:
                    if np.any(lon < 0):  # Only adjust if there are negative longitudes
                        lon = np.where(lon < 0, lon + 360, lon)
                
                # Extract and mask data values
                data_extract = np.ma.masked_invalid(data.values.squeeze())
                
                return lon, lat, data_extract
                
        except Exception as e:
            raise RuntimeError(f"Error accessing file data: {str(e)}")

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def getEEZ(ax, m, local_path=None, geojson_url=None, color='black', linewidth=1, linestyle='--'):
        """
        Plot EEZ boundaries from local file if available, otherwise from GeoServer URL.

        Parameters:
        - ax: matplotlib axis object
        - m: Basemap object (for coordinate transformation)
        - local_path: Path to local shapefile or GeoJSON (.shp, .geojson, or .zip containing shapefile)
        - geojson_url: URL to fetch GeoJSON from GeoServer
        - color, linewidth, linestyle: plot properties
        """
        gdf = None

        # Try reading from local file first, if provided
        if local_path is not None and os.path.exists(local_path):
            try:
                # Support for zipped shapefiles
                if local_path.endswith('.zip'):
                    gdf = gpd.read_file(f"zip://{local_path}")
                else:
                    gdf = gpd.read_file(local_path)
            except Exception as e:
                print(f"Failed to read local file {local_path}: {e}")

        # If no local data or failed, try GeoServer
        if gdf is None and geojson_url is not None:
            try:
                geojson_response = requests.get(geojson_url)
                if geojson_response.status_code == 200:
                    geojson_data = geojson_response.json()
                    gdf = gpd.GeoDataFrame.from_features(geojson_data['features'])
                else:
                    print(f"Failed to retrieve GeoJSON from {geojson_url}")
            except Exception as e:
                print(f"Error fetching GeoJSON from {geojson_url}: {e}")

        # If we still have no data, abort
        if gdf is None:
            print("No EEZ data available to plot.")
            return

        # Ensure correct CRS
        if gdf.crs is None:
            gdf = gdf.set_crs('EPSG:4326', allow_override=True)
        else:
            gdf = gdf.to_crs('EPSG:4326')

        # Plot boundaries, handling LineString, MultiLineString, Polygon, MultiPolygon and dateline wrap
        for geom in gdf.geometry:
            if not geom.is_valid:
                geom = geom.buffer(0)
            # LineString
            if geom.geom_type == 'LineString':
                x, y = m(*geom.xy)
                x = np.array(x)
                x = np.where(x < 0, x + 360, x)
                ax.plot(x, y, marker=None, color=color, linewidth=linewidth, linestyle=linestyle)
            # MultiLineString
            elif geom.geom_type == 'MultiLineString':
                for line in geom.geoms:
                    x, y = m(*line.xy)
                    x = np.array(x)
                    x = np.where(x < 0, x + 360, x)
                    ax.plot(x, y, marker=None, color=color, linewidth=linewidth, linestyle=linestyle)
            # Polygon
            elif geom.geom_type == 'Polygon':
                x, y = m(*geom.exterior.xy)
                x = np.array(x)
                x = np.where(x < 0, x + 360, x)
                ax.plot(x, y, marker=None, color=color, linewidth=linewidth, linestyle=linestyle)
            # MultiPolygon
            elif geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    x, y = m(*poly.exterior.xy)
                    x = np.array(x)
                    x = np.where(x < 0, x + 360, x)
                    ax.plot(x, y, marker=None, color=color, linewidth=linewidth, linestyle=linestyle)
            # Other geometry types will be ignored

    @staticmethod
    def cm2inch(*tupl):
        inch = 2.54
        if type(tupl[0]) == tuple:
            return tuple(i/inch for i in tupl[0])
        else:
            return tuple()
    
    @staticmethod
    def add_z_if_needed(s):
        if len(s) == 0:
            return 'Z'  # or just return s if you want empty string to remain empty
        if s[-1] != 'Z':
            return s + 'Z'
        return s

    @staticmethod
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

    @staticmethod
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
        time2 = Plotter.add_z_if_needed(time)
        return time2

    @staticmethod
    def get_dap_config(layer_map_data):
        dap_url = layer_map_data.url
        dap_variable = layer_map_data.layer_name
        dapvaribsplit = dap_variable.split(',')
        if "cache" in dap_url:
            idx = dap_url.find("POP")
            dap_url = "https://ocean-thredds01.spc.int/thredds/dodsC/" + dap_url[idx:]
        else:
            dap_url = layer_map_data.url.replace("wms", "dodsC")
        #if len(dapvaribsplit) >= 1:
        #    dap_variable = dapvaribsplit[0]
        return dap_url, dap_variable

    @staticmethod
    def get_title(layer_map_data,time):
        new_name = []
        week = False
        date = datetime.strptime(Plotter.add_z_if_needed(time), "%Y-%m-%dT%H:%M:%SZ")
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
                    if "66" in layer_map_data.get_map_names[0]:
                        cleaned = layer_map_data.get_map_names[0].replace('{', '').replace('}', '').replace('66', '')
                        formatted_date = date.strftime(layer_map_data.get_map_names[1])
                        date_str = layer_map_data.get_map_names[1]
                        start_date = date
                        end_date = start_date + relativedelta(months=5)
                        formatted_range = f"{start_date.strftime('%b %Y')} - {end_date.strftime('%b %Y')}"
                        title_suffix = "%s : %s" % (cleaned, formatted_range)
                    elif "122" in layer_map_data.get_map_names[0]:
                        if "Remove" in layer_map_data.get_map_names[0]:
                            cleaned = layer_map_data.get_map_names[0].replace('{', '').replace('}', '').replace('122', '').replace('Anomalies', '').replace('Remove', '')
                        else:
                            cleaned = layer_map_data.get_map_names[0].replace('{', '').replace('}', '').replace('122', '')
                        #cleaned = layer_map_data.get_map_names[0].replace('{', '').replace('}', '').replace('122', '')
                        formatted_date = date.strftime(layer_map_data.get_map_names[1])
                        date_str = layer_map_data.get_map_names[1]
                        start_date = date
                        end_date = start_date + relativedelta(months=11)
                        formatted_range = f"{start_date.strftime('%b %Y')} - {end_date.strftime('%b %Y')}"
                        title_suffix = "%s : %s" % (cleaned, formatted_range)
                    else:
                        cleaned = layer_map_data.get_map_names[0].replace('{', '').replace('}', '')
                        formatted_date = date.strftime(layer_map_data.get_map_names[1])
                        date_str = layer_map_data.get_map_names[1]
                        start_date = date
                        end_date = start_date + relativedelta(months=2)
                        formatted_range = f"{start_date.strftime('%b %Y')} - {end_date.strftime('%b %Y')}"
                        title_suffix = "%s : %s" % (cleaned, formatted_range)
                elif "Decile" in layer_map_data.get_map_names[0]:
                    if "66" in layer_map_data.get_map_names[0]:
                        cleaned = layer_map_data.get_map_names[0].replace('{', '').replace('}', '').replace('66', '')
                        formatted_date = date.strftime(layer_map_data.get_map_names[1])
                        date_str = layer_map_data.get_map_names[1]
                        start_date = date
                        end_date = start_date + relativedelta(months=5)
                        formatted_range = f"{start_date.strftime('%b %Y')} - {end_date.strftime('%b %Y')}"
                        title_suffix = "%s : %s" % (cleaned, formatted_range)
                    elif "122" in layer_map_data.get_map_names[0]:
                        cleaned = layer_map_data.get_map_names[0].replace('{', '').replace('}', '').replace('122', '')
                        formatted_date = date.strftime(layer_map_data.get_map_names[1])
                        date_str = layer_map_data.get_map_names[1]
                        start_date = date
                        end_date = start_date + relativedelta(months=11)
                        formatted_range = f"{start_date.strftime('%b %Y')} - {end_date.strftime('%b %Y')}"
                        title_suffix = "%s : %s" % (cleaned, formatted_range)
                    elif "33" in layer_map_data.get_map_names[0]:
                        cleaned = layer_map_data.get_map_names[0].replace('{', '').replace('}', '').replace('33', '')
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


    @staticmethod
    def plot_filled_contours_no_zero_levels(title,is_imperial_layer,
            ax, ax_legend, lon, lat, data, 
            min_color_plot=None, max_color_plot=None, steps=None,
            cmap_name='RdBu_r', units='(°C)', levels=None, white_color=(1, 1, 1, 1)):

        base = plt.get_cmap(cmap_name)

        
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

        

        # Create colorbar with imperial labels if it's an imperial layer
        # Create colorbar with imperial labels if it's an imperial layer
        if is_imperial_layer:
            if 'C' in units:
                # Convert Celsius to Fahrenheit
                def celsius_to_fahrenheit(c):
                    return (c * 9/5) + 32
                def celsius_to_fahrenheit_anom(c):
                    return (c * 9/5)
                
                # Create Fahrenheit tick labels
                if "Anomal" in title:
                    imperial_levels = [celsius_to_fahrenheit_anom(level) for level in levels]
                else:
                    imperial_levels = [celsius_to_fahrenheit(level) for level in levels]
                
                # Format the labels with appropriate precision
                imperial_labels = []
                for val in imperial_levels:
                    if val == 0:
                        imperial_labels.append("0")
                    elif abs(val) < 1:
                        imperial_labels.append(f"{val:.2f}")
                    elif abs(val) < 10:
                        imperial_labels.append(f"{val:.1f}")
                    else:
                        imperial_labels.append(f"{val:.0f}")
                
                # Update units label
                units_label = '(°F)'
                
            elif 'mm' in units.lower():
                # Convert millimeters to feet: 1 mm = 0.00328084 feet
                def mm_to_feet(mm):
                    return mm * 0.00328084
                
                # Create feet tick labels
                imperial_levels = [mm_to_feet(level) for level in levels]
                
                # Format the labels with appropriate precision
                imperial_labels = []
                for val in imperial_levels:
                    if val == 0:
                        imperial_labels.append("0")
                    elif abs(val) < 0.01:
                        imperial_labels.append(f"{val:.4f}")
                    elif abs(val) < 0.1:
                        imperial_labels.append(f"{val:.3f}")
                    elif abs(val) < 1:
                        imperial_labels.append(f"{val:.2f}")
                    elif abs(val) < 10:
                        imperial_labels.append(f"{val:.1f}")
                    else:
                        imperial_labels.append(f"{val:.0f}")
                
                # Update units label
                units_label = '(ft)'
            
            else:
                # For other imperial conversions (like meters to feet), add here
                imperial_levels = levels
                imperial_labels = [str(level) for level in levels]
                units_label = units
            
            # Create colorbar with imperial labels
            cbar = plt.colorbar(cs, cax=ax_legend)
            cbar.set_ticks(levels)  # Keep the original levels for positioning
            cbar.set_ticklabels(imperial_labels)  # But show imperial values
            cbar.set_label(units_label, fontsize=7, rotation=0, va='center', ha='left', labelpad=1)
                
        else:
            # Regular metric colorbar
            cbar = plt.colorbar(cs, cax=ax_legend)
            cbar.set_ticks(levels)
            cbar.set_label(units, fontsize=7, rotation=0, va='center', ha='left', labelpad=1)

        # Style the colorbar
        cbar.ax.tick_params(labelsize=8, pad=8, direction='out', length=6, width=1)
        
        try:
            cbar.solids.set_edgecolor("face")
        except Exception:
            pass

        return cs, cbar

    """
    def plot_filled_contours_no_zero_levels(
            ax, ax_legend, lon, lat, data, 
            min_color_plot=None, max_color_plot=None, steps=None,
            cmap_name='RdBu_r', units='(°C)', levels=None, white_color=(1, 1, 1, 1)):

        

        base = plt.get_cmap(cmap_name)

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

        # Detect a single “gap across zero”: levels[i] < 0 < levels[i+1]
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

        # Plot
        cs = ax.contourf(
            lon, lat, data,
            levels=levels,
            cmap=cmap,
            norm=norm,
            extend='both'
        )

        # Colorbar
        cbar = plt.colorbar(cs, cax=ax_legend)
        cbar.set_ticks(levels)
        cbar.set_ticklabels([str(t) for t in levels])
        cbar.ax.tick_params(labelsize=8, pad=2, direction='out', length=6, width=1)
        cbar.set_label(units, fontsize=8, rotation=0, va='center', ha='left', labelpad=1)

        try:
            cbar.solids.set_edgecolor("face")
        except Exception:
            pass

        return cs, cbar
    """

    @staticmethod
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

    @staticmethod
    def plot_filled_contours_no_zero(is_imperial_layer,ax, ax_legend, lon, lat, data, 
                            min_color_plot, max_color_plot, steps,
                            cmap_name='RdBu_r', units='(°C)'):
        #print('accesss')
        # Create fixed levels for contours, excluding zero
        levels = np.arange(min_color_plot, max_color_plot, steps)
        levels = levels[levels != 0]  # Remove zero level
        
        # Plot filled contours with fixed levels
        cs = ax.contourf(
            lon, lat, data,
            levels=levels,
            cmap=cmap_name,
            extend='both'  # Adds arrows if data exceeds min/max
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
            cbar.set_label(units_label, fontsize=7, rotation=0, va='center', ha='left', labelpad=1)
            
        else:
            # Regular Celsius colorbar
            cbar = plt.colorbar(cs, cax=ax_legend)
            cbar.set_ticks(levels)
            cbar.set_label(units, fontsize=7, rotation=0, va='center', ha='left', labelpad=1)

        # Style the colorbar
        cbar.ax.tick_params(labelsize=8, pad=2, direction='out', length=6, width=1)
        
        return cs, cbar

    """
    def plot_filled_contours_no_zero(ax, ax_legend, lon, lat, data, 
                            min_color_plot, max_color_plot, steps,
                            cmap_name='RdBu_r', units='(°C)'):

        # Create fixed levels for contours, excluding zero
        levels = np.arange(min_color_plot, max_color_plot, steps)
        levels = levels[levels != 0]  # Remove zero level
        
        # Plot filled contours with fixed levels
        cs = ax.contourf(
            lon, lat, data,
            levels=levels,
            cmap=cmap_name,
            extend='both'  # Adds arrows if data exceeds min/max
        )
        
        # Add colorbar with matching ticks
        cbar = plt.colorbar(cs, cax=ax_legend)
        cbar.set_ticks(levels)  # Same ticks as contour levels
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label(
            units,
            fontsize=6,
            rotation=0,
            va='center',
            ha='left',
            labelpad=1
        )
        
        return cs, cbar
    """
    @staticmethod
    def plot_filled_contours(is_imperial_layer,ax, ax_legend, lon, lat, data, 
                            min_color_plot, max_color_plot, steps,
                            cmap_name='RdBu_r', units='(°C)'):
        import numpy as np
        import matplotlib.pyplot as plt

        # Create fixed levels for contours
        levels = np.arange(min_color_plot, max_color_plot, steps)

        # Determine number of decimal places in "steps"
        steps_str = str(steps)
        if '.' in steps_str:
            n_decimals = len(steps_str.split('.')[-1])
        else:
            n_decimals = 0

        # Set tick pad proportional to decimal places in 'steps'
        tick_pad = 2 + n_decimals * 3  # adjust multiplier as needed

        # Plot filled contours with fixed levels
        cs = ax.contourf(
            lon, lat, data,
            levels=levels,
            cmap=cmap_name,
            extend='both'  # Adds arrows if data exceeds min/max
        )
        cbar = plt.colorbar(cs, cax=ax_legend)
        
        if is_imperial_layer and re.search(r'\bm\b', units.lower()):
            # Convert current tick values from meters to feet
            current_ticks = cbar.get_ticks()
            feet_ticks = [tick * 3.28084 for tick in current_ticks]  # meters to feet
            
            # Create formatted labels
            feet_labels = []
            for f_val in feet_ticks:
                if f_val == 0:
                    feet_labels.append("0")
                elif abs(f_val) < 1:
                    feet_labels.append(f"{f_val:.2f}")
                elif abs(f_val) < 10:
                    feet_labels.append(f"{f_val:.1f}")
                else:
                    feet_labels.append(f"{f_val:.0f}")
            
            # Set the new labels
            cbar.set_ticklabels(feet_labels)
            units_label = '(ft)'
            cbar.ax.tick_params(labelsize=8, pad=tick_pad)
            cbar.set_label(
                units_label,
                fontsize=7,
                rotation=0,
                va='center',
                ha='left',
                labelpad=1
            )
        elif is_imperial_layer and 'C' in units:
            # Convert levels from Celsius to Fahrenheit
            fahrenheit_levels = [(level * 9/5) + 32 for level in levels]  # Celsius to Fahrenheit
            
            # Create formatted labels
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
            
            # Set the new labels
            cbar.set_ticks(levels)  # Keep original levels for positioning
            cbar.set_ticklabels(fahrenheit_labels)
            units_label = '(°F)'
            cbar.ax.tick_params(labelsize=8, pad=tick_pad)
            cbar.set_label(
                units_label,
                fontsize=7,
                rotation=0,
                va='center',
                ha='left',
                labelpad=1
            )
        elif 'mm' in units.lower():
            # Convert millimeters to feet: 1 mm = 0.00328084 feet
            def mm_to_feet(mm):
                return mm * 0.00328084
            
            # Create feet tick labels
            imperial_levels = [mm_to_feet(level) for level in levels]
            
            # Format the labels with appropriate precision
            imperial_labels = []
            for val in imperial_levels:
                if val == 0:
                    imperial_labels.append("0")
                elif abs(val) < 0.01:
                    imperial_labels.append(f"{val:.4f}")
                elif abs(val) < 0.1:
                    imperial_labels.append(f"{val:.3f}")
                elif abs(val) < 1:
                    imperial_labels.append(f"{val:.2f}")
                elif abs(val) < 10:
                    imperial_labels.append(f"{val:.1f}")
                else:
                    imperial_labels.append(f"{val:.0f}")
            
            # Set the new ticks and labels - THIS WAS MISSING
            cbar.set_ticks(levels)  # Keep original levels for positioning
            cbar.set_ticklabels(imperial_labels)
            units_label = '(ft)'
            cbar.ax.tick_params(labelsize=8, pad=tick_pad)
            cbar.set_label(
                units_label,
                fontsize=7,
                rotation=0,
                va='center',
                ha='left',
                labelpad=1
            )
        else:
            units_label = units
            cbar.set_ticks(levels)  # Same ticks as contour levels

            # Format tick labels to match the number of decimals in steps
            #tick_labels = [f"{level:.{n_decimals}f}".replace("-0.0", "0.0") for level in levels]
            #cbar.set_ticklabels(tick_labels)

            # Format tick labels to match the number of decimals in steps
            #tick_labels = [f"{level:.{n_decimals}f}" for level in levels]
            #cbar.set_ticklabels(tick_labels)

            #Format tick labels to match the number of decimals in steps
            def format_tick(level):
                if abs(level) < 1e-10:  # Effectively zero
                    return f"0.0"
                formatted = f"{level:.{n_decimals}f}"
                return "0.0" if formatted == "-0.0" else formatted
            
            tick_labels = [format_tick(level) for level in levels]
            cbar.set_ticklabels(tick_labels)

            cbar.ax.tick_params(labelsize=8, pad=tick_pad)
            if "Decade" in units:
                cbar.set_label(
                    units,
                    fontsize=7,
                    rotation=90,
                    va='center',
                    ha='center',
                    labelpad=5
                )
            else:
                cbar.set_label(
                    units,
                    fontsize=7,
                    rotation=0,
                    va='center',
                    ha='left',
                    labelpad=1
                )
        
        
        """
        # Add colorbar with matching ticks
        cbar = plt.colorbar(cs, cax=ax_legend)
        cbar.set_ticks(levels)  # Same ticks as contour levels

        # Format tick labels to match the number of decimals in steps
        tick_labels = [f"{level:.{n_decimals}f}" for level in levels]
        cbar.set_ticklabels(tick_labels)

        cbar.ax.tick_params(labelsize=7, pad=tick_pad)
        cbar.set_label(
            units,
            fontsize=6,
            rotation=0,
            va='center',
            ha='left',
            labelpad=1
        )
        """

        return cs, cbar

    """
    def plot_filled_contours(ax, ax_legend, lon, lat, data, 
                        min_color_plot, max_color_plot, steps,
                        cmap_name='RdBu_r', units='(°C)'):
        import numpy as np
        import matplotlib.pyplot as plt

        # Create fixed levels for contours
        levels = np.arange(min_color_plot, max_color_plot, steps)

        # Determine number of decimal places in "steps"
        steps_str = str(steps)
        if '.' in steps_str:
            n_decimals = len(steps_str.split('.')[-1])
        else:
            n_decimals = 0

        # Set tick pad proportional to decimal places in 'steps'
        tick_pad = 2 + n_decimals * 3  # adjust multiplier as needed

        # Plot filled contours with fixed levels
        cs = ax.contourf(
            lon, lat, data,
            levels=levels,
            cmap=cmap_name,
            extend='both'  # Adds arrows if data exceeds min/max
        )

        # Add colorbar with matching ticks
        cbar = plt.colorbar(cs, cax=ax_legend)
        cbar.set_ticks(levels)  # Same ticks as contour levels

        # Format tick labels to match the number of decimals in steps
        tick_labels = [f"{level:.{n_decimals}f}" for level in levels]
        cbar.set_ticklabels(tick_labels)

        cbar.ax.tick_params(labelsize=7, pad=tick_pad)
        cbar.set_label(
            units,
            fontsize=6,
            rotation=0,
            va='center',
            ha='left',
            labelpad=1
        )

        return cs, cbar
    """
    @staticmethod
    def _mask_sst(data, units_hint=""):
        """
        Mask SST-like data to avoid coastline artifacts:
        - Mask NaN/Inf
        - Mask unrealistic SST values (<= -3 or >= 45 °C)
        - Mask exact zeros (common land sentinel in some local files)
        """
        ma = np.ma.masked_invalid(data)
        ma = np.ma.masked_where((ma <= -3.0) | (ma >= 45.0), ma)
        ma = np.ma.masked_where(np.isclose(ma, 0.0), ma)
        return ma
    """
    @staticmethod
    def plot_climatology(is_imperial_layer,
        dap_url, time, ax, ax_legend, lon, lat, data, 
        min_color_plot, max_color_plot, steps,
        cmap_name='RdBu_r', units='(°C)', local_path=False, local_path_str=None
    ):
        # Color levels
        levels = np.arange(min_color_plot, max_color_plot, steps)

        # Mask SST to prevent coastlines from appearing as isolines
        data_masked = _mask_sst(data, units_hint=units)

        # Filled contours
        cs = ax.contourf(
            lon, lat, data_masked,
            levels=levels,
            cmap=cmap_name,
            extend='both',
            corner_mask=True
        )

        # SST 29°C contour
        ax.contour(
            lon, lat, data_masked,
            levels=[29],
            colors='purple',
            linewidths=2,
            linestyles='solid',
            zorder=6,
            corner_mask=True
        )

        # Try plotting climatology 29°C contour
        try:
            clim_lon, clim_lat, sst_clim = getfromDAP(
                dap_url, time, "sst_clim", adjust_lon=True,
                local_path=local_path, local_path_str=local_path_str
            )
            sst_clim_masked = _mask_sst(sst_clim, units_hint=units)
            cont = ax.contour(
                clim_lon, clim_lat, sst_clim_masked,
                levels=[29],
                colors='green',
                linewidths=2,
                linestyles='solid',
                zorder=7,
                corner_mask=True
            )
            # Optional: warn if not drawn at all
            if len(cont.allsegs[0][0]) == 0:
                print("Warning: No green climatology contour drawn at 29°C (level not present in data).")
        except Exception as e:
            print(f"Climatology 29°C contour not plotted: {e}")

        # Legend
        legend_elements = [
            Line2D([0], [0], color='purple', lw=2, label='SST 29°C'),
            Line2D([0], [0], color='green', lw=2, label='Climatology 29°C')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=6)

        # Colorbar
        if is_imperial_layer:
            if 'C' in units:
                # Convert Celsius to Fahrenheit
                def celsius_to_fahrenheit(c):
                    return (c * 9/5) + 32
                
                # Create Fahrenheit tick labels
                imperial_levels = [celsius_to_fahrenheit(level) for level in levels]
                
                # Format the labels with appropriate precision
                imperial_labels = []
                for val in imperial_levels:
                    if val == 0:
                        imperial_labels.append("0")
                    elif abs(val) < 1:
                        imperial_labels.append(f"{val:.2f}")
                    elif abs(val) < 10:
                        imperial_labels.append(f"{val:.1f}")
                    else:
                        imperial_labels.append(f"{val:.0f}")
                
                # Update units label
                units_label = '(°F)'
                
            elif 'mm' in units.lower():
                # Convert millimeters to feet: 1 mm = 0.00328084 feet
                def mm_to_feet(mm):
                    return mm * 0.00328084
                
                # Create feet tick labels
                imperial_levels = [mm_to_feet(level) for level in levels]
                
                # Format the labels with appropriate precision
                imperial_labels = []
                for val in imperial_levels:
                    if val == 0:
                        imperial_labels.append("0")
                    elif abs(val) < 0.01:
                        imperial_labels.append(f"{val:.4f}")
                    elif abs(val) < 0.1:
                        imperial_labels.append(f"{val:.3f}")
                    elif abs(val) < 1:
                        imperial_labels.append(f"{val:.2f}")
                    elif abs(val) < 10:
                        imperial_labels.append(f"{val:.1f}")
                    else:
                        imperial_labels.append(f"{val:.0f}")
                
                # Update units label
                units_label = '(ft)'
            
            else:
                # For other imperial conversions (like meters to feet), add here
                imperial_levels = levels
                imperial_labels = [str(level) for level in levels]
                units_label = units
            
            # Create colorbar with imperial labels
            cbar = plt.colorbar(cs, cax=ax_legend)
            cbar.set_ticks(levels)  # Keep the original levels for positioning
            cbar.set_ticklabels(imperial_labels)  # But show imperial values
            cbar.set_label(units_label, fontsize=7, rotation=0, va='center', ha='left', labelpad=1)
                
        else:
            # Regular metric colorbar
            cbar = plt.colorbar(cs, cax=ax_legend)
            cbar.set_ticks(levels)
            cbar.set_label(units, fontsize=7, rotation=0, va='center', ha='left', labelpad=1)

        # Style the colorbar
        cbar.ax.tick_params(labelsize=8, pad=2, direction='out', length=6, width=1)

        return cs, cbar

    """
    @staticmethod
    def plot_climatology(is_imperial_layer,dap_url, time, ax, ax_legend, lon, lat, data, 
                        min_color_plot, max_color_plot, steps,
                        cmap_name='RdBu_r', units='(°C)', local_path=False, local_path_str=None):
        # Color levels
        levels = np.arange(min_color_plot, max_color_plot, steps)

        # Mask SST to prevent coastlines from appearing as isolines
        data_masked = Plotter._mask_sst(data, units_hint=units)

        # Filled contours
        cs = ax.contourf(
            lon, lat, data_masked,
            levels=levels,
            cmap=cmap_name,
            extend='both',
            corner_mask=True
        )

        
        ax.contour(
            lon, lat, data_masked,
            levels=[29],
            colors='purple',
            linewidths=2,
            linestyles='solid',
            zorder=6,
            corner_mask=True
        )


        try:
            clim_lon, clim_lat, sst_clim = Plotter.getfromDAP(
                dap_url, time, "sst_clim", adjust_lon=True,
                local_path=local_path, local_path_str=local_path_str
            )
            sst_clim_masked = Plotter._mask_sst(sst_clim, units_hint=units)
            ax.contour(
                clim_lon, clim_lat, sst_clim_masked,
                levels=[29],
                colors='green',
                linewidths=2,
                linestyles='solid',
                zorder=7,
                corner_mask=True
            )
        except Exception:
            pass  # silently skip if clim not available

        # Legend
        legend_elements = [
            Line2D([0], [0], color='purple', lw=1, label='SST 29°C'),
            Line2D([0], [0], color='green', lw=1, label='Climatology 29°C')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=6)

        # Colorbar
        cbar = plt.colorbar(cs, cax=ax_legend)
        cbar.set_ticks(levels)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label(units, fontsize=6, rotation=0, va='center', ha='left', labelpad=1)

        return cs, cbar
    
    @staticmethod
    def plot_filled_pcolor(ax, ax_legend, lon, lat, data, 
                    min_color_plot, max_color_plot, steps,
                    cmap_name='RdBu_r', units='(°C)'):
        # Create fixed levels for color normalization
        levels = np.arange(min_color_plot, max_color_plot, steps)
        
        # Create a BoundaryNorm to discretize the colorbar
        norm = BoundaryNorm(levels, ncolors=256, clip=True)
        
        # Plot pcolor with fixed levels
        pc = ax.pcolormesh(
            lon, lat, data,
            norm=norm,
            cmap=cmap_name,
            shading='auto'  # Can be 'nearest', 'flat', 'auto', etc.
        )
        
        # Add colorbar with matching ticks
        cbar = plt.colorbar(pc, cax=ax_legend)
        cbar.set_ticks(levels)  # Same ticks as contour levels
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label(
            units,
            fontsize=6,
            rotation=0,
            va='center',
            ha='left',
            labelpad=1
        )
        
        return pc, cbar

    @staticmethod
    def plot_wave_field(is_imperial_layer,ax, ax_legend, m, lon, lat, wave_height, wave_dir,
                    min_color_plot, max_color_plot, steps, region, step,
                    cmap_name='jet', units='m',
                    scale=30, arrow_scale=0.5):
        
        # Convert wave direction to components (add 180° to reverse direction)
        wave_dir_rad = np.radians(wave_dir + 180)  # Reverse direction for quiver plot
        u = wave_height * np.sin(wave_dir_rad)  # Eastward component
        v = wave_height * np.cos(wave_dir_rad)  # Northward component
        
        # Create grid coordinates
        x, y = m(*np.meshgrid(lon, lat))
        
        # Create levels and normalization
        levels = np.arange(min_color_plot, max_color_plot, steps)
        norm = BoundaryNorm(levels, ncolors=256)
        
        # Plot wave height field
        cs = ax.pcolormesh(
            x, y, wave_height,
            cmap=cmap_name,
            norm=norm,
            shading='auto'
        )
        
        cbar = plt.colorbar(cs, cax=ax_legend)

        if is_imperial_layer and 'm' in units.lower():
            # Convert current tick values from meters to feet
            current_ticks = cbar.get_ticks()
            feet_ticks = [tick * 3.28084 for tick in current_ticks]  # meters to feet
            
            # Create formatted labels
            feet_labels = []
            for f_val in feet_ticks:
                if f_val == 0:
                    feet_labels.append("0")
                elif abs(f_val) < 1:
                    feet_labels.append(f"{f_val:.2f}")
                elif abs(f_val) < 10:
                    feet_labels.append(f"{f_val:.1f}")
                else:
                    feet_labels.append(f"{f_val:.0f}")
            
            # Set the new labels
            cbar.set_ticklabels(feet_labels)
            units_label = '(ft)'
        else:
            units_label = units

        cbar.set_label(
            units_label,
            fontsize=6,
            rotation=0,
            va='center',
            ha='left',
            labelpad=1
        )
        
        # Create directional arrows (already corrected by +180 above)
        theta = wave_dir_rad  # Use the already corrected direction
        u_arrows = arrow_scale * np.sin(theta)
        v_arrows = arrow_scale * np.cos(theta)
        
        q = ax.quiver(x[::step, ::step], y[::step, ::step], 
                    u_arrows[::step, ::step], v_arrows[::step, ::step],
                    scale=scale, width=0.003, 
                    headwidth=2.5, headlength=3, headaxislength=2.5,
                    color='black', pivot='middle', minshaft=2,
                    edgecolor='black', linewidth=0.3)
        
        # Add quiver key (without text)
        qk = plt.quiverkey(q, 0.82, 0.12, 1, 
                        '', labelpos='E',
                        coordinates='axes', fontproperties={'size': 9},
                        labelsep=0.05, labelcolor='black')
        cbar.ax.tick_params(labelsize=8)
        
        return cs, q, cbar

    """
    def plot_wave_field(ax, ax_legend, m, lon, lat, wave_height, wave_dir,
                    min_color_plot, max_color_plot, steps, region, step,
                    cmap_name='jet', units='m',
                    scale=30, arrow_scale=0.5):
        
        # Convert wave direction to components (add 180° to reverse direction)
        wave_dir_rad = np.radians(wave_dir + 180)  # Reverse direction for quiver plot
        u = wave_height * np.sin(wave_dir_rad)  # Eastward component
        v = wave_height * np.cos(wave_dir_rad)  # Northward component
        
        # Create grid coordinates
        x, y = m(*np.meshgrid(lon, lat))
        
        # Create levels and normalization
        levels = np.arange(min_color_plot, max_color_plot, steps)
        norm = BoundaryNorm(levels, ncolors=256)
        
        # Plot wave height field
        cs = ax.pcolormesh(
            x, y, wave_height,
            cmap=cmap_name,
            norm=norm,
            shading='auto'
        )
        
        # Create colorbar in specified legend axes
        cbar = plt.colorbar(cs, cax=ax_legend)
        cbar.set_label(
            units,
            fontsize=5,
            rotation=0,
            va='center',
            ha='left',
            labelpad=1
        )
        
        # Create directional arrows (already corrected by +180 above)
        theta = wave_dir_rad  # Use the already corrected direction
        u_arrows = arrow_scale * np.sin(theta)
        v_arrows = arrow_scale * np.cos(theta)
        
        q = ax.quiver(x[::step, ::step], y[::step, ::step], 
                    u_arrows[::step, ::step], v_arrows[::step, ::step],
                    scale=scale, width=0.003, 
                    headwidth=2.5, headlength=3, headaxislength=2.5,
                    color='black', pivot='middle', minshaft=2,
                    edgecolor='black', linewidth=0.3)
        
        # Add quiver key (without text)
        qk = plt.quiverkey(q, 0.82, 0.12, 1, 
                        '', labelpos='E',
                        coordinates='axes', fontproperties={'size': 9},
                        labelsep=0.05, labelcolor='black')
        cbar.ax.tick_params(labelsize=7)
        
        return cs, q, cbar
    """
    @staticmethod
    def plot_discrete_map(ax, ax_legend, lons, lats, bleaching_data, 
                            cmap_colors=None, colorbar_labels=None):
    
        # Validate inputs
        if bleaching_data.ndim != 2:
            raise ValueError("bleaching_data must be a 2D array")
        
        if len(cmap_colors) != len(colorbar_labels):
            raise ValueError("Number of colors must match number of labels")
        
        try:
            # Calculate bounds automatically based on number of categories
            n_categories = len(cmap_colors)
            bounds = np.arange(n_categories + 1)  # [0, 1, 2, ..., n_categories]
            
            # Calculate tick positions (middle of each color band)
            ticks = bounds[:-1] + 0.5
            
            # Create colormap and normalization
            cmap = mcolors.ListedColormap(cmap_colors)
            norm = mcolors.BoundaryNorm(bounds, cmap.N)
            
            # Create plot
            cs = ax.pcolormesh(lons, lats, bleaching_data, 
                            cmap=cmap, norm=norm, 
                            shading='auto')
            
            # Create colorbar
            cbar = plt.colorbar(cs, cax=ax_legend)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(colorbar_labels)
            cbar.ax.tick_params(labelsize=6)
            
            return cs, cbar
            
        except Exception as e:
            raise RuntimeError(f"Error plotting coral bleaching data: {str(e)}")

    @staticmethod
    def plot_discrete_map_ranges(ax, ax_legend, lons, lats, bleaching_data,
                            cmap_colors=None, colorbar_labels=None, ranges=None):
        """
        Final working version that properly handles:
        - Discontinuous ranges (like 2-3, 4-7)
        - Exactly matches 7 colors to 7 range segments
        - Maintains your original range definitions
        """
        # Validate inputs
        if bleaching_data.ndim != 2:
            raise ValueError("bleaching_data must be a 2D array")
        
        if len(cmap_colors) != len(colorbar_labels) or len(cmap_colors) != len(ranges):
            raise ValueError("Number of colors, labels, and ranges must match")

        try:
            # Create segments from ranges
            segments = []
            for r in ranges:
                if '-' in r:
                    start, end = map(float, r.split('-'))
                    segments.append((start, end))
                else:
                    val = float(r)
                    segments.append((val, val))  # Treat single values as range with equal start/end
            
            # Create colormap with exactly N colors for N segments
            cmap = mcolors.ListedColormap(cmap_colors)
            
            # Create normalization that maps each segment to one color
            # We'll use the midpoint of each segment to determine color mapping
            norm = mcolors.Normalize(vmin=min(s[0] for s in segments), 
                                vmax=max(s[1] for s in segments))
            
            # Create plot - we'll manually map values to colors
            # First, create an array where each value is mapped to its segment index
            segment_idx = np.zeros_like(bleaching_data, dtype=int)
            for i, (start, end) in enumerate(segments):
                mask = (bleaching_data >= start) & (bleaching_data <= end)
                segment_idx[mask] = i
            
            # Now plot using the segment indices
            cs = ax.pcolormesh(lons, lats, segment_idx,
                            cmap=cmap, 
                            vmin=0, vmax=len(segments)-1,
                            shading='auto')
            
            # Calculate midpoints for ticks
            ticks = [(seg[0] + seg[1])/2 for seg in segments]
            
            # Create colorbar
            cbar = plt.colorbar(cs, cax=ax_legend)
            cbar.set_ticks(np.arange(len(segments)))  # One tick per segment
            cbar.set_ticklabels(colorbar_labels)
            cbar.ax.tick_params(labelsize=6)
            
            # Adjust label rotation if needed
            for label in cbar.ax.get_xticklabels():
                label.set_rotation(45)
                label.set_horizontalalignment('right')
            
            return cs, cbar
            
        except Exception as e:
            raise RuntimeError(f"Error plotting discrete map: {str(e)}")

    @staticmethod
    def add_logo_and_footer(fig, ax, ax2, ax2_pos, region, 
                        copyright_text, footer_text, dataset_text,
                        logo_path="./Logo_cropped.png"):
        # Add logo
        """
        try:
            logo_img = Image.open(logo_path)
            if region == 1:
                logo_ax = fig.add_axes([0.08, 0.85, 0.12, 0.12])  # [left, bottom, width, height]
            else:
                logo_ax = fig.add_axes([0.12, 0.85, 0.12, 0.12])
            
            logo_ax.imshow(logo_img)
            logo_ax.axis('off')
        except FileNotFoundError:
            print(f"Logo file not found at {logo_path}")
        """

        ax2_pos = ax2.get_position()

        # Calculate text position - slightly below ax2 with padding
        text_y = ax2_pos.y0 - 0.02  # 0.02 is 2% of figure height below ax2

        # Set common text properties
        text_props = {
            'va': 'top',       # Vertical alignment at top of text
            'fontsize': 6,     # Slightly smaller font for better fit
            'transform': fig.transFigure  # Use figure coordinates
        }

        # Left-aligned copyright text (aligned with ax2's left edge)
        fig.text(
            x=ax2_pos.x0-0.11, 
            y=text_y-0.03,
            s=copyright_text,
            ha='left',
            **text_props
        )

        # Right-aligned dataset text (aligned with ax2's right edge)
        # Using the dataset_text variable you already have from get_title()
        fig.text(
            x=ax2_pos.x1+0.11, 
            y=text_y-0.03,
            s=dataset_text,
            ha='right',
            **text_props
        )

        # Adjust subplots
        plt.subplots_adjust(bottom=0.15)

    @staticmethod
    def plot_levels_pcolor(ax, ax_legend, lons, lats, chl_data,cmap_name='jet', units='mg/m³',levels=[]):
        # Clip data to level boundaries
        chl_clipped = np.clip(chl_data, levels[0], levels[-1])
        
        # Create colormap with one less color than levels
        cmap = plt.get_cmap(cmap_name, len(levels)-1)
        
        # Create normalization with extend to handle out-of-range values
        norm = mcolors.BoundaryNorm(levels, cmap.N)
        
        # Plot with discrete levels
        cs = ax.pcolormesh(lons, lats, chl_clipped,
                        cmap=cmap,
                        norm=norm,
                        shading='auto')
        
        # Create colorbar with exact level ticks
        cbar = plt.colorbar(cs, cax=ax_legend, extend='both')
        cbar.set_ticks(levels)  # Set ticks exactly at level boundaries
        cbar.set_ticklabels([f"{x:.2f}" for x in levels])
        cbar.set_label(units, fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        
        return cs, cbar

    @staticmethod
    def plot_levels_contour(ax, ax_legend, lons, lats, chl_data, cmap_name='jet',
                            units='mg/m³', levels=None, add_contours=True,
                            contour_kwargs=None):
        
        # Default contour style
        if contour_kwargs is None:
            contour_kwargs = {
                'colors': 'k',  # Black contour lines
                'linewidths': 0.5,
                'linestyles': 'solid',
                'alpha': 0.5  # Semi-transparent
            }
        
        # Clip data to level boundaries
        chl_clipped = np.clip(chl_data, levels[0], levels[-1])
        
        # Create colormap and normalization
        cmap = plt.get_cmap(cmap_name, len(levels)-1)
        norm = mcolors.BoundaryNorm(levels, cmap.N)
        
        # Plot filled colors
        mesh = ax.pcolormesh(lons, lats, chl_clipped,
                            cmap=cmap,
                            norm=norm,
                            shading='auto')
        
        # Add contour lines if requested
        if add_contours:
            # Create 2D grid if needed (for contour)
            if lons.ndim == 1 or lats.ndim == 1:
                lon_grid, lat_grid = np.meshgrid(lons, lats)
            else:
                lon_grid, lat_grid = lons, lats
                
            contours = ax.contour(lon_grid, lat_grid, chl_clipped,
                                levels=levels,
                                **contour_kwargs)
            
            # Optionally add contour labels
            ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')
        
        # Create colorbar
        cbar = plt.colorbar(mesh, cax=ax_legend, extend='both')
        cbar.set_ticks(levels)
        cbar.set_ticklabels([f"{x:.2f}" for x in levels])
        cbar.set_label(units, fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        
        return mesh, cbar

    @staticmethod
    def plot_current_magnitude(ax, ax_legend, lon, lat, uo, vo, region,
                            min_color_plot=None, max_color_plot=None, steps=None,
                            cmap_name='viridis', units='(m/s)',
                            show_arrows=True, arrow_scale=1.0, density=5,
                            arrow_color='k', min_speed=0.01, **kwargs):

        # Handle coordinate arrays
        if lon.ndim == 1 or lat.ndim == 1:
            lon, lat = np.meshgrid(lon, lat)
        
        # Compute current magnitude
        speed = np.sqrt(uo**2 + vo**2)

        # Set up color normalization
        """
        if all(v is not None for v in [min_color_plot, max_color_plot, steps]):
            levels = np.linspace(min_color_plot, max_color_plot,
                                int((max_color_plot - min_color_plot) / steps) + 1)
            norm = BoundaryNorm(levels, ncolors=256)
        else:
            norm = None
            levels = None
        """
        levels = [0.05, 0.1, 0.15, 0.2,0.25, 0.3,0.35, 0.4,0.45, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]
        norm = BoundaryNorm(levels, ncolors=256)

        # Create colormap
        cmap = plt.get_cmap(cmap_name)
        #norm = Normalize(vmin=0.0, vmax=1.5)
        #levels = None
        levels = np.arange(0.0, 1.5 + 0.05, 0.05)  # Ensures 1.5 is included

        # Use BoundaryNorm for discrete color steps
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)


        # Plot scalar field
        pcm = ax.pcolormesh(lon, lat, speed, cmap=cmap, norm=norm, shading='auto', **kwargs)

        scale_factor = 0.08  # Adjust this factor to control the arrow size
        u_scaled = uo * scale_factor
        v_scaled = vo * scale_factor

        # Define a slice to skip drawing some of the quiver arrows to reduce clutter
        if int(region) == 1:
            skip = (slice(None, None, 30), slice(None, None, 30))
        elif int(region) == 5 or int(region) == 11:
            skip = (slice(None, None, 10), slice(None, None, 10))
        else:
            skip = (slice(None, None, 5), slice(None, None, 5))

        # Use the quiver function to display current vectors with their direction and intensity
        quiv = ax.quiver(lon[skip], lat[skip], u_scaled[skip], v_scaled[skip], color='black', scale=2, width=0.003, headwidth=4)


        # Enhanced arrow plotting
        #quiv = None
        """
        if show_arrows:
            # Subsample for clarity
            skip = (slice(None, None, density), slice(None, None, density))
            lon_sub = lon[skip]
            lat_sub = lat[skip]
            uo_sub = uo[skip]
            vo_sub = vo[skip]
            speed_sub = speed[skip]
            
            # Filter weak currents
            mask = speed_sub >= min_speed
            lon_sub = lon_sub[mask]
            lat_sub = lat_sub[mask]
            uo_sub = uo_sub[mask]
            vo_sub = vo_sub[mask]
            speed_sub = speed_sub[mask]
            
            # Automatic scaling if arrow_scale=None
            if arrow_scale is None:
                arrow_scale = 50 / speed_sub.mean() if speed_sub.mean() > 0 else 1.0
            
            # Arrow coloring
            arrow_coloring = speed_sub if arrow_color == 'magnitude' else arrow_color
            
            # Plot scaled arrows
            quiv = ax.quiver(
                lon_sub, lat_sub,
                uo_sub, vo_sub,
                scale=2,            # Inverse relationship with scale
                scale_units='inches',             # Better for geographic plots
                width=0.003,                     # Shaft thickness
                headwidth=4,                      # Arrowhead width
                headlength=4,                      # Arrowhead length
                headaxislength=3.5,               # Head base to tip
                minlength=0.5,                     # Minimum arrow length
                color=arrow_coloring,
                pivot='mid',                       # Arrows centered on points
                angles='xy',                       # Proper geographic angles
                zorder=5                           # Draw above pcolormesh
            )
            
            # Add reference arrow if using magnitude scaling
            if arrow_color != 'magnitude' and len(speed_sub) > 0:
                median_speed = np.median(speed_sub)
                ax.quiverkey(quiv, 0.85, 0.05, median_speed, 
                            f'{median_speed:.2f} m/s', 
                            coordinates='axes',
                            labelpos='E')
        """
        # Colorbar
        cbar = plt.colorbar(pcm, cax=ax_legend)
        label_step = 0.1  # Adjust to control label frequency (e.g., 0.2 for even fewer)
        selected_ticks = np.arange(levels[0], levels[-1] + label_step, label_step)
        cbar.set_ticks(selected_ticks)  
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label(units, fontsize=6, rotation=0, va='center', ha='left', labelpad=1)
        
        #cbar = plt.colorbar(pcm, cax=ax_legend)
        #if levels is not None:
        #    cbar.set_ticks(levels)
        #cbar.ax.tick_params(labelsize=7)
        #cbar.set_label(units, fontsize=6, rotation=0,va='center', ha='left', labelpad=1)

        return pcm, quiv, cbar

    @staticmethod
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

    @staticmethod
    def plot_city_names(ax, m, short_name, city_file='config/pac_names.json'):

        # Load city data
        cities = pd.read_json(city_file)

        # Filter for the selected country's cities
        filtered_cities = cities[cities['country_code'] == short_name]

        # If no cities found, return without plotting
        if filtered_cities.empty:
            return

        # Transform coordinates
        longitudes = filtered_cities['lon'].values
        latitudes = filtered_cities['lat'].values
        x_coords, y_coords = m(longitudes, latitudes)
        font_zize = 6

        if short_name == "PAC":
            font_zize = 5
        # Plot city names
        for x, y, name in zip(x_coords, y_coords, filtered_cities['name']):
            ax.text(x + 0.1, y + 0.1, name,
                    fontsize=font_zize, color='black',
                    ha='left', va='center')
    @staticmethod
    def setup_static_directories(STATIC_DIR,SUB_DIRECTORIES):
        """Create all required static subdirectories"""
        # Create base static directory if it doesn't exist
        STATIC_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create each subdirectory
        for sub_dir in SUB_DIRECTORIES:
            dir_path = STATIC_DIR / sub_dir
            dir_path.mkdir(exist_ok=True)

    @staticmethod
    def clean_old_files(
        static_dir: Union[str, Path],
        subdirectories: List[str],
        expire_days: int = 50
    ) -> None:
        """
        Remove files older than expire_days from specified subdirectories
        
        Args:
            static_dir: Path to the static directory
            subdirectories: List of subdirectories to clean
            expire_days: Number of days after which files are considered expired
        """
        base_dir = Path(static_dir)
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(days=expire_days)
        
        for subdir in subdirectories:
            target_dir = base_dir / subdir
            deleted_count = 0
            
            # Skip if directory doesn't exist
            if not target_dir.exists():
                #print(f"Directory {target_dir} does not exist - skipping")
                continue
                
            #print(f"Cleaning directory: {target_dir}")
            
            for item in target_dir.iterdir():
                if item.is_file():
                    try:
                        file_mtime = datetime.fromtimestamp(os.path.getmtime(item))
                        if file_mtime < cutoff_time:
                            os.remove(item)
                            deleted_count += 1
                            #print(f"  Removed: {item.name} (last modified: {file_mtime.date()})")
                    except Exception as e:
                        print(f"  Error processing {item.name}: {str(e)}")
    @staticmethod
    def read_sea_level_data(filename):
        """Parse the sea level data file format"""
        data = []
        with open(filename, 'r') as file:
            lines = file.readlines()
            
            start_reading = False
            for line in lines:
                line = line.strip()
                
                if line.startswith("Mth Year  Gaps"):
                    start_reading = True
                    continue
                
                if line.startswith("Mean sea level") or line.startswith("Maximum recorded"):
                    break
                
                if start_reading and line and line[0].isdigit():
                    parts = line.split()
                    if len(parts) >= 7:
                        record = {
                            "Month": int(parts[0]),
                            "Year": int(parts[1]),
                            "Gaps": int(parts[2]),
                            "Good": int(parts[3]),
                            "Minimum": float(parts[4]),
                            "Maximum": float(parts[5]),
                            "Mean": float(parts[6]),
                            "St Devn": float(parts[7]) if len(parts) > 7 else None,
                        }
                        data.append(record)
        
        return pd.DataFrame(data)

    @staticmethod
    def extract_from_dap_ugrid(url, target_time, variable_name, mesh_lon_name='mesh_node_lon',
                        mesh_lat_name='mesh_node_lat', mesh_tri_name='mesh_face_node'):
        """
        Extracts variable data, mesh node coordinates, and triangulation from an OpenDAP URL for a given time.

        Parameters
        ----------
        url : str
            OpenDAP/NetCDF URL.
        target_time : str
            ISO timestamp (e.g., '2025-05-25T23:00:00.000000000').
        variable_name : str
            Name of the variable to extract.
        mesh_lon_name, mesh_lat_name : str
            Names of longitude and latitude variables for mesh nodes.
        mesh_tri_name : str
            Name of variable containing face-node connectivity (triangles).
        
        Returns
        -------
        lon, lat : np.ndarray
            Mesh node coordinates.
        triangles : np.ndarray
            Face-node connectivity.
        data : np.ndarray
            Variable data at the closest time step, masked for NaNs.
        act_time : str
            The actual model time string used.
        """
        try:
            with xr.open_dataset(url, mask_and_scale=True, decode_cf=True) as ds:
                # Handle time axis
                if isinstance(ds.time.values[0], bytes):
                    time_str = [t.decode('utf-8') for t in ds.time.values]
                    time_dt = np.array([datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ") for t in time_str])
                else:
                    time_dt = [pd.to_datetime(t).to_pydatetime() for t in ds.time.values]
                # Convert target time
                if "." in target_time:
                    target_dt = pd.to_datetime(target_time).to_pydatetime()
                else:
                    target_dt = datetime.strptime(target_time, "%Y-%m-%dT%H:%M:%SZ")
                # Find closest time
                time_index = np.argmin([abs((t - target_dt).total_seconds()) for t in time_dt])
                act_time = str(ds.time.values[time_index])
                # Extract mesh coordinates and triangles
                lon = ds[mesh_lon_name].data
                lat = ds[mesh_lat_name].data
                triangles = ds[mesh_tri_name].data
                # Extract variable
                if variable_name not in ds.variables:
                    raise ValueError(f"Variable '{variable_name}' not found. Available: {list(ds.variables.keys())}")
                var = ds[variable_name].isel(time=time_index)
                # If variable has 3 dims (e.g. (depth, node, face)), select first
                if len(var.shape) == 3:
                    var = var.isel({var.dims[0]: 0})
                # Reduce to 1D if needed
                data = np.ma.masked_invalid(var.data.squeeze())
                return lon, lat, triangles, data, act_time
        except Exception as e:
            raise RuntimeError(f"Error accessing OpenDAP data: {str(e)}")
    
    @staticmethod
    def plot_mesh_variable(
            lon, lat, triangles, data,
            vmin=None, vmax=None, 
            cmap='viridis', var_label='Variable', units='', 
            levels=None, 
            ax=None, ax_legend=None, 
            outdir=None, fname_prefix='plot', time=None,
            show=False
        ):
        """
        Plot a variable on unstructured mesh and return plot and colorbar handles.
        If ax and ax_legend are provided, plot on them; else create new figure.
        If outdir is given, save to file.
        """
        # Triangulation and mask NaNs (for unstructured mesh)
        triang = mpl.tri.Triangulation(lon, lat, triangles)
        mask = np.any(np.isnan(data[triang.triangles]), axis=1)
        triang.set_mask(mask)

        # Setup colormap and norm
        if levels is not None:
            norm = mpl.colors.BoundaryNorm(levels, ncolors=256, clip=True)
        else:
            norm = None

        # Setup axes
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
            created_fig = True

        # Plotting
        tcf = ax.tricontourf(
            triang, data,
            levels=levels if levels is not None else 60,
            cmap=cmap, vmin=vmin, vmax=vmax, norm=norm
        )
        ax.set_aspect('equal')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        if time:
            ax.set_title(f'{var_label} {time}')
            
        # Colorbar
        if ax_legend is not None:
            cbar = plt.colorbar(tcf, cax=ax_legend)
        elif created_fig:
            cbar = plt.colorbar(tcf, ax=ax)
        else:
            cbar = None
        if cbar is not None:
            if levels is not None:
                cbar.set_ticks(levels)
            cbar.set_label(f"{var_label} {units}")

        # Save if requested
        if outdir is not None:
            os.makedirs(outdir, exist_ok=True)
            fname = f"{fname_prefix}_{time.replace(':','').replace('-','').replace('T','_')}.png" if time else f"{fname_prefix}.png"
            plt.savefig(os.path.join(outdir, fname), dpi=180)
        if show and created_fig:
            plt.show()
        if created_fig:
            plt.close()
        return tcf, cbar

    @staticmethod
    def get_custom_colormap(nColors=None, vmin=None, vmax=None, cmap_name='gist_ncar', cmap=None):
        if cmap is None:
            cmap = plt.cm.get_cmap(cmap_name)
        if nColors is not None:
            bounds = np.linspace(vmin, vmax, nColors+1)
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
            return cmap, norm, bounds
        return cmap
    
    """
    @staticmethod
    def get_layer_dataset_download_info(layer_id, time=None, root_dir=None, mapper_filename='layer_dataset_mapper.json'):

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
        if layer_id == "16":
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
    """
    @staticmethod
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
        if layer_id == "16" or layer_id == "54" or layer_id == "6" or layer_id == "27" or layer_id == "29" or layer_id == "3" or layer_id == "53" or layer_id == "28":
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

    @staticmethod
    def imperial_layers(layer_id):
        vbool = False
        if layer_id == 5:
            vbool = True
        if layer_id == 2:
            vbool = True
        if layer_id == 6:
            vbool = True
        if layer_id == 7:
            vbool = True
        if layer_id == 11:
            vbool = True
        if layer_id == 12:
            vbool = True
        if layer_id == 16:
            vbool = True
        if layer_id == 17:
            vbool = True
        if layer_id == 18:
            vbool = True
        if layer_id == 20:
            vbool = True
        if layer_id == 26:
            vbool = True
        if layer_id == 33:
            vbool = True
        if layer_id == 34:
            vbool = True
        if layer_id == 35:
            vbool = True
        if layer_id == 36:
            vbool = True
        if layer_id == 38:
            vbool = True
        if layer_id == 39:
            vbool = True
        if layer_id == 42:
            vbool = True
        if layer_id == 47:
            vbool = True
        if layer_id == 48:
            vbool = True
        if layer_id == 54:
            vbool = True
        return vbool


    @staticmethod
    def plot_ugrid_mesh(ax2,url,target_time,variable_name,min_color_plot,max_color_plot,steps,unit,title,is_direction,get_custom_colormap,extract_from_dap_ugrid,west_bound):
        lon, lat, triangles, data, act_time = extract_from_dap_ugrid(url, target_time, variable_name)
        hs_dir = None
        if is_direction:
            _, _, _, hs_dir, _ = extract_from_dap_ugrid(url, target_time, 'dirm')
        else:
            hs_dir = None
        fcmap = Path(__file__).parent / "Hs_colormap.dat"
        #fcmap = os.path.join(os.path.abspath('.'), 'Hs_colormap.dat')
        colors = np.loadtxt(fcmap)
        colors = np.hstack((colors, np.ones((len(colors[:, 1]), 1))))
        cmap = mpl.colors.ListedColormap(colors)
        nColors = int(np.ceil(max_color_plot / steps))
        cmap, norm, bounds = get_custom_colormap(nColors=nColors, vmin=min_color_plot, vmax=max_color_plot, cmap=cmap)
        ticks = bounds[::5] if len(bounds) > 10 else bounds

        levels = np.arange(min_color_plot, max_color_plot + steps, steps)
        if triangles.min() == 1:
            triangles = triangles - 1

        if (float(west_bound) < 0) and (lon.min() > 0):
            lon = np.where(lon > 180, lon - 360, lon)
        elif (float(west_bound) > 0) and (lon.min() < 0):
            lon = np.where(lon < 0, lon + 360, lon)

        lon_min, lon_max = np.nanmin(lon), np.nanmax(lon)
        lat_min, lat_max = np.nanmin(lat), np.nanmax(lat)
        lon_margin = (lon_max - lon_min) * 0.01
        lat_margin = (lat_max - lat_min) * 0.01

        plot_west = lon_min - lon_margin + 0.011
        plot_east = lon_max + lon_margin - 0.011
        plot_south = lat_min - lat_margin + 0.011
        plot_north = lat_max + lat_margin - 0.011

        ax2.cla()
        divider = make_axes_locatable(ax2)
        ax_legend = divider.append_axes("right", size="5%", pad=0.12)

        triang = mpl.tri.Triangulation(lon, lat, triangles)
        nan_mask = np.isnan(data)
        tri_mask = np.any(np.where(nan_mask[triang.triangles], True, False), axis=1)
        triang.set_mask(tri_mask)

        cs = ax2.tricontourf(triang, data, cmap=cmap, norm=norm, levels=levels)

        ax2.set_xlim(plot_west, plot_east)
        ax2.set_ylim(plot_south, plot_north)
        ax2.set_aspect('auto')
        ax2.tick_params(axis='both', labelsize=6)
        ax2.set_title(title,pad=10, fontsize=8)
        ax2.grid(
            which='both',
            color='lightgray',
            linestyle=':',
            linewidth=0.8,
            alpha=0.8
        )
        if hs_dir is not None:
            x_arr = np.linspace(lon.min(), lon.max(), 30)
            y_arr = np.linspace(lat.min(), lat.max(), 30)
            xlon, ylat = np.meshgrid(x_arr, y_arr)
            xlon = xlon.flatten()
            ylat = ylat.flatten()
            from scipy.interpolate import NearestNDInterpolator
            interp = NearestNDInterpolator(list(zip(lon, lat)), hs_dir)
            zdir = interp(xlon, ylat)
            zdir = 270 - zdir
            zdir[zdir < 0] += 360
            udir = np.cos(np.deg2rad(zdir))
            vdir = np.sin(np.deg2rad(zdir))
            ax2.quiver(xlon, ylat, udir, vdir, units='xy', zorder=2, color='k', width=0.003, headwidth=3, headlength=5, alpha=0.8)

        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = mpl.pyplot.colorbar(
            sm,
            cax=ax_legend,
            orientation='vertical',
            extend='max',
            format='{x:.2f}',
            drawedges=True,
            label=f'{unit}',
            norm=norm,
            ticks=ticks,
            boundaries=bounds
        )
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label(f'{unit}', fontsize=6)
        ax_legend.set_title("")
