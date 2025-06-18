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

matplotlib.use('Agg')  # Non-interactive backend
plt.switch_backend('Agg') 

class Plotter:
    @staticmethod
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

    @staticmethod
    def getfromDAP(url, target_time, variable_name, adjust_lon=False):
        try:
            # Open dataset with SSL verification
            with xr.open_dataset(url, engine='netcdf4', mask_and_scale=True, decode_cf=True) as ds:
                
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
            raise RuntimeError(f"Error accessing OpenDAP data: {str(e)}")

    @staticmethod
    def plot_coastline_from_geoserver(ax,m,country, style='polygon'):
        """
        Plot coastline polygons from GeoServer WFS with proper dateline handling
        
        Parameters:
        - ax: matplotlib axis object
        - geoserver_url: Base URL of GeoServer (e.g., 'http://localhost:8080/geoserver')
        - layer_name: Name of the coastline layer (e.g., 'coastline')
        - workspace: GeoServer workspace (default: 'naturalearth')
        - style: 'polygon' for filled polygons or 'line' for just boundaries
        """
        try:
            wfs_url = f"https://opmgeoserver.gem.spc.int/geoserver/spc/wfs?service=WFS&version=2.0.0&request=GetFeature&typeNames=spc:{country}_coastline&srsName=EPSG:4326&outputFormat=application/json"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            # Option 1: Using requests first then geopandas
            response = requests.get(wfs_url, headers=headers)
            gdf = gpd.read_file(response.content)

            """
            # Construct WFS request URL
            wfs_url = "https://opmgeoserver.gem.spc.int/geoserver/spc/wfs?service=WFS&version=2.0.0&request=GetFeature&typeNames=spc:"+country+"_coastline&srsName=EPSG:4326&outputFormat=application/json"
            print(wfs_url)
            # Read data directly from GeoServer
            gdf = gpd.read_file(wfs_url)
            """
            
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
                            if style == 'polygon':
                                ax.fill(x, y, color='#A9A9A9', ec='black', lw=0.5, zorder=2)
                            else:
                                ax.plot(x, y, color='black', lw=0.5, zorder=2)
                        elif poly.geom_type == 'MultiPolygon':
                            for part in poly.geoms:
                                x, y = m(part.exterior.coords.xy[0], part.exterior.coords.xy[1])
                                if style == 'polygon':
                                    ax.fill(x, y, color='#A9A9A9', ec='black', lw=0.5, zorder=2)
                                else:
                                    ax.plot(x, y, color='black', lw=0.5, zorder=2)
            
        except Exception as e:
            print(f"Error plotting coastline from GeoServer: {str(e)}")

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
    def getEEZ(ax,geojson_url,m):
        geojson_response = requests.get(geojson_url)

        if geojson_response.status_code == 200:
            # Load GeoJSON data using geopandas
            geojson_data = geojson_response.json()
            gdf = gpd.GeoDataFrame.from_features(geojson_data['features'])

            # Ensure the GeoDataFrame is in the same CRS as the map (EPSG:4326)
            gdf = gdf.set_crs('EPSG:4326', allow_override=True)

            # Convert GeoJSON coordinates to Basemap projection coordinates
            for geom in gdf.geometry:
                if geom.is_valid:  # Check if the geometry is valid
                    # For each geometry, extract the boundary (line) and convert to Basemap projection
                    for geom_line in geom.geoms:  # In case the geometry is MultiPolygon or similar
                        # Convert coordinates to Basemap projection
                        x, y = m(geom_line.xy[0], geom_line.xy[1])  # Convert coordinates to Basemap projection
                        
                        # Shift the longitudes to avoid the 180° cutoff
                        x = np.array(x)  # Ensure x is a numpy array for element-wise operations
                        x = np.where(x < 0, x + 360, x)  # For longitudes < 0 (e.g., -170°), shift to +180°
                        
                        # Plot the boundary line
                        ax.plot(x, y, marker=None, color='black', linewidth=1,linestyle='--')  # Plot the boundary line

        else:
            print("Failed to retrieve the GeoJSON data.")

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

    @staticmethod
    def plot_filled_contours(ax, ax_legend, lon, lat, data, 
                            min_color_plot, max_color_plot, steps,
                            cmap_name='RdBu_r', units='(°C)'):
        # Create fixed levels for contours
        levels = np.arange(min_color_plot, max_color_plot, steps)
        
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

    @staticmethod
    def plot_climatology(dap_url, time, ax, ax_legend, lon, lat, data, 
                            min_color_plot, max_color_plot, steps,
                            cmap_name='RdBu_r', units='(°C)'):
        # Create fixed levels for contours
        levels = np.arange(min_color_plot, max_color_plot, steps)
        
        # Plot filled contours with fixed levels
        cs = ax.contourf(
            lon, lat, data,
            levels=levels,
            cmap=cmap_name,
            extend='both'  # Adds arrows if data exceeds min/max
        )
        contour_29 = ax.contour(
            lon, lat, data,
            levels=[29],
            colors='purple',
            linewidths=2,
            linestyles='solid',
            zorder=5,
            label=f'(SST))'
        )
        clim_lon, clim_lat, sst_clim = Plotter.getfromDAP(dap_url, time, "sst_clim",adjust_lon=True)
        
        contour_clim = ax.contour(
                clim_lon, clim_lat, sst_clim,
                levels=[29],
                colors='green',
                linewidths=2,
                linestyles='solid',
                zorder=6,
                label='Climatology'
            )
        # Optional: Add labels to the contour line
        #ax.clabel(contour_29, inline=True, fontsize=8, fmt='%1.0f')
        legend_elements = [
            Line2D([0], [0], color='purple', lw=1, label=f'SST'),
            Line2D([0], [0], color='green', lw=1, label='Climatology')
        ]

        # Add legend
        ax.legend(handles=legend_elements, loc='upper right', fontsize=6)

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
            fontsize=8,
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
        cbar.ax.tick_params(labelsize=6)
        
        return cs, q, cbar

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
            x=ax2_pos.x0-0.08, 
            y=text_y-0.03,
            s=copyright_text,
            ha='left',
            **text_props
        )

        # Right-aligned dataset text (aligned with ax2's right edge)
        # Using the dataset_text variable you already have from get_title()
        fig.text(
            x=ax2_pos.x1+0.08, 
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

        # Plot city names
        for x, y, name in zip(x_coords, y_coords, filtered_cities['name']):
            ax.text(x + 0.1, y + 0.1, name,
                    fontsize=6, color='black',
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

    @staticmethod
    def plot_ugrid_mesh(ax2,url,target_time,variable_name,min_color_plot,max_color_plot,steps,unit,title,is_direction,get_custom_colormap,extract_from_dap_ugrid,west_bound):
        lon, lat, triangles, data, act_time = extract_from_dap_ugrid(url, target_time, variable_name)
        hs_dir = None
        if is_direction:
            _, _, _, hs_dir, _ = extract_from_dap_ugrid(url, target_time, 'dirm')
        else:
            hs_dir = None

        fcmap = os.path.join(os.path.abspath('.'), 'Hs_colormap.dat')
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
