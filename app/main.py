import sys
from pathlib import Path
# Add /app to Python path (since Docker copies files to /app)
sys.path.append(str(Path(__file__).parent))
from typing import Union
from fastapi import FastAPI, Request, Response, APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
import io
import matplotlib.pyplot as plt  # Correct import
import numpy as np
from plotter import Plotter
from mpl_toolkits.basemap import Basemap
from fastapi.responses import FileResponse
import hashlib
from datetime import datetime
import os
from datetime import datetime, timedelta
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from scipy.stats import linregress
import pandas as pd
import requests
import matplotlib.pyplot as plt
from PIL import Image
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import io, os
import httpx
from pathlib import Path
from urllib.parse import parse_qs, urlencode
import hashlib
import zlib
import gzip
from typing import Dict
import logger

app = FastAPI(
    docs_url="/plotter/docs",
    redoc_url="/plotter/redoc",
    openapi_url="/plotter/openapi.json",
    favicon_url="/plotter/favicon.ico"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
ocean_router = APIRouter(prefix="/plotter")

# Configuration
#BASE_DIR = Path(__file__).parent.parent
#STATIC_DIR = BASE_DIR / "app" / "static"
STATIC_DIR = Path("/app/app/static") 
STATIC_DIR.mkdir(exist_ok=True, parents=True)

SUB_DIRECTORIES_TO_CLEAN = ["maps", "tide", "thredds"]  
# List of subdirectories you want to create
SUB_DIRECTORIES = [
    "maps",
    "legend",
    "tide",
    "basemap",
    "eez",
    "pacificnames",
    "thredds",
    "coastline"
]
CACHE_EXPIRE_DAYS = 50


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    favicon_path = Path(__file__).parent / "icon.ico"
    return FileResponse(favicon_path) if favicon_path.exists() else JSONResponse(
        {"error": "Not found"}, status_code=404)


@app.on_event("startup")
async def startup_event():
    """Initialize application state."""
    try:
        Plotter.setup_static_directories(STATIC_DIR, SUB_DIRECTORIES)
        
        # Initialize scheduler for cleanup tasks
        scheduler = BackgroundScheduler()
        scheduler.add_job(
            cleanup_old_files,
            'interval',
            days=1,
            next_run_time=datetime.now()
        )
        scheduler.start()
    except Exception as e:
        raise

async def cleanup_old_files():
    """Cleanup old files from static directories."""
    cutoff = datetime.now() - timedelta(days=30)
    for sub_dir in SUB_DIRECTORIES:
        dir_path = STATIC_DIR / sub_dir
        if dir_path.exists():
            for file in dir_path.iterdir():
                if file.is_file() and datetime.fromtimestamp(file.stat().st_mtime) < cutoff:
                    file.unlink()
                    logger.info(f"Deleted old file: {file}")


#ROOT
@ocean_router.get("/")
def read_root():
    return {"Message": "Pacific ocean portal plotter, powered by OpenAPI"}

#MAPS FOR OCEAN PORTAL
@ocean_router.get("/getMap")
async def generate_plot_2(request: Request,region: int = 1,layer_map: int = 2,time: str = '2025-05-14T12:00:00Z',use_cache: bool = True):
    # Generate unique filename based on parameters
    params_hash = hashlib.md5(f"{region}_{layer_map}_{time}".encode()).hexdigest()
    filename = "plot_%s_%s_%s.png" % (region,layer_map,params_hash)
    #filename = f"plot_{params_hash}.png"
    filepath = STATIC_DIR / "maps" / filename

    # Check cache first
    if use_cache and filepath.exists():
        return FileResponse(filepath, media_type="image/png")

    try:
        config = Plotter.get_config_variables()
        #####PARAMETER#####
        #region = 3
        layer_id = layer_map
        #time= add_z_if_needed("2024-10-01T00:00:00Z")
        resolution = "l"
        #####PARAMETER#####

        layer_map_data = Plotter.fetch_wms_layer_data(layer_id)

        #REMOVE DEMO
        #time = Plotter.demo_time(layer_map_data)
        #REMOVE DEMO
        #####MAIN#####
        dap_url, dap_variable = Plotter.get_dap_config(layer_map_data)
        title, dataset_text = Plotter.get_title(layer_map_data,time)
        cmap_name, plot_type, min_color_plot, max_color_plot, steps, units, levels, discrete = Plotter.get_plot_config(layer_map_data)
        west_bound, east_bound, south_bound, north_bound, country_name, short_name = Plotter.getBBox(region)
        if short_name == "PAC":
            resolution = "l"
        eez_url = Plotter.getCountryData(region)

        ##PLOTTING
        figsize = Plotter.cm2inch((15,13))
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        ax.axis('off')

        ax2 = fig.add_axes([0.09, 0.2, 0.8, 0.65])
        title = "%s \n %s" % (country_name,title)
        ax2.set_title(title, pad=10, fontsize=8)

        m = Basemap(projection='cyl', llcrnrlat=south_bound, urcrnrlat=north_bound, 
                    llcrnrlon=west_bound, urcrnrlon=east_bound, resolution=resolution, ax=ax2)

        Plotter.plot_map_grid(m, south_bound, north_bound, west_bound, east_bound,region)

        # Add colorbar to ax2
        ax2_pos = ax2.get_position()
        ax_legend_width = 0.03  # Width of the legend
        ax_legend_gap = 0.1    # Gap between ax2 and ax_legend
        ax_legend = fig.add_axes([ax2_pos.x1 +0.02, ax2_pos.y0, ax_legend_width, ax2_pos.height])



        ##MAIN PLOTTER
        if plot_type == "contourf":
            lon, lat, data_extract = Plotter.getfromDAP(dap_url, time, dap_variable,adjust_lon=True)
            cs, cbar = Plotter.plot_filled_contours(ax=ax2, ax_legend=ax_legend, lon=lon, lat=lat, data=data_extract,\
                min_color_plot=min_color_plot, max_color_plot=max_color_plot, steps=steps, cmap_name=cmap_name, units=units
            )
        elif plot_type == "contourf_nozero":
            lon, lat, data_extract = Plotter.getfromDAP(dap_url, time, dap_variable,adjust_lon=True)
            cs, cbar = Plotter.plot_filled_contours_no_zero(ax=ax2, ax_legend=ax_legend, lon=lon, lat=lat, data=data_extract,\
                min_color_plot=min_color_plot, max_color_plot=max_color_plot, steps=steps, cmap_name=cmap_name, units=units
            )
        elif plot_type == "pcolormesh":
            lon, lat, data_extract = Plotter.getfromDAP(dap_url, time, dap_variable,adjust_lon=True)
            cs, cbar = Plotter.plot_filled_pcolor(ax=ax2, ax_legend=ax_legend, lon=lon, lat=lat, data=data_extract,\
                min_color_plot=min_color_plot, max_color_plot=max_color_plot, steps=steps, cmap_name=cmap_name, units=units
            )
        elif plot_type == "wave_with_dir":
            wave_height_varib, wave_dir_varib = dap_variable.split(',')
            lon, lat, wave_height = Plotter.getfromDAP(dap_url, time, wave_height_varib, adjust_lon=True)
            _, _, wave_dir = Plotter.getfromDAP(dap_url, time, wave_dir_varib, adjust_lon=True)
            step = 10
            if int(region) == 1:
                step = 30
            cs, q, cbar = Plotter.plot_wave_field(ax2, ax_legend, m, lon, lat, wave_height, wave_dir,\
                                    min_color_plot, max_color_plot, steps,region, step, cmap_name=cmap_name, units=units)
        elif plot_type == "discrete":
            lons, lats, bleaching_data = Plotter.getfromDAP(dap_url, time, dap_variable, adjust_lon=True)
            splitBy_ = discrete.split("_")
            if len(splitBy_) > 1:
                colors = splitBy_[0]
                split_1 = splitBy_[1]
                range_nums, range_name = split_1.split('%')
                color_arr = np.array(eval(colors), dtype=str)
                range_nums_arr = np.array(eval(range_nums), dtype=str)
                range_name_arr = np.array(eval(range_name), dtype=str)

                cs, cbar = Plotter.plot_discrete_map_ranges(ax=ax2, ax_legend=ax_legend, lons=lons, lats=lats, bleaching_data=bleaching_data,\
                    cmap_colors=color_arr, colorbar_labels=range_name_arr, ranges=range_nums_arr)
            else:
                tmp_color, tmp_label = discrete.split('-')
                color_arr = np.array(eval(tmp_color), dtype=str)
                label_arr = np.array(eval(tmp_label), dtype=str)

                cs, cbar = Plotter.plot_discrete_map(ax=ax2, ax_legend=ax_legend, lons=lons, lats=lats, bleaching_data=bleaching_data,\
                    cmap_colors=color_arr, colorbar_labels=label_arr)

        elif plot_type == "levels_pcolor":
            lons, lats, chl_data = Plotter.getfromDAP(dap_url, time, dap_variable, adjust_lon=True)
            Plotter.plot_levels_pcolor(ax2, ax_legend, lons, lats, chl_data,cmap_name, units=units,levels=levels)

        elif plot_type == "levels_contourf":
            lons, lats, chl_data = Plotter.getfromDAP(dap_url, time, dap_variable, adjust_lon=True)
            Plotter.plot_levels_contour(ax2, ax_legend, lons, lats, chl_data,cmap_name, units=units,levels=levels,)

        elif plot_type == "climate":
            split_varib = dap_variable.split(",")
            lon, lat, data_extract = Plotter.getfromDAP(dap_url, time, split_varib[0],adjust_lon=True)
            cs, cbar = Plotter.plot_climatology(dap_url,time,ax=ax2, ax_legend=ax_legend, lon=lon, lat=lat, data=data_extract,\
                min_color_plot=min_color_plot, max_color_plot=max_color_plot, steps=steps, cmap_name=cmap_name, units=units
            )
        elif plot_type == "currents":
            lon, lat, uo = Plotter.getfromDAP(dap_url, time, 'uo', adjust_lon=True)
            _, _, vo = Plotter.getfromDAP(dap_url, time, 'vo', adjust_lon=True)
            pcm, quiv, cbar = Plotter.plot_current_magnitude(
                ax=ax2,
                ax_legend=ax_legend,
                lon=lon,
                lat=lat,
                uo=uo,
                vo=vo,
                region=region,
                min_color_plot=min_color_plot,
                max_color_plot=max_color_plot,
                steps=0.1,
                cmap_name=cmap_name,
                units=units,
                show_arrows=True,
                arrow_scale=3,      # Replaces arrow_size (higher = bigger arrows)
                density=50,          # More arrows than before (since we're scaling by magnitude)
                arrow_color='white',  # Options: color string, or 'magnitude' to color by speed
                min_speed=0.05       # Hide very weak currents (adjust based on your data range)
            )


        #ADD LOGO AND FOOTER
        Plotter.add_logo_and_footer(fig=fig, ax=ax, ax2=ax2, ax2_pos=ax2_pos, region=1, copyright_text=config.copyright_text,\
            footer_text=config.footer_text,dataset_text=dataset_text)

        #PLOT EEZ
        Plotter.getEEZ(ax2,eez_url,m)
        if short_name == "PAC":
            m.drawcoastlines(linewidth=0.3)
            m.fillcontinents(color='#A9A9A9', lake_color='white')
            m.drawcountries()
        else:
            Plotter.plot_coastline_from_geoserver(ax2,m,short_name)

        #Plotter.plot_coastline_from_geoserver(ax2,m,short_name)
        Plotter.plot_city_names(ax2,m,short_name, Path(__file__).parent.parent / "app" / "config" / "pac_names.json")
        # Save directly to file if caching
        if use_cache:
            plt.savefig(filepath, format="png", bbox_inches='tight', pad_inches=0.1, dpi=300)
            plt.close(fig)
            return FileResponse(filepath, media_type="image/png")
        else:
            # If not using cache, use BytesIO
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0.1, dpi=300)
            plt.close(fig)
            buf.seek(0)
            return StreamingResponse(buf, media_type="image/png", headers={"Cache-Control": "no-store"})
    except Exception as e:
        plt.close('all')  # Ensure all figures are closed on error
        return {"error": str(e)}

#LEGEND FOR OCEAN PORTAL
@ocean_router.get("/GetLegendGraphic")
async def generate_legend(request: Request,layer_map: int = 1,mode: str = '',min_color: float = 0.0,max_color: float = 0.0,step: float = 0.0,color: str = '',unit: str = '',use_cache: bool = True):
    # Generate a unique cache key based on all parameters
    cache_params = f"{layer_map}_{mode}_{min_color}_{max_color}_{step}_{color}_{unit}"
    #cache_key = "legend_" + hashlib.md5(cache_params.encode()).hexdigest()
    cache_key = "legend_%s_%s" % (layer_map, hashlib.md5(cache_params.encode()).hexdigest())
    cache_path = STATIC_DIR / "legend" / f"{cache_key}.png"

    # Return cached file if exists and caching is enabled
    if use_cache and cache_path.exists():
        return FileResponse(cache_path, media_type="image/png")

    fig, ax = plt.subplots(figsize=(6, 1), facecolor='white')
    fig.subplots_adjust(bottom=0.5)

    if mode == 'coral_bleaching':
        # Coral bleaching configuration
        colors = ['#ADD8E6', '#FFFF00', '#FFA500', '#FF0000', '#800000']
        labels = ['No Stress', 'Watch', 'Warning', 'Alert Level 1', 'Alert Level 2']
        bounds = [0, 1, 2, 3, 4, 5]
        title = 'Coral Bleaching Alert Level'
        
    elif mode == 'marine_heat_wave':
        # Marine Heat Wave configuration
        colors = ['#B0E0E6', '#FFFF00', '#FFA500', '#FF0000', '#8B0000']
        labels = ['0', '1', '2', '3', '4']
        bounds = [0, 1, 2, 3, 4, 5]
        title = 'Marine Heat Wave Level'
        
    elif mode == 'decile':
        # Decile configuration
        colors = ['#00305A', '#4A89AF', '#A9C8DA', '#FFFFFF', '#F4B7A1', '#A8413F', '#5B001F']
        labels = ['Lowest\non Record', 'Very much\nbelow average', 'Below\nAverage', 
                'Average', 'Above\nAverage', 'Very much\nabove average', 'Highest\non record']
        bounds = [0, 1, 2, 3, 4, 5, 6, 7]
        title = 'Decile Categories'
        # Adjust figure size to accommodate multi-line labels
        fig.set_size_inches(9, 1.8)
        fig.subplots_adjust(bottom=0.45)
        
    else:
        # Standard continuous colorbar
        cmap = getattr(plt.cm, color)
        norm = mpl.colors.Normalize(vmin=float(min_color), vmax=float(max_color))
        ticks = np.arange(float(min_color), float(max_color) + float(step)/2, float(step))
        
        cb = mpl.colorbar.ColorbarBase(
            ax,
            cmap=cmap,
            norm=norm,
            orientation='horizontal',
            extend='both',
            ticks=ticks,
            spacing='uniform'
        )
        # Corrected this line - using set_ticklabels instead of set_xticklabels
        cb.set_ticklabels([f'{tick:g}' for tick in ticks])
        cb.set_label(unit, labelpad=10)
        buf = io.BytesIO()
        plt.savefig(cache_path, bbox_inches='tight', pad_inches=0.1, dpi=150, facecolor='white')
        plt.close()
        
        if use_cache:
            return FileResponse(cache_path, media_type="image/png")
        else:
            return FileResponse(cache_path, media_type="image/png", headers={"Cache-Control": "no-store"})

        #plt.savefig(fname, bbox_inches='tight', pad_inches=0.1, dpi=200, facecolor='white')
        #plt.close()

    # Create discrete colormap for categorical data
    cmap = ListedColormap(colors)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # Calculate tick positions (center of each color band)
    tick_positions = [i + 0.5 for i in range(len(labels))]

    # Create colorbar with discrete colors
    cb = mpl.colorbar.ColorbarBase(
        ax,
        cmap=cmap,
        norm=norm,
        orientation='horizontal',
        boundaries=bounds,
        ticks=tick_positions,
        spacing='uniform',
        extend='neither'
    )

    # Configure ticks and labels
    cb.set_ticks(tick_positions)
    cb.set_ticklabels(labels)
    if mode == 'standard':
        cb.set_label(title, labelpad=10)

    # Adjust label formatting
    if mode == 'decile':
        cb.ax.tick_params(labelsize=8, rotation=0)  # No rotation, just multi-line
        # Center-align the multi-line labels
        for label in cb.ax.get_xticklabels():
            label.set_horizontalalignment('center')
            label.set_verticalalignment('top')
            label.set_linespacing(0.8)
    else:
        cb.ax.tick_params(labelsize=9)

    # Save with white background
    buf = io.BytesIO()
    plt.savefig(cache_path, bbox_inches='tight', pad_inches=0.1, dpi=150, facecolor='white')
    plt.close()

    # Return the cached file
    if use_cache:
        return FileResponse(cache_path, media_type="image/png")
    else:
        # If not using cache, return with headers to prevent client caching
        return FileResponse(cache_path, media_type="image/png", headers={"Cache-Control": "no-store"})

#CALCULATE MIN MAX TIDE HISTORICAL DATA
@ocean_router.get("/tide_hindcast")
async def generate_tide_hindcast(country: str,location: str,station_id: str,use_cache: bool = False):
    """
    Replacement for ocean-cgi.spc.int/cgi-bin/tide_hindcast.py
    Example request: /tide_hindcast?country=Fiji&location=Suva&station_id=IDO70063
    """
    # Generate cache key
    cache_params = f"{country}_{location}_{station_id}"
    cache_key = "tide_" + hashlib.md5(cache_params.encode()).hexdigest()
    cache_path = STATIC_DIR / "tide" / f"{cache_key}.png"

    # Return cached file if exists and caching is enabled
    if use_cache and cache_path.exists():
        return FileResponse(cache_path, media_type="image/png")

    try:
        # Download data file
        url = f"http://reg.bom.gov.au/ntc/{station_id}/{station_id}SLD.txt"
        temp_file = STATIC_DIR / "tide" / f"{station_id}SLD.txt"
        
        response = requests.get(url)
        response.raise_for_status()
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(response.text)

        # Process data
        df = Plotter.read_sea_level_data(temp_file)
        df["Date"] = pd.to_datetime(df.assign(Day=1)[["Year", "Month", "Day"]])

        # Calculate trend
        x = np.arange(len(df))
        slope_mean, intercept_mean, _, _, _ = linregress(x, df["Mean"])
        slope_mm_per_year = slope_mean * 12 * 1000
        df["Mean_Trend"] = intercept_mean + slope_mean * x

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6.5))
        ax.plot(df["Date"], df["Mean"], label="Mean", color="blue")
        ax.plot(df["Date"], df["Maximum"], label="Maximum", color="red", alpha=0.5)
        ax.plot(df["Date"], df["Minimum"], label="Minimum", color="green", alpha=0.5)
        ax.plot(df["Date"], df["Mean_Trend"], "--", color="blue", label="Mean Trend")

        # Add annotations
        ax.text(
            df["Date"].iloc[5], max(df["Mean"]),
            f"Mean Trend Slope: {slope_mm_per_year:.2f} mm/year",
            fontsize=12, color="blue", bbox=dict(facecolor="white", alpha=0.6)
        )

        # Set axis limits
        ax.set_xlim([df["Date"].min(), df["Date"].max()])
        ax.set_ylim([
            min(df["Minimum"].min(), df["Mean_Trend"].min()) * 0.98,
            max(df["Maximum"].max(), df["Mean_Trend"].max()) * 1.02
        ])

        # Formatting
        ax.set_xlabel("Year")
        ax.set_ylabel("Sea Level (m)")
        ax.legend()
        ax.set_title(f"Relative Sea Level\n{country} - {location}")
        ax.grid(True)

        # Add footer text
        ax2_pos = ax.get_position()
        ax.text(
            -0.08, ax2_pos.y0-0.17, "© Pacific Community (SPC) 2025",
            transform=ax.transAxes, fontsize=7, verticalalignment='top'
        )
        ax.text(
            -0.08, ax2_pos.y0-0.195, "Climate and Ocean Support Program in the Pacific (COSPPac)",
            transform=ax.transAxes, fontsize=7, verticalalignment='top'
        )

        # Save to cache
        plt.savefig(cache_path, dpi=300, bbox_inches='tight')
        plt.close()

        return FileResponse(
            cache_path,
            media_type="image/png",
            headers={"Cache-Control": "no-store"} if not use_cache else None
        )

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Data download failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plot generation failed: {str(e)}")
    finally:
        if 'temp_file' in locals() and temp_file.exists():
            temp_file.unlink()

#CACHE EEZ
GEOSERVER_URL = "https://geonode.pacificdata.org/geoserver"
TILE_SERVER_URL = "https://spc-osm.spc.int/tile/{z}/{x}/{y}.png"
GEOSERVER_URL_ONPREM = "https://opmgeoserver.gem.spc.int/geoserver"
THREDDS_SERVER_URL = "https://ocean-thredds01.spc.int/thredds/wms"


def compress_data(data: bytes) -> bytes:
    """Compress data using gzip with maximum compression"""
    return gzip.compress(data, compresslevel=9)

def decompress_data(data: bytes) -> bytes:
    """Decompress gzipped data"""
    return gzip.decompress(data)

def normalize_wms_params(params: Dict) -> str:
    """Normalize WMS parameters for consistent caching"""
    standard_keys = [
        'service', 'version', 'request', 'layers',
        'styles', 'bbox', 'width', 'height', 'srs',
        'format', 'transparent'
    ]
    return urlencode({k: params[k][0] for k in standard_keys if k in params})

def get_cache_path(path: str, query: str) -> Path:
    """Generate cache path based on layer type"""
    # Extract layer type from path
    if "/cache/eez/" in path:
        layer_type = "eez"
    elif "/cache/thredds/" in path:
        layer_type = "thredds"
    elif "/cache/basemap/" in path:
        layer_type = "basemap"
    elif "/cache/pacificnames/" in path:
        layer_type = "pacificnames"
    elif "/cache/coastline/" in path:
        layer_type = "coastline"
    else:
        layer_type = "other"
    
    cache_key = hashlib.md5(f"{path}?{query}".encode()).hexdigest()
    return STATIC_DIR / layer_type / f"{cache_key}.gz"

def get_tile_cache_path(z: int, x: int, y: int) -> Path:
    """Generate cache path for tiles using z/x/y structure"""
    cache_dir = STATIC_DIR / "basemap" / str(z) / str(x)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{y}.png.gz"

def normalize_thredds_params(query_params):
    """Normalize THREDDS/WMS query parameters to ensure consistent caching."""
    normalized = {}
    
    # Standard WMS parameters to keep
    wms_params = ['request', 'layers', 'styles', 'format', 'transparent', 
                 'version', 'time', 'width', 'height', 'srs', 'bbox', 
                 'colorscalerange', 'opacity']
    
    for param in wms_params:
        if param in query_params:
            normalized[param] = query_params[param]
    
    # Special handling for time parameter to normalize format
    if 'time' in normalized:
        time_val = normalized['time'][0]
        try:
            # Convert to ISO format if not already
            dt = datetime.fromisoformat(time_val.replace('Z', '+00:00'))
            normalized['time'] = [dt.strftime('%Y-%m-%dT%H:%M:%SZ')]
        except ValueError:
            pass
    
    # Sort parameters for consistent URL generation
    return urlencode(normalized, doseq=True)

@ocean_router.get("/cache/{layer_type}/{path:path}")
async def cached_layer_proxy(request: Request, layer_type: str):
    """Handle cached requests for different layer types"""
    return await cache_middleware(request, lambda r: JSONResponse(
        {"error": "Middleware didn't handle this request"},
        status_code=500
    ))

@app.middleware("http")
async def cache_middleware(request: Request, call_next):
    path = request.url.path
    # Handle basemap tile requests
    if path.startswith("/plotter/cache/basemap/"):
        try:
            # Extract z/x/y from path like /plotter/cache/basemap/{z}/{x}/{y}.png
            parts = path.split('/')
            if len(parts) < 7:
                return JSONResponse(
                    {"error": "Invalid tile URL format"},
                    status_code=400
                )
                
            z, x, y = int(parts[4]), int(parts[5]), int(parts[6].split('.')[0])
            
            cache_path = get_tile_cache_path(z, x, y)
            
            # Serve from cache if available
            if cache_path.exists():
                file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
                if file_age < timedelta(days=CACHE_EXPIRE_DAYS):
                    with open(cache_path, 'rb') as f:
                        return StreamingResponse(
                            iter([decompress_data(f.read())]),
                            media_type="image/png",
                            headers={"X-Cache-Status": "HIT"}
                        )
                else:
                    cache_path.unlink()  # Remove expired cache
            
            # Fetch from tile server if not in cache
            tile_url = TILE_SERVER_URL.format(z=z, x=x, y=y)
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(tile_url)
                response.raise_for_status()
                
                # Cache and return the response
                cache_path.write_bytes(compress_data(response.content))
                
                return StreamingResponse(
                    iter([response.content]),
                    media_type="image/png",
                    headers={"X-Cache-Status": "MISS"}
                )
                
        except Exception as e:
            return JSONResponse(
                {"error": f"Failed to fetch tile: {str(e)}"},
                status_code=500
            )
    
    elif "/plotter/cache/pacificnames/" in path:
        if "/wms" in path:
            query_params = parse_qs(str(request.query_params))
            
            if query_params.get('request') == ['GetMap']:
                try:
                    # Ensure layers parameter includes the OSM Pacific Islands layer
                    if 'layers' not in query_params:
                        query_params['layers'] = ['spc:osm_pacific_islands_2']
                    
                    normalized_params = normalize_wms_params(query_params)
                    cache_path = get_cache_path(path, normalized_params)
                    
                    # Ensure cache directory exists
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Serve from cache if available
                    if cache_path.exists():
                        file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
                        if file_age < timedelta(days=CACHE_EXPIRE_DAYS):
                            with open(cache_path, 'rb') as f:
                                return StreamingResponse(
                                    iter([decompress_data(f.read())]),
                                    media_type="image/png",
                                    headers={"X-Cache-Status": "HIT"}
                                )
                        else:
                            cache_path.unlink()  # Remove expired cache
                    
                    # Construct GeoServer URL
                    geoserver_path = path.split("/cache/pacificnames/")[1]
                    geoserver_url = f"{GEOSERVER_URL_ONPREM}/spc/wms?{normalized_params}"
                    
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        response = await client.get(geoserver_url)
                        response.raise_for_status()
                        
                        cache_path.write_bytes(compress_data(response.content))
                        
                        return StreamingResponse(
                            iter([response.content]),
                            media_type=response.headers.get('content-type', 'image/png'),
                            headers={"X-Cache-Status": "MISS"}
                        )
                        
                except Exception as e:
                    return JSONResponse(
                        {"error": f"GeoServer request failed: {str(e)}"},
                        status_code=500
                    )
    # Handle different cache types
    elif "/cache/thredds/" in path:
        query_params = parse_qs(str(request.query_params))
        
        # Handle both GET and HEAD requests for WMS
        if query_params.get('request') in [['GetMap'], ['GetCapabilities']]:
            try:
                # Extract the original THREDDS path from the cache path
                thredds_path = path.split("/cache/thredds/")[1]
                
                # Normalize and prepare parameters for the THREDDS server
                normalized_params = normalize_thredds_params(query_params)
                
                # Construct cache path using both the dataset path and parameters
                cache_path = get_cache_path(path, normalized_params)
                
                # For HEAD requests, just check cache status
                if request.method == "HEAD":
                    cache_exists = cache_path.exists()
                    if cache_exists:
                        file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
                        if file_age < timedelta(days=CACHE_EXPIRE_DAYS):
                            return Response(headers={"X-Cache-Status": "HIT"})
                        else:
                            cache_path.unlink()
                            return Response(headers={"X-Cache-Status": "MISS"})
                    return Response(headers={"X-Cache-Status": "MISS"})
                
                # For GET requests, handle full caching logic
                if request.method == "GET":
                    # Ensure cache directory exists
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Serve from cache if available
                    if cache_path.exists():
                        file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
                        if file_age < timedelta(days=CACHE_EXPIRE_DAYS):
                            with open(cache_path, 'rb') as f:
                                return StreamingResponse(
                                    iter([decompress_data(f.read())]),
                                    media_type="image/png",
                                    headers={"X-Cache-Status": "HIT"}
                                )
                        else:
                            cache_path.unlink()  # Remove expired cache
                    
                    # Construct the full THREDDS server URL
                    thredds_url = f"https://ocean-thredds01.spc.int/thredds/wms/{thredds_path}?{normalized_params}"
                    
                    # Forward request to THREDDS server
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        response = await client.get(thredds_url)
                        response.raise_for_status()
                        
                        # Cache the response
                        cache_path.write_bytes(compress_data(response.content))
                        
                        return StreamingResponse(
                            iter([response.content]),
                            media_type=response.headers.get('content-type', 'image/png'),
                            headers={
                                "X-Cache-Status": "MISS",
                                "X-THREDDS-Cache": "true"
                            }
                        )
                    
            except httpx.HTTPStatusError as e:
                return JSONResponse(
                    {"error": f"THREDDS server error: {str(e)}"},
                    status_code=e.response.status_code
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                return JSONResponse(
                    {"error": f"Internal server error: {str(e)}"},
                    status_code=500
                )
        
        # Return 200 OK for OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            return Response(status_code=200)
        
        # Return 404 for unsupported requests
        return Response(status_code=404)
    
    elif "/cache/eez/" in path or "/cache/basemap/" in path or "/cache/pacificnames/" in path or "/cache/coastline/" in path:
        if "/wms" in path or "/ows" in path:
            query_params = parse_qs(str(request.query_params))
            
            if query_params.get('request') == ['GetMap']:
                try:
                    normalized_params = normalize_wms_params(query_params)
                    cache_path = get_cache_path(path, normalized_params)
                    
                    # Ensure cache directory exists
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Serve from cache if available
                    if cache_path.exists():
                        file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
                        if file_age < timedelta(days=CACHE_EXPIRE_DAYS):
                            with open(cache_path, 'rb') as f:
                                return StreamingResponse(
                                    iter([decompress_data(f.read())]),
                                    media_type="image/png",
                                    headers={"X-Cache-Status": "HIT"}
                                )
                        else:
                            cache_path.unlink()  # Remove expired cache
                    
                    # Determine actual GeoServer URL
                    geoserver_path = path.split("/cache/")[1].split("/", 1)[1]
                    geoserver_url = f"{GEOSERVER_URL}/{geoserver_path}?{normalized_params}"
                    
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        response = await client.get(geoserver_url)
                        response.raise_for_status()
                        
                        cache_path.write_bytes(compress_data(response.content))
                        
                        return StreamingResponse(
                            iter([response.content]),
                            media_type=response.headers.get('content-type', 'image/png'),
                            headers={"X-Cache-Status": "MISS"}
                        )
                        
                except Exception as e:
                    return JSONResponse(
                        {"error": "Internal server error"},
                        status_code=500
                    )
    
    return await call_next(request)



app.include_router(ocean_router)