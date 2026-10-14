"""
Standalone copy of the /plotter/getMap flow for "contourf_nozero_levels" layers,
with the Nino 3 / 3.4 / 4 boxes overlaid. Independent of app/main.py.

Usage (from anywhere):
    python nino_plot/plot_nino.py
    python nino_plot/plot_nino.py --time 2026-09-01T00:00:00Z --out nino_plot/out.png
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.basemap import Basemap

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "app"))
from plotter import Plotter  # noqa: E402

# name: (colour, label x, y-offset in degrees so overlapping edges stay visible)
NINO_STYLE = {
    "Nino 4":   ("#0000FF", 172, 0.0),
    "Nino 3.4": ("#00FF00", 212, 0.5),
    "Nino 3":   ("#FFFFFF", 250, 0.5),
}


def plot_nino_regions(ax, geojson_path):
    nino_gdf = gpd.read_file(geojson_path)
    nino_gdf = nino_gdf[nino_gdf["name"].isin(NINO_STYLE)]
    n34_geom = nino_gdf.loc[nino_gdf["name"] == "Nino 3.4", "geometry"].iloc[0]
    for _, row in nino_gdf.iterrows():
        colour, label_x, dy = NINO_STYLE[row["name"]]
        outline = row.geometry.boundary
        if row["name"] == "Nino 3":
            # Don't draw the Nino 3 outline inside/along Nino 3.4 (only 240-270 part)
            outline = outline.difference(n34_geom)
        for line in getattr(outline, "geoms", [outline]):
            xs, ys = line.xy
            ax.plot(xs, [y + dy for y in ys], color=colour, linewidth=1, zorder=10)
        ax.text(label_x, 5.5 + dy, row["name"], color=colour, fontsize=7, fontweight="bold",
                ha="center", va="bottom", zorder=11)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=int, default=1)
    parser.add_argument("--layer", type=int, default=35)
    parser.add_argument("--time", default="2026-09-01T00:00:00Z")
    parser.add_argument("--unit", default="metric")
    parser.add_argument("--token", default="null")
    parser.add_argument("--datasets-root", default="/app/datasets",
                        help="Local dataset root; falls back to OPeNDAP if the file isn't there")
    parser.add_argument("--out", default=str(HERE / "nino_plot.png"))
    args = parser.parse_args()

    out_path = Path(args.out).resolve()
    # Plotter uses ./vector/... relative paths, same as the server
    os.chdir(PROJECT_ROOT)

    config = Plotter.get_config_variables()
    layer_map_data = Plotter.fetch_wms_layer_data(args.layer, args.token)

    info = Plotter.get_layer_dataset_download_info(str(args.layer), args.time, args.datasets_root)
    local_file_name = "%s/%s" % (info["path"], info["file_name"]) if info != 0 else ""
    check_local = bool(local_file_name) and os.path.exists(local_file_name)
    print("Data source:", local_file_name if check_local else "OPeNDAP")

    is_imperial_layer = Plotter.imperial_layers(args.layer) and args.unit == "imperial"

    dap_url, dap_variable = Plotter.get_dap_config(layer_map_data)
    title, dataset_text = Plotter.get_title(layer_map_data, args.time)
    cmap_name, plot_type, min_color_plot, max_color_plot, steps, units, levels, discrete = \
        Plotter.get_plot_config(layer_map_data)
    if plot_type != "contourf_nozero_levels":
        print(f"Warning: layer {args.layer} is '{plot_type}', this script only reproduces contourf_nozero_levels")

    west_bound, east_bound, south_bound, north_bound, country_name, short_name = Plotter.getBBox(args.region)

    # Plotting
    fig, ax = plt.subplots(figsize=Plotter.cm2inch((15, 13)), dpi=300)
    ax.axis("off")

    ax2 = fig.add_axes([0.09, 0.2, 0.8, 0.65])
    title = "%s \n %s" % (country_name, title)
    ax2.set_title(title, pad=10, fontsize=8)

    m = Basemap(projection="cyl", llcrnrlat=south_bound, urcrnrlat=north_bound,
                llcrnrlon=west_bound, urcrnrlon=east_bound, resolution="l", ax=ax2)
    Plotter.plot_map_grid(m, south_bound, north_bound, west_bound, east_bound, args.region)

    ax2_pos = ax2.get_position()
    ax_legend = fig.add_axes([ax2_pos.x1 + 0.02, ax2_pos.y0, 0.03, ax2_pos.height])

    lon, lat, data_extract = Plotter.getfromDAP(dap_url, args.time, dap_variable, adjust_lon=True,
                                                local_path=check_local, local_path_str=local_file_name)
    cs, cbar = Plotter.plot_filled_contours_no_zero_levels(
        title, is_imperial_layer, ax=ax2, ax_legend=ax_legend, lon=lon, lat=lat, data=data_extract,
        min_color_plot=min_color_plot, max_color_plot=max_color_plot, steps=steps,
        cmap_name=cmap_name, units=units, levels=levels)

    plot_nino_regions(ax2, HERE / "nino_regions.geojson")

    # Colorbar tick formatting (generic branch from getMap)
    if cbar is not None:
        if is_imperial_layer:
            for t in cbar.ax.get_yticklabels():
                t.set_horizontalalignment("left")
            cbar.ax.tick_params(axis="y", pad=2, length=0)
        else:
            step_compact = f"{float(steps):g}"
            step_decimals = len(step_compact.split(".")[1]) if "." in step_compact else 0
            cbar_decimals = max(1, step_decimals)
            cbar_width = 5 + cbar_decimals

            def _cbar_fmt(x, pos, d=cbar_decimals, w=cbar_width):
                if abs(x) < 1e-12:
                    x = 0.0
                label = f"{x:{w}.{d}f}"
                if float(label) == 0.0:
                    label = label.replace("-", "")
                return label

            cbar.ax.yaxis.set_major_formatter(FuncFormatter(_cbar_fmt))
            for t in cbar.ax.get_yticklabels():
                t.set_horizontalalignment("left")
            cbar.ax.tick_params(axis="y", pad=-1, length=0)

    Plotter.add_logo_and_footer(fig=fig, ax=ax, ax2=ax2, ax2_pos=ax2_pos, region=1,
                                copyright_text=config.copyright_text, footer_text=config.footer_text,
                                dataset_text=dataset_text)

    # EEZ + coastline
    if short_name == "PAC":
        Plotter.getEEZ(ax2, m, local_path="./vector/final_eez_country/PAC_EEZ_v3.zip")
        m.drawcoastlines(linewidth=0.3)
        m.fillcontinents(color="#A9A9A9", lake_color="white")
        m.drawcountries()
    else:
        Plotter.getEEZ(ax2, m, local_path="./vector/final_eez_country/%s.zip" % short_name)
        Plotter.plot_coastline_from_geoserver(ax2, m, "./vector/Individual_coastline/%s_coastline.zip" % short_name)

    Plotter.plot_city_names(ax2, m, short_name, PROJECT_ROOT / "app" / "config" / "pac_names.json")

    plt.savefig(out_path, format="png", bbox_inches="tight", pad_inches=0.1, dpi=300)
    plt.close(fig)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
