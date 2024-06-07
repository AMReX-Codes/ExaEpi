"""
Generates movie frames based on given .shp files and 
plotxxxxx/ data results directly from simulation.
These frames will generally lie in ./frames/, and can be converted
to movie using ffmpeg, e.g.
ffmpeg -framerate 1 -r 30 -i frames/frame%05d.png -pix_fmt yuv420p movie.mp4
Can adjust inputs directly at bottom of file or as command-line input:

python generate_frames.py [sim results dir] [.shp dir] [output dir (optional)]

Endpoints of color range can be set in or passed into generate_plot,
or in main function, as necessary.
"""

import numpy as np

import yt
from yt.frontends.boxlib.data_structures import AMReXDataset

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
import shapefile
# import fiona

import os
import sys

def get_per_data(name: str):
    """Generate a dataframe mapping FIPS codes to percent of population infected.
    (total population as measured by # of agents in simulation, not by census info)
    Input: directory of yt data (often in form of pltxxxxx)
    Output: pd.DataFrame with columns "FIPS" and "per"
    """

    ds = AMReXDataset(name)
    ad = ds.all_data()

    counties = np.unique(ad["FIPS"])

    def get_per(county):
        mask = ad["FIPS"] == county
        total = ad["total"][mask].sum()
        # log(1 + infected/total), defaulting to 0 if total == 0
        return np.log(1 + ad["infected"][mask].sum().value / total.value) if total != 0 else 0.0

    per_df = pd.DataFrame()
    per_df["FIPS"] = counties.d.astype(int)
    per_df["per"] = per_df.apply(lambda row: get_per(row["FIPS"]), axis=1)
    ds.close()
    return per_df

def get_raw_data(name: str):
    """Generate a dataframe mapping FIPS codes to log of population infected.
    Input: directory of yt data (often in form of pltxxxxx)
    Output: pd.DataFrame with columns "FIPS" and "per"
    """

    ds = AMReXDataset(name)
    ad = ds.all_data()

    counties = np.unique(ad["FIPS"])

    def get_raw(county):
        mask = ad["FIPS"] == county
        return np.log(1 + ad["infected"][mask].sum().value)

    raw_df = pd.DataFrame()
    raw_df["FIPS"] = counties.d.astype(int)
    raw_df["per"] = raw_df.apply(lambda row: get_raw(row["FIPS"]), axis=1)
    ds.close()
    return raw_df

# example: prefix = "../data/San_Francisco_Bay_Region_2020_Census_Tracts/region_2020_censustract"
def get_gdf(prefix: str):
    """Generates a GeoDataFrame from .shp, .dbf, and .prj files.
    All files must be present and must begin with prefix
    """

    # with open(prefix + ".shp", "rb") as shp, \
    # open(prefix + ".dbf", "rb") as dbf, \
    # open(prefix + ".prj", "rb") as prj:
    #     r = shapefile.Reader(shp=shp, dbf=dbf, prj=prj)
    #     attributes, geometry = [], []
    #     field_names = [field[0] for field in r.fields[1:]]  
    #     for row in r.shapeRecords():  
    #         geometry.append(shape(row.shape.__geo_interface__))  
    #         attributes.append(dict(zip(field_names, row.record)))  
    #     r.close()
    # gdf = gpd.GeoDataFrame(data = attributes, geometry = geometry)
    
    # Below code requires a .shx file but seems to be much faster than above
    # building .shx file can easily be done for you by running code inside a
    # with fiona.Env(SHAPE_RESTORE_SHX = "YES"):
    # block (remember to import fiona explicitly)
    gdf = gpd.read_file(prefix + ".shp", driver="esri")
    
    cols = list(gdf.columns)

    # Try to find county code columns. CAPS for CA/US state data, lowercase for BA data
    if "STATEFP" in cols and "COUNTYFP" in cols:
        gdf["FIPS"] = (gdf["STATEFP"] + gdf["COUNTYFP"]).astype("int")
    elif "statefp" in cols and "countyfp" in cols: 
        gdf["FIPS"] = (gdf["statefp"] + gdf["countyfp"]).astype("int")
    else:
        print("FIPS columns not recognized!")
    return gdf

def generate_plot(per_df, gdf, vmin = None, vmax = None, crop_usa = False):
    """Using a DataFrame mapping FIPS to infected percents
    and a GeoDataFrame with a FIPS column, generates plot
    """

    per_gdf = gdf.merge(per_df, left_on = "FIPS", right_on = "FIPS")

    fig, ax = plt.subplots(figsize=(10,10))
    ax.set(aspect='equal', xticks=[], yticks=[])
    if vmin == None and vmax == None:
        per_gdf.plot(column= 'per', ax = ax, cmap = "Purples", legend=True)
    else:
        per_gdf.plot(column= 'per', ax = ax, cmap = "Purples", legend=True, vmin=vmin, vmax=vmax)
    
    # If using tl_2020_us_county, crop to contiguous US states
    if crop_usa:
        # including Alaska + Hawaii:
        # ax.set_ylim((18, 72))
        # ax.set_xlim((-170, -66))
        ax.set_xlim((-125, -66))
        ax.set_ylim((24, 50))

    return fig

if __name__ == "__main__":
    yt.set_log_level(50)

    # CA: data/CA_2020_Census_Tracts/tl_2020_06_tract
    # BA: data/San_Francisco_Bay_Region_2020_Census_Tracts/region_2020_censustract
    # US: data/US_2020_Census_Tracts/tl_2020_us_county
    argc = len(sys.argv)
    data_dir = sys.argv[1] if len(sys.argv()) > 1 else "/global/cfs/projectdirs/m3623/test/output_usa/"
    data_names = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith("plt")])
    
    prefix = sys.argv[2] if len(sys.argv()) > 2 else "../../data/tl_2020_us_county"
    gdf = get_gdf(prefix)

    output_dir = sys.argv[3] if len(sys.argv()) > 3 else "./frames_usa/"
    for i in range(len(data_names)):
        # vmin and vmax are endpoints for color range; 16 > log(population of LA) is a safe upper bound
        # for per-capita, endpoints should be set to much less
        fig = generate_plot(get_raw_data(data_names[i]), gdf, vmin=0, vmax=16, crop_usa = True)
        fig.savefig(output_dir + "frame{:05d}".format(i))
        plt.close(fig)
