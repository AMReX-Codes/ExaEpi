import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import colors
from matplotlib import cm
import os
import fiona
import geopandas as geopd
import pysal as ps
import shapefile
import pandas as pd
from shapely.geometry import shape
import numpy as np
import os
import sys
import re

argList= sys.argv[1:]
if len(argList)>= 1: infCasesDir= argList[0]+"/"
if len(argList)>= 2: CensusDir  = argList[1]+"/"
if len(argList)>= 3: inputMetric  = argList[2]
if len(argList)>= 3: outputMetric  = argList[3]

gdf = geopd.GeoDataFrame(columns=["STATEFP00", "COUNTYFP00", "TRACTCE00", "CTIDFP00", "NAME00", "NAMELSAD00", "MTFCC00",
                                  "FUNCSTAT00", "ALAND00", "AWATER00", "INTPTLAT00", "INTPTLON00", "geometry"])

infCasesFiles= [os.path.join(infCasesDir, f) for f in os.listdir(infCasesDir) if f.endswith(".csv")]

for file in os.listdir(CensusDir):
    filename = CensusDir + file + "/" + file
    myshp = open(filename+".shp", "rb")
    mydbf = open(filename+".dbf", "rb")
    myprj = open(filename+".prj", "rb")
    r = shapefile.Reader(shp=myshp, dbf=mydbf, prj=myprj)
    attributes, geometry = [], []
    field_names = [field[0] for field in r.fields[1:]]
    for row in r.shapeRecords():
            geometry.append(shape(row.shape.__geo_interface__))
            attributes.append(dict(zip(field_names, row.record)))
    df = geopd.GeoDataFrame(data = attributes, geometry = geometry)
    gdf= gdf._append(df)
    r.close()

gdf['fips'] = gdf["CTIDFP00"].astype(int)
gdf1= gdf.copy()
gdf1['per'] = 0.0
gdf1.set_index('fips',inplace=True)

#use cmap with a norm to convert a number to RGB
#two slope norm
#set center to 0, adjust the negative vmin to control how fast color changes from 0 to vmax
vmin = -10.0
vmax = 6.0
norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.000, vmax=vmax)
cmap = cm.cool

sorted_infCasesFiles= sorted(infCasesFiles, key=lambda x: int( re.findall('\d+', x)[0]))
for caseFile in sorted_infCasesFiles:
    idx = int(re.findall('\d+', caseFile)[0])
    print("processing file", caseFile)
    df000 = pd.read_csv(caseFile, header=None, names=["fips", "per"], dtype={'fips':int, 'per':float})
    df000.set_index('fips',inplace=True)
    gdf_join = pd.merge(gdf,df000, left_on='fips', right_on='fips', how='inner')
    gdf_join.set_index('fips',inplace=True)
    gdf1= pd.concat([gdf1,gdf_join])
    fig, ax = plt.subplots(figsize=(20,10))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, ticks=np.linspace(0, 6, 7), label="Log10 Cumulative Cases", orientation="vertical")
    ax.xaxis.set_ticks_position('top')
    #ax.set_xlim(-150, -100)
    ax.set(aspect='equal', xticks=[], yticks=[])
    gdf1.plot(column= 'per', ax = ax, norm=norm, cmap=cmap)#, legend=False)
    output= "ca"+str(idx+1).zfill(4)+".png"
    plt.savefig(output)
