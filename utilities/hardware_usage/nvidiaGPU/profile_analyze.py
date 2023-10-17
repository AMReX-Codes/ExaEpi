import subprocess as sp
import numpy as np
import pandas as pd
from io import StringIO
import os
import re
import shutil
import sqlite3
from IPython.display import display
import matplotlib.pyplot as plt
from utils import *
import sys

nsightcompute = sp.run(["which", "nv-nsight-cu-cli"], stdout=sp.PIPE).stdout.decode('utf-8').strip()
if not "nv-nsight-cu-cli" in nsightcompute:
    print("Nsight Compute NOT FOUND")
    exit()

tracedir = sp.run(["ls", "-la"], stdout=sp.PIPE).stdout.decode('utf-8').strip()
if not "ncu_traces" in tracedir:
    sp.run(["mkdir", "ncu_traces"])
tracedir = "./ncu_traces"

homedir = os.path.dirname(os.getcwd())
outputdir = ["."]

def parse_time(df_metrics):
    selectkeys = ["ID", "Name"]
    resultkeys = ["ID", "Name"]
    metricdf   = df_metrics.copy()
    profiledf  = pd.DataFrame(columns=selectkeys)
    # get cycles
    metricname = "CUDA Cycles"
    cyclesdf   = metricdf.loc[(metricdf["Metric Name"]=="sm__cycles_elapsed") & (metricdf["Metric Type"]=="total"),
                           selectkeys+["Metric Unit", "Metric Value"]].reset_index(drop=True).sort_values(by=selectkeys).rename(columns={"Metric Value": metricname}).copy()
    # get rates
    metricname = "CUDA Rates"
    ratesdf = metricdf.loc[(metricdf["Metric Name"]=="sm__cycles_elapsed") & (metricdf["Metric Type"]=="rate"),
                           selectkeys+["Metric Unit", "Metric Value"]].reset_index(drop=True).sort_values(by=selectkeys).rename(columns={"Metric Value": metricname}).copy()
    # check consistency
    if not cyclesdf[['ID', 'Name']].equals(ratesdf[['ID', 'Name']]):
        raise ValueError("CUDA Time data not consistent")
    # adjust metric unit
    if(ratesdf.size >0 and cyclesdf.size >0):
        ratesdf["CUDA Rates"]= pd.to_numeric(ratesdf["CUDA Rates"].replace(',', '', regex=True))
        cyclesdf["CUDA Cycles"]= pd.to_numeric(cyclesdf["CUDA Cycles"].replace(',', '', regex=True))
        ratesdf.loc[ratesdf["Metric Unit"].str.contains("cycle/nsecond"), ["CUDA Rates"]] *= 1e9
        ratesdf.loc[ratesdf["Metric Unit"].str.contains("cycle/usecond"), ["CUDA Rates"]] *= 1e6
    # manual merge and compute CUDA Time
        cyclesdf["CUDA Rates"] = list(ratesdf["CUDA Rates"])
        cyclesdf["CUDA Time"] = cyclesdf["CUDA Cycles"] / cyclesdf["CUDA Rates"]
    # merge with output
        profiledf = cyclesdf[selectkeys+['CUDA Time']].copy()
    ### Combine
        profiledf['Invocations'] = 1
        profiledf = profiledf.groupby(resultkeys).sum().reset_index()
        profiledf.sort_values(by=resultkeys).reset_index(drop=True, inplace=True)
        display(profiledf)
    return profiledf

### Integer operations
def parse_intOps(df_metrics):
    selectkeys = ["ID", "Name"]
    resultkeys = ["ID", "Name"]
    metricdf   = df_metrics.copy()

    metricdf["Metric Value"]= pd.to_numeric(metricdf["Metric Value"].replace(',', '', regex=True))
    profiledf  = pd.DataFrame(columns=selectkeys)
    metrics = ['smsp__sass_thread_inst_executed_op_integer_pred_on']
    tmpdf = metricdf.loc[ metricdf["Metric Name"].isin(metrics), resultkeys+["Metric Value"] ].copy()
    tmpdf = tmpdf.groupby(resultkeys).sum().reset_index().rename(columns={"Metric Value": "INT OPs"})
    print(tmpdf)
    if (tmpdf.size >0):
        profiledf = tmpdf[resultkeys+["INT OPs"]]
        profiledf.sort_values(by=resultkeys).reset_index(drop=True, inplace=True)
        display(profiledf)
    del metricdf['ID']
    return profiledf

def parse_dram(df_metrics):
    selectkeys = ["ID", "Name"]
    resultkeys = ["ID", "Name"]
    metricdf   = df_metrics.copy()
    profiledf  = pd.DataFrame(columns=selectkeys)
    profiledf  = profiledf.fillna(0.)
    if (metricdf.size >0):
        metricdf.loc[metricdf["Metric Unit"].str.contains("Gbyte"), ["Metric Value"]] *= 1e9
        metricdf.loc[metricdf["Metric Unit"].str.contains("Mbyte"), ["Metric Value"]] *= 1e6
        metricdf.loc[metricdf["Metric Unit"].str.contains("Kbyte"), ["Metric Value"]] *= 1e3

    #project out
    dramdf = metricdf.loc[metricdf["Metric Name"].str.contains("dram__bytes"), resultkeys+["Metric Value"] ].copy()
    dramdf = dramdf.groupby(resultkeys).sum().reset_index().rename(columns={"Metric Value": "DRAM Bytes"})
    # merge
    if (dramdf.size >0):
        profiledf = dramdf[resultkeys+["DRAM Bytes"]]
        profiledf.sort_values(by=resultkeys).reset_index(drop=True, inplace=True)
    del metricdf['ID']
    return profiledf


#run the program and collect ncu traces

sp.run(["dcgmi", "profile", "--pause"])
sp.run(["ncu", "-o", "./ncu_traces/ncu.kernel_all.metric_cyclePerSec.avg.cycles.avg", "--metrics=sm__cycles_elapsed.avg.per_second,sm__cycles_elapsed.avg"]+sys.argv[1:])
sp.run(["ncu", "-o", "./ncu_traces/ncu.kernel_all.metric_dramread", "--metrics=dram__bytes_read.sum"]+sys.argv[1:])
sp.run(["ncu", "-o", "./ncu_traces/ncu.kernel_all.metric_dramwrite", "--metrics=dram__bytes_write.sum"]+sys.argv[1:])
sp.run(["ncu", "-o", "./ncu_traces/ncu.kernel_all.metric_flops", "--metrics=smsp__sass_thread_inst_executed_op_integer_pred_on.sum"]+sys.argv[1:])

#combination of markers and colors (8x3=24 for now)
color_list=  ['r', 'g', 'b']
marker_list= ['o', 'v', '*', 's', 'p', '*', 'h', 'd']
plt.figure()
allKernelName = ""
dirCnt=0

if True:
  #get all the files
  files = []
  files += [ os.path.join(tracedir,x) for x in os.listdir(tracedir) if ((os.path.splitext(x)[-1] == ".ncu-rep"))]

  #recs
  records = []
  #build feature list:
  for path in files:
    file = os.path.basename(path)
    #path
    path = os.path.dirname(path)
    #splitup
    splt = file.split(".")
    prefix = ".".join(splt[0:-1])
    #append to records
    records.append({"prefix": prefix, "file": os.path.join(path, file)})
#put in df
  recorddf = pd.DataFrame(records).sort_values(["prefix"])
  resultkeys          = ["ID", "Name"]
  profiledf_time      = pd.DataFrame(columns=resultkeys)
  profiledf_fp32      = pd.DataFrame(columns=resultkeys)
  profiled_allIntOps    = pd.DataFrame(columns=resultkeys)
  profiledf_DRAM      = pd.DataFrame(columns=resultkeys)
  profiled_allDRAM    = pd.DataFrame(columns=resultkeys)
  aggregatedKernelName   = ""

  for pref in recorddf["prefix"]:
    file = os.path.basename(path)
    #set empty lists
    df_times = []
    df_timeline = []
    df_summary = []
    df_metrics = []

    #project frame
    files = recorddf.loc[ recorddf["prefix"] == pref, "file" ].values
    #project the invididual files
    metricfile = [x for x in files if x.endswith(".ncu-rep")][0]
    #get the parameters from the filename
    parameters = parse_filename_nsight(metricfile)
    splt= pref.split(".")
    kernelName= splt[1]
            
    #metrics
    #open subprocess and communicate
    metricdf = import_nsight_metric(metricfile, ncu=nsightcompute)
    for key in parameters:
        metricdf[key] = parameters[key]

    #fuse read/write metrics together:
    unique_metrics = metricdf["Metric Name"].unique()
        
    unique_metrics = set([x.replace(".sum","").replace(".per_second","").replace(".avg","").replace("_write","").replace("_read","").replace("_ld","").replace("_st","") for x in unique_metrics])
    unique_metrics = set([x.replace(".sum","").replace(".per_second","").replace(".avg","") for x in unique_metrics])
    unique_units = metricdf["Metric Unit"].unique()
    #add the metric type
    metricdf["Metric Type"] = "total"
    #read
    metricdf.loc[ metricdf[ "Metric Name" ].str.contains("_read"), "Metric Type" ] = "read"
    metricdf.loc[ metricdf[ "Metric Name" ].str.contains("_ld"), "Metric Type" ] = "read"
    #write
    metricdf.loc[ metricdf[ "Metric Name" ].str.contains("_write"), "Metric Type" ] = "write"
    metricdf.loc[ metricdf[ "Metric Name" ].str.contains("_st"), "Metric Type" ] = "write"
    #rate
    metricdf.loc[ metricdf[ "Metric Name" ].str.contains(".per_second"), "Metric Type" ] = "rate"
                
    for metric in unique_metrics:
        metricdf.loc[ metricdf[ "Metric Name"].str.startswith(metric), "Metric Name" ] = metric
                
    #append to DF:
    df_metrics.append(metricdf)
    
    metricdf = pd.concat(df_metrics)
    
        #compute the profile
    parsedTime          = parse_time(metricdf)
    if parsedTime.size>0:
        profiledf_time      =  parsedTime
        aggregatedKernelName  = kernelName[len("kernel"):]

    parsedIntOps          =  parse_intOps(metricdf)
    if parsedIntOps.size >0 :
        profiledf_fp32       =  parsedIntOps
        aggregatedKernelName = kernelName[len("kernel"):]

    parsedDRAM          =  parse_dram(metricdf)
    if parsedDRAM.size >0 :
        #profiledf_DRAM  =  profiledf_DRAM.append(parsedDRAM)
        profiledf_DRAM  =  parsedDRAM
        aggregatedKernelName = kernelName[len("kernel"):]
  allKernelName += aggregatedKernelName
  profiled_allIntOps = profiledf_time
  profiled_allDRAM = profiledf_time
  profiled_allIntOps = profiled_allIntOps.merge(profiledf_fp32[resultkeys+["INT OPs"]], on=resultkeys, how="inner")
  profiled_allIntOps = profiled_allIntOps.merge(profiledf_DRAM[resultkeys+["DRAM Bytes"]], on=resultkeys, how="inner")

  profiled_allIntOps["INT/s"]   = profiled_allIntOps["INT OPs"] / profiled_allIntOps["CUDA Time"]
  profiled_allIntOps["INT/B"]   = profiled_allIntOps["INT OPs"] / profiled_allIntOps["DRAM Bytes"]

  AI_IntOps= profiled_allIntOps["INT/B"]
  INTRateCol=  profiled_allIntOps["INT/s"]

  #remove the underscore
  myColor=  color_list[dirCnt % len(color_list)]
  myMarker= marker_list[dirCnt // len(color_list)]
  lbl= aggregatedKernelName[1:]
  roofline= plt.scatter(AI_IntOps, INTRateCol, marker=myMarker, color=myColor, label=lbl)
  plt.legend(numpoints=1, loc='lower left', prop={'size': 8})
  dirCnt = dirCnt+1

x = np.arange(0,12.535,0.01)
x1 = np.arange(1,10000,0.1)
x2 = np.arange(0,1,0.001)
y1 = np.full(len(x1), 108*4*1.41*32*1e9)
plt.plot(x, 1555*x*1e9, color='black',linestyle='-',linewidth=2.0)  
plt.plot(x1, y1, color='black',linestyle='-',linewidth=2.0)  
plt.plot(x2, 19492*x2*1e9, color='green',linestyle='-',linewidth=2.0)  
plt.xlabel("Arithmetic Intensity (IntOps/Byte)")
plt.ylabel("Op Rate (IntOps/s)")
plt.yscale('log')
plt.xscale('log')
plt.show()
output= "Roofline.pdf"
plt.savefig(output)
