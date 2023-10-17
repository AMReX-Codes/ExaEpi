import subprocess as sp
import numpy as np
import pandas as pd
from io import StringIO
import os
import re
import shutil
import sqlite3

def parse_filename_nsight(filename):
    result={}
    result["Kernel"] = str(re.match(r'.*\.kernel_(.*?)\.',filename).groups()[0])
    return result

def import_nsight_metric(filename, ncu):
    #execute nvprof and parse file
    args = [ncu,"--csv","-i",filename]
    #skiprows = 2

    #open subprocess and communicate
    p = sp.Popen(args, stdout=sp.PIPE, stderr=sp.PIPE)
    stdout, stderr = p.communicate()

    #get timeline from csv
    profiledf = pd.read_csv(StringIO(stdout.decode("utf-8")),skiprows=0) #.dropna(how="all").rename(columns={"Kernel": "Name"})

    #clean up
    del profiledf["Process ID"]
    del profiledf["Process Name"]
    del profiledf["Host Name"]
    del profiledf["Kernel Time"]
    del profiledf["Context"]
    #del profiledf["Stream"]
    del profiledf["Section Name"]

    profiledf.rename(columns={"Kernel Name": "Name"}, inplace=True)

    #return result
    return profiledf

