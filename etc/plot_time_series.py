#!/usr/bin/env python

import yt
from yt.frontends.boxlib.api import AMReXDataset
from typing import IO
#import pylab as plt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import argparse
import glob

class CaseCounts:
    _total = None
    _never_infected = None
    _infected = None
    _immune = None
    _previously_infected = None

    def value(self,input):
        if input:
            output = input
        else:
            output = 0
        return output

    def total(self,num_total=None):
        if num_total:
            self._total = num_total
        else:
            return self.value(self._total)

    def never_infected(self,num=None):
        if num:
            self._never_infected = num
        else:
            return self.value(self._never_infected)

    def infected(self,num_infected=None):
        if num_infected:
            self._infected = num_infected
        else:
            return self.value(self._infected)

    def immune(self,num_immune=None):
        if num_immune:
            self._immune = num_immune
        else:
            return self.value(self._immune)

    def previously_infected(self,num=None):
        if num:
            self._previously_infected = num
        else:
            return self.value(self._previously_infected)

def get_args() -> argparse.ArgumentParser:
    '''
    Get command line arguments

    We need at least two command line arguments:
    - One specifying the data sets that make up the time series
    - One specifying the time series output name
    For the output we might generate both a CSV file with the data
    as well as a PNG file with a plot of the time series.
    '''
    describe = '''
This tool extracts the data of a collection of time points and 
computes overall case numbers for each of them. The totals are
stored in a CSV file for further processing. They are also plotted
for a quick visual inspection of the results.'''
    parser = argparse.ArgumentParser(description=describe)
    parser.add_argument("directories",help="a name or regular expression for the data directories")
    parser.add_argument("outfile",help="basename for the output files")
    return parser.parse_args()

def get_directories(pattern: str) -> list:
    '''
    Takes a pattern that identifies a set of sub-directories,
    looks actual directory names up, stores them in a sorted
    list (just alphabetical sorting for now), and returns 
    the resulting list.
    '''
    dir_list = glob.glob(pattern)
    dir_list.sort()
    return dir_list

def get_timepoint(directory: str) -> CaseCounts:
    '''
    For a given directory extract the case counts
    and return them in a container object.
    '''
    cases = CaseCounts()
    ds = AMReXDataset(directory)
    ad = ds.all_data()
    cases.total(ad["total"].sum())
    cases.never_infected(ad["never_infected"].sum())
    cases.infected(ad["infected"].sum())
    cases.immune(ad["immune"].sum())
    cases.previously_infected(ad["previously_infected"].sum())
    return cases

def write_timepoint(fp: IO[str], cases: CaseCounts) -> None:
    '''
    Write the data of a time point to a CSV file.
    '''
    fp.write(f'{int(cases.total())}, ')
    fp.write(f'{int(cases.never_infected())}, ')
    fp.write(f'{int(cases.infected())}, ')
    fp.write(f'{int(cases.immune())}, ')
    fp.write(f'{int(cases.previously_infected())}, ')
    fp.write(f'\n')

def write_timeseries(dirlist: list, output: str) -> None:
    '''
    Create the CSV file, iterate over all directories extracting
    the case numbers, and write the data to the CSV file.
    Note we do not check the output file name as the plotting 
    function needs to be using the same name. So that checking needs
    to happen elsewhere.
    '''
    with open(output,'w') as fp:
        fp.write("\"total\",\"never infected\",\"infected\",\"immune\",\"previously infected\",\n")
        for dir in dirlist:
            cases = get_timepoint(dir)
            write_timepoint(fp,cases)

def get_csv_name(output: str) -> str:
    '''
    Given a string generate a CSV file name.
    '''
    if output[-4:] == ".csv":
        csv = output
    else:
        csv = output + ".csv"
    return csv

def get_png_name(output: str) -> str:
    '''
    Given a string generate a PNG file name.
    '''
    if output[-4:] == ".png":
        png = output
    else:
        png = output + ".png"
    return png

def plot_timeseries(csv_file: str, png_file: str) -> None:
    '''
    Use pandas to read CSV file and then plot it with
    MatPlotLib.
    '''
    fig = plt.figure(1)
    axx = fig.add_subplot()
    df = pd.read_csv(csv_file,index_col=False,usecols=[0,1,2,3,4])
    df.plot(ax=axx)
    fig.show()
    fig.savefig(png)
    if fig.waitforbuttonpress():
        pass

if __name__ == "__main__":
    args = get_args()
    dirs = get_directories(args.directories)
    csv = get_csv_name(args.outfile)
    write_timeseries(dirs,csv)
    png = get_png_name(args.outfile)
    plot_timeseries(csv,png)

