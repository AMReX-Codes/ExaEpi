import numpy as np
import pandas as pd
from hdf5_process import get_home_and_work_counties

def get_cross_table(data_dir: str):
    """Returns a pandas DataFrame with the agent time matrix,
    representing the fraction of time the population from a home county
    spends in each other county, in aggregate"""

    # Gets a list of home/work counties for each particle as a 2xN array
    # (home as first row, work as second row)
    # and a list of all counties.
    particles, counties = get_home_and_work_counties(data_dir)

    # Create a table of counts of particles from a given home who work
    # at a given county 
    table = pd.pivot_table(pd.DataFrame(particles.T, columns=['home', 'outside county']),
                           index = 'outside county', columns = 'home', aggfunc = len, fill_value=0)

    # Normalize the table, then average with identity matrix to represent
    # agents spending half their time at home.
    return (table / table.sum() + np.eye(table.shape[0])) / 2

def main():
    table = get_cross_table("/pscratch/sd/a/arnav/ganning/sim/180_run")
    print(table)

if __name__ == "__main__":
    main()
