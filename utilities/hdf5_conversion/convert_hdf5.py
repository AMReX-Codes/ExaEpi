import numpy as np

import yt
from yt.frontends import boxlib
from yt.frontends.boxlib.data_structures import AMReXDataset
yt.set_log_level(50)

import h5py
from mpi4py import MPI

import sys
import os

time_varying_fields = {
    "particle_disease_counter": "float16", # countdown for particle immunity
    "particle_infection_prob": "float32", # probability of infection at EOD
    "particle_position_x": "float16", # box position x
    "particle_position_y": "float16", # box position y
    "particle_status": "int8", # status: 0 = uninfected, 1 = infected, 2 = immune, 3 = susceptible, 4 = dead
    "particle_symptomatic": "int8", # symptomatic status: 0 = will be, 1 = is, 2 = will not be until recovery
    "particle_treatment_timer": "uint16", # 
    "particle_withdrawn": "int8", # withdrawn or not
    }

constant_fields = {
    "particle_cpu": "uint8", # what CPU each particle is at
    "particle_id": "uint32", # particle ID
    "particle_age_group": "uint8", # age group of particle
    "particle_family": "uint32", # family ID
    "particle_home_i": "uint16", # home index one
    "particle_home_j": "uint16", # home index two
    "particle_incubation_period": "float32", # incubation period 
    "particle_infectious_period": "float32", # infectious period
    "particle_nborhood": "uint8", # particle neighborhood
    "particle_school": "int8", # school: -1 for none, 5 for some early school (preschool?)
    "particle_strain": "uint8", # strain number
    "particle_symptomdev_period": "float32", # symptom development period
    "particle_work_i": "uint16", # work index one
    "particle_work_j": "uint16", # work index two
    "particle_work_nborhood": "uint8", # work neighborhood
    "particle_workgroup": "uint16", # work group
    }

argc = len(sys.argv)
data_dir = sys.argv[1] if argc > 1 else "/global/cfs/projectdirs/m3623/test/bay/"
output_path = sys.argv[2] if argc > 2 else "data.hdf5"

data_names = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith("plt")])
ds = AMReXDataset(data_names[0])
time = len(data_names)
agent_num = ds.particle_type_counts["agents"]
ds.close()

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

f = h5py.File(output_path, "a", libver="latest", driver="mpio", comm=comm)

for field, dtype in time_varying_fields.items():
    f.create_dataset(field, (time, agent_num), dtype=dtype)
for field, dtype in constant_fields.items():
    f.create_dataset(field, (agent_num,), dtype=dtype)
block_size = np.ceil(time / size).astype(int)

for i in range(block_size):
    dataset_idx = rank * block_size + i
    if dataset_idx >= time:
        break
    ds = AMReXDataset(data_names[dataset_idx])
    ad = ds.all_data()
    for field, dtype in time_varying_fields.items():
        f[field][dataset_idx] = ad[field].astype(dtype)
    if dataset_idx == 0:
        for field, dtype in constant_fields.items():
            f[field][()] = ad[field].astype(dtype)
    ds.close()
f.close()
