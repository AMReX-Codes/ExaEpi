import numpy as np
import yt
from yt.frontends import boxlib
from yt.frontends.boxlib.data_structures import AMReXDataset
from collections import defaultdict
import sys

yt.set_log_level(50)

fn = sys.argv[1]

ds = AMReXDataset(fn)
ad = ds.all_data()

inf = ad['infected'].d
tot = ad['total'].d
fips = ad['FIPS'].d
tract = ad['Tract'].d

d_inf = defaultdict(int)
for i, to, f, tr in zip(inf, tot, fips, tract):
    code = 1000000*int(f) + int(tr)
    if code < 0:
        continue
    d_inf[code] += i

for k, v in d_inf.items():
    print("{:011d}".format(k), ",", v)
