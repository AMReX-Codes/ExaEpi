import numpy as np
import yt
from yt.frontends.boxlib.data_structures import AMReXDataset
from collections import defaultdict

ds = AMReXDataset("plt00000")
ad = ds.all_data()

inf = ad['infected'].d
tot = ad['total'].d
fips = ad['FIPS'].d
tract = ad['Tract'].d

d_inf = defaultdict(int)
d_tot = defaultdict(int)
for i, to, f, tr in zip(inf, tot, fips, tract):
    code = 1000000*int(f) + int(tr)
    if code < 0:
        continue
    d_inf[code] += i
    d_tot[code] += to

for k, v in d_inf.items():
    print(k, ",", v / d_tot[k])
