import yt
from yt.frontends.boxlib.data_structures import AMReXDataset
import pylab as plt

fn = "plt00168"

ds = AMReXDataset(fn)

ad = ds.all_data()
print(ad["total"].sum())
print(ad["infected"].sum())
print(ad["immune"].sum())
print(ad["previously_infected"].sum())

plt.pcolormesh(ad["total"].reshape(3000, 3000))
plt.savefig("test")
