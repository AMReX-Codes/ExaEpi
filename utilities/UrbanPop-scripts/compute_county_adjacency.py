#!/usr/bin/env python3
# Compute nationwide county-adjacency pairs from a county boundary shapefile.
#
# Used by upop_to_exaepi.py to bound the final school-allocation fallback tier (currently
# university students) to a student's home county plus its immediate geographic neighbors,
# instead of the entire state.

import argparse
import os

os.environ.setdefault("SHAPE_RESTORE_SHX", "YES")

import geopandas as gpd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--counties_shapefile",
        default="data/US_2000_Counties/tl_2000_us_county.shp",
        help="County boundary shapefile with a 5-digit state+county FIPS 'geoid' column",
    )
    parser.add_argument(
        "--output", default="data/UrbanPop/county_adjacency.csv", help="Output CSV file"
    )
    args = parser.parse_args()

    counties = gpd.read_file(args.counties_shapefile)
    # This shapefile's polygons have small topology gaps between nominally-adjacent counties,
    # so an exact "touches" test misses most real neighbors (avg ~1.6/county vs. the ~5.8/county
    # the Census county adjacency file reports). A small buffer closes those gaps.
    buffered = counties.geometry.buffer(0.001)
    sindex = buffered.sindex
    geoids = counties["geoid"].tolist()
    num_pairs = 0
    with open(args.output, "w") as f:
        f.write("geoid,neighbor_geoid\n")
        for i, geom in enumerate(buffered):
            for j in sindex.query(geom, predicate="intersects"):
                if j != i:
                    f.write(f"{geoids[i]},{geoids[j]}\n")
                    num_pairs += 1
    print(f"Wrote {num_pairs} adjacency pairs for {len(counties)} counties to {args.output}")


if __name__ == "__main__":
    main()
