#!/usr/bin/env python

"""Plot ExaEpi and Epicast infection-spread choropleths for the same sequence of days, stacked in
two rows (Epicast on top, ExaEpi below) with one column per day -- so the visual comparison from
plot_geo.py/plot_geo_epicast.py and the day-scalar comparison from plot_geo_compare.py can be read
together: each day's column is labeled with that day's Spearman rho, infection-weighted Pearson r,
and infection-weighted RMSE (see plot_geo_compare.py for how these are computed and what they
mean), directly above the two choropleths they summarize.

Days are given explicitly as a list (the same --day style as plot_geo_epicast.py), since a single
day value has to resolve to both an Epicast snapshot (reconstructed from the events log) and a
matching ExaEpi plotfile directory (looked up by the day parsed from its name) -- unlike
plot_geo.py, which takes plotfile directories directly and has no independent notion of "day" to
share with Epicast.
"""

import os
import sys
import argparse
import pandas as pd
import geopandas as gp
import matplotlib

# This script only ever saves figures to a file, never displays them -- force the non-interactive
# Agg backend so rendering never touches an X server.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib as mp  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_geo import load_exaepi_grid_stats, _parse_day_from_plot_dir  # noqa: E402
from plot_geo_epicast import reconstruct_epicast_snapshot  # noqa: E402
from plot_geo_compare import expand_plot_dirs, compare_day  # noqa: E402
from read_epicast_events import read_events_bin  # noqa: E402


def main():
    plt.rcParams["xtick.labelsize"] = 32
    plt.rcParams["ytick.labelsize"] = 32
    plt.rcParams["font.size"] = 48

    parser = argparse.ArgumentParser(
        description="Plot ExaEpi and Epicast choropleths stacked in two rows (Epicast on top, "
        "ExaEpi below), one column per day, each column labeled with that day's rho/r/RMSE"
    )
    parser.add_argument(
        "--exaepi_dir",
        "-p",
        required=True,
        nargs="+",
        help="Where to find ExaEpi plotfiles -- same as plot_geo_compare.py's --plot_dirs: a "
        "parent directory containing many plotfile subdirectories, a single plotfile directory, "
        "or a glob pattern. Which specific day(s) get plotted is chosen by --day, not by which "
        "directories match here -- this just needs to cover them.",
    )
    parser.add_argument(
        "--events_file",
        "-f",
        required=True,
        help="Epicast run.events.bin file",
    )
    parser.add_argument(
        "--day",
        "-d",
        type=int,
        nargs="+",
        default=[None],
        help="One or more 0-based days to plot, one column each (default: the last day in the "
        "events file). Each day is used to both reconstruct the Epicast snapshot and look up the "
        "matching ExaEpi plotfile (by the day parsed from its directory name).",
    )
    parser.add_argument(
        "--shape_files",
        "-s",
        required=True,
        nargs="+",
        help="Census shape files (.shp) at the granularity being compared -- tract by default, "
        "or county if --county_level is passed.",
    )
    parser.add_argument(
        "--states_file",
        "-e",
        required=True,
        help="Shape file for US states",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="geo_dual.pdf",
        help="Output file name for plot",
    )
    parser.add_argument(
        "--coord_bounds",
        "-b",
        default=[-170, -66.6, 18.5, 71.5],
        nargs="+",
        help="Range for longitude/latitude: lon_min lon_max lat_min lat_max",
    )
    parser.add_argument(
        "--county_level",
        action="store_true",
        default=False,
        help="Compare/plot at the Census county level instead of the default Census tract level "
        "-- Epicast's native granularity, and the finest level at which the two can be compared.",
    )
    args = parser.parse_args()

    geo_unit = "county" if args.county_level else "tract"
    geo_unit_pl = "counties" if args.county_level else "tracts"

    print("Reading Epicast data from", args.events_file)
    events_df, demog_df = read_events_bin(args.events_file)
    print(f"Read {len(events_df):,} events, {len(demog_df)} Census tracts")

    exaepi_dirs = expand_plot_dirs(args.exaepi_dir)
    day_to_plotdir = {_parse_day_from_plot_dir(d): d for d in exaepi_dirs}
    print(f"Found {len(day_to_plotdir)} ExaEpi plotfile days:", sorted(day_to_plotdir))

    shp_dfs = []
    state_codes = []
    for fname in args.shape_files:
        if not fname.endswith(".shp"):
            print(
                "WARNING: file",
                fname,
                "passed with --shape_files does not appear to be a shapefile with .shp extension",
            )
            continue
        print("Reading data from", fname)
        shp_dfs.append(gp.read_file(fname))
        state_code = os.path.basename(fname).split("_")[2]
        state_codes.append(state_code)

    shp_data = pd.concat(shp_dfs)
    shp_data.GEOID10 = shp_data.GEOID10.astype("int64")
    print("Read in", len(shp_data), f"Census {geo_unit_pl}")

    states = gp.read_file(args.states_file)
    states = states[states.STATE.isin(state_codes)]
    max_count = 30000

    example = {"county": "tl_2010_35_county10.shp", "tract": "tl_2010_35_tract10.shp"}[geo_unit]

    # For each requested day, reconstruct both simulators' snapshots, compute the same rho/r/RMSE
    # comparison plot_geo_compare.py reports, and merge each onto the shapefile geometry for
    # plotting. panels holds one (exaepi_geo_df, epicast_geo_df, label) tuple per day/column.
    panels = []
    for day in args.day:
        epicast_df, resolved_day = reconstruct_epicast_snapshot(
            events_df, demog_df, day=day, county_level=args.county_level
        )
        if day is not None and resolved_day != day:
            print(f"WARNING: requested day {day}, but Epicast clamped it to day {resolved_day}")

        if resolved_day not in day_to_plotdir:
            available = ", ".join(str(d) for d in sorted(day_to_plotdir))
            raise SystemExit(
                f"No ExaEpi plotfile found for day {resolved_day} among --exaepi_dir. Available "
                f"days: {available}"
            )
        plot_dir = day_to_plotdir[resolved_day]
        exaepi_df = load_exaepi_grid_stats(
            plot_dir, tract_level=not args.county_level, county_level=args.county_level
        )

        rho, pval, r, rmse, n, _ = compare_day(exaepi_df, epicast_df)
        if rho is None:
            stats_str = f"(only {n} matched {geo_unit_pl})"
        else:
            stats_str = f"ρ={rho:.2f}, r={r:.2f}\nRMSE={rmse:.3f}"
        label = (f"Day {resolved_day}", stats_str)

        exaepi_geo_df = pd.merge(shp_data, exaepi_df, on=["GEOID10"], how="inner")
        epicast_geo_df = pd.merge(shp_data, epicast_df, on=["GEOID10"], how="inner")
        if exaepi_geo_df.empty or epicast_geo_df.empty:
            raise SystemExit(
                f"No rows matched after merging day {resolved_day}: check --shape_files is a "
                f"Census {geo_unit.upper()} shapefile (e.g. {example}) covering the same state as "
                f"the ExaEpi/Epicast data."
            )
        panels.append((exaepi_geo_df, epicast_geo_df, label))

    # Bounds are the union across every panel's data (both rows), so the whole grid shares one
    # consistent geographic extent instead of each panel framing itself differently.
    all_lon = pd.concat([df.INTPTLON10.astype("float") for pair in panels for df in pair[:2]])
    all_lat = pd.concat([df.INTPTLAT10.astype("float") for pair in panels for df in pair[:2]])
    xmin = max(float(args.coord_bounds[0]), float(all_lon.min()) - 0.5)
    xmax = min(float(args.coord_bounds[1]), float(all_lon.max()) + 0.5)
    xrange = xmax - xmin
    ymin = max(float(args.coord_bounds[2]), float(all_lat.min()) - 0.5)
    ymax = min(float(args.coord_bounds[3]), float(all_lat.max()) + 0.5)
    yrange = ymax - ymin

    n = len(panels)
    panel_width = 12.0
    fig_x = panel_width * n
    fig_y = 2 * panel_width * yrange / xrange
    print(f"Plot dimensions: lng/lat {xmin}, {xmax}, {ymin}, {ymax}, figure size: {fig_x}, {fig_y}")

    fig, axes = plt.subplots(2, n, figsize=(fig_x, fig_y), squeeze=False)

    norm = mp.colors.LogNorm(vmin=1.0, vmax=max_count)
    for j, (exaepi_geo_df, epicast_geo_df, label) in enumerate(panels):
        for row, (geo_df, row_name) in enumerate([(epicast_geo_df, "Epicast"), (exaepi_geo_df, "ExaEpi")]):
            ax = axes[row][j]
            states.boundary.plot(ax=ax, lw=1, color="black")
            geo_df.plot(ax=ax, column="infected", cmap="OrRd", legend=False, norm=norm)  # type: ignore
            ax.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)
            ax.set_frame_on(False)
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            if j == 0:
                ax.set_ylabel(row_name, fontsize=64)
        day_str, stats_str = label
        axes[0][j].set_title(day_str, fontsize=64, pad=140)
        axes[0][j].text(
            0.5, 1.03, stats_str, transform=axes[0][j].transAxes, ha="center", va="bottom", fontsize=44, linespacing=1.4
        )

    # A single colorbar spanning both rows, rather than one per panel -- built from an explicit
    # ScalarMappable (since legend=False above) and handed every axes in the grid so matplotlib
    # sizes/positions it to span the full two-row height instead of attaching to just one panel.
    sm = mp.cm.ScalarMappable(norm=norm, cmap="OrRd")
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.ax.tick_params(labelsize=40)

    print("Plotting results to", args.output)
    plt.savefig(args.output, bbox_inches="tight")


if __name__ == "__main__":
    main()
