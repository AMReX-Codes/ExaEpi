#!/usr/bin/env python

"""Compare an ExaEpi run against an Epicast run over a sequence of days, at the level of individual
communities (see plot_geo.py / plot_geo_epicast.py for the single-choropleth-per-day view this
builds on). Two outputs are produced:
  - a day-scalar summary plot with three series:
      * Spearman's rho: rank correlation of per-community infection rate. Captures whether the two
        simulators agree on WHICH communities are hit hardest -- the relative ranking -- regardless
        of magnitude. This makes it sensitive to a failure mode where two communities have very
        similar infection rates but happen to swap relative order between the two simulators --
        rho treats that the same as a large, meaningful reordering.
      * infection-weighted Pearson's r: correlation of the raw infection rates, weighted by how
        much infection is actually present in each community (see below). Magnitude-sensitive, but
        as a correlation it's still blind to a systematic offset (e.g. one simulator running
        uniformly 10% hotter everywhere).
      * infection-weighted RMSE: root-mean-square of (rate_exaepi - rate_epicast) per community,
        the same weighting. This is the one direct answer to "how far apart are the two simulators'
        numbers, county by county" -- not a correlation at all, so it isn't fooled by rank swaps
        between similar values and isn't blind to a systematic offset. Plotted on a secondary axis
        since it's in rate units (0-1), not the -1..1 range of the two correlations.
    All three use the SAME weight per community: the average of the two simulators' infected counts
    there, rather than population. Population weighting would let a large county sitting at
    near-zero infection dominate the metric just because it has a lot of people, even though its
    rate difference there is essentially noise; weighting by how much infection is actually present
    focuses each metric on communities where a disagreement is actually meaningful.
  - a community-by-community scatter plot, one panel per day: ExaEpi infection rate vs Epicast
    infection rate, one point per matched community, with a y=x reference line -- the full
    per-community comparison underlying that day's scalars, for spotting which specific communities
    the two simulators disagree on.

Both simulators' loaders already return a per-community DataFrame keyed by GEOID10 (see
load_exaepi_grid_stats / reconstruct_epicast_snapshot), so the two are simply merged on GEOID10 per
day. Epicast's finest geographic unit is the Census tract, so tract level (the default here) is the
finest granularity at which the two can be compared; pass --county_level to compare at the coarser
county level instead.
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib

# This script only ever saves figures to a file, never displays them -- force the non-interactive
# Agg backend so rendering never touches an X server. Must happen before pyplot is imported.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_geo import load_exaepi_grid_stats, _parse_day_from_plot_dir  # noqa: E402
from plot_geo_epicast import reconstruct_epicast_snapshot  # noqa: E402
from read_epicast_events import read_events_bin  # noqa: E402


def _is_plotfile_dir(path):
    """An ExaEpi/AMReX plotfile directory always contains a top-level 'Header' file -- use that,
    rather than just the directory name, to tell an individual plotfile apart from a parent
    directory that merely holds several of them.
    """
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "Header"))


def expand_plot_dirs(paths):
    """Expand each of `paths` into the individual ExaEpi plotfile directories it refers to, so
    callers can point at a whole run's worth of output without listing every plt* directory by
    hand. Each entry in `paths` may be: a single plotfile directory (e.g. plt00050), a parent
    directory containing many plotfile subdirectories (e.g. a run's output directory), or a glob
    pattern (e.g. "results/plt*"). Returns the resulting directories deduplicated and sorted by the
    day parsed from their name.
    """
    expanded = []
    for path in paths:
        path = path.rstrip("/")
        if _is_plotfile_dir(path):
            expanded.append(path)
        elif os.path.isdir(path):
            children = sorted(
                os.path.join(path, name) for name in os.listdir(path) if _is_plotfile_dir(os.path.join(path, name))
            )
            if not children:
                raise SystemExit(f"No plotfile subdirectories (containing a Header file) found under {path}")
            expanded.extend(children)
        else:
            matches = sorted(p for p in glob.glob(path) if _is_plotfile_dir(p))
            if not matches:
                raise SystemExit(f"No plotfile directories matched: {path}")
            expanded.extend(matches)

    seen = set()
    unique = [d for d in expanded if not (d in seen or seen.add(d))]
    unique.sort(key=_parse_day_from_plot_dir)
    return unique


def weighted_pearsonr(x, y, w):
    """Weighted Pearson correlation coefficient between x and y, weighted by w. Unlike plain
    Pearson r, a community's contribution to the correlation scales with its weight -- so a handful
    of low-weight communities disagreeing doesn't move r as much as a handful of high-weight ones
    would.
    """
    x, y, w = np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(w, dtype=float)
    wsum = w.sum()
    xbar = (w * x).sum() / wsum
    ybar = (w * y).sum() / wsum
    cov_xy = (w * (x - xbar) * (y - ybar)).sum()
    var_x = (w * (x - xbar) ** 2).sum()
    var_y = (w * (y - ybar) ** 2).sum()
    return cov_xy / np.sqrt(var_x * var_y)


def weighted_rmse(x, y, w):
    """Weighted root-mean-square of (x - y), weighted by w -- a direct magnitude-of-disagreement
    metric (not a correlation), so it isn't fooled by two similar values swapping relative order
    and isn't blind to a systematic offset between x and y the way a correlation coefficient is.
    """
    x, y, w = np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(w, dtype=float)
    return np.sqrt((w * (x - y) ** 2).sum() / w.sum())


def compare_day(exaepi_df, epicast_df):
    """Merge one day's ExaEpi and Epicast per-community DataFrames on GEOID10 and return
    (rho, pval, r, rmse, n, merged_df): the Spearman rank correlation, infection-weighted Pearson
    correlation, and infection-weighted RMSE of infection rate (infected / pop) between the two.
    Rate rather than raw infected count is compared so that communities of very different
    population size are compared on a like-for-like basis. r and rmse are weighted by each
    community's average infected count (across the two simulators) rather than its population, so
    that a community currently at or near zero infection doesn't get outsized influence just
    because it has a large population -- a rate difference there is mostly noise, whereas the same
    difference in a heavily-infected community reflects a real, larger-magnitude disagreement.
    merged_df carries a rate_exaepi/rate_epicast column per matched community (GEOID10) -- the
    community-by-community comparison underlying the summary scalars, for callers that want to look
    beyond them. Returns (None, None, None, None, n, merged_df) if fewer than two communities match,
    since none of these are meaningful below that.
    """
    df = pd.merge(exaepi_df, epicast_df, on="GEOID10", suffixes=("_exaepi", "_epicast"))
    df = df[(df.pop_exaepi > 0) & (df.pop_epicast > 0)].copy()
    df["rate_exaepi"] = df.infected_exaepi / df.pop_exaepi
    df["rate_epicast"] = df.infected_epicast / df.pop_epicast
    if len(df) < 2:
        return None, None, None, None, len(df), df
    rho, pval = spearmanr(df.rate_exaepi, df.rate_epicast)
    weight = (df.infected_exaepi + df.infected_epicast) / 2.0
    r = weighted_pearsonr(df.rate_exaepi, df.rate_epicast, weight)
    rmse = weighted_rmse(df.rate_exaepi, df.rate_epicast, weight)
    return rho, pval, r, rmse, len(df), df


def main():
    parser = argparse.ArgumentParser(
        description="Compare ExaEpi and Epicast per-community infection rates across days: a "
        "Spearman-rho-vs-day summary plot, plus a community-by-community scatter plot per day"
    )
    parser.add_argument(
        "--plot_dirs",
        "-p",
        required=True,
        nargs="+",
        help="ExaEpi plotfile directories to compare, one per day. Each entry may be a single "
        "plotfile directory (e.g. plt00050), a parent directory containing many plotfile "
        "subdirectories (e.g. a run's whole output directory), or a glob pattern (e.g. "
        "'results/plt*') -- so a whole run's output can be pointed at directly instead of listing "
        "every plotfile by hand. The day for each is parsed from its trailing digits.",
    )
    parser.add_argument(
        "--events_file",
        "-f",
        required=True,
        help="Epicast run.events.bin file to compare against",
    )
    parser.add_argument(
        "--county_level",
        action="store_true",
        default=False,
        help="Compare at the Census county level (5-digit GEOID) instead of the default Census "
        "tract level (11-digit GEOID) -- Epicast's native granularity, and the finest level at "
        "which the two simulators can be compared.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="geo_compare.pdf",
        help="Output file name for the rho-vs-day plot",
    )
    parser.add_argument(
        "--epicast_day_offset",
        type=int,
        default=0,
        help="Shift the Epicast day used for comparison by this many days relative to the ExaEpi "
        "day (e.g. 20 compares ExaEpi day D against Epicast day D+20). Diagnostic option for "
        "checking that the comparison metrics are actually sensitive to a temporal misalignment "
        "between the two runs, rather than e.g. being dominated by shared population geography.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional file to also write the day,rho,pval,pearson_r,rmse,rmse_people,n table to as CSV",
    )
    parser.add_argument(
        "--scatter_output",
        "-s",
        default=None,
        help="Output file name for the community-by-community scatter plot (one panel per day: "
        "ExaEpi infection rate vs Epicast infection rate, one point per matched community). If "
        "omitted, this plot is skipped and only the rho-vs-day plot is produced.",
    )
    args = parser.parse_args()

    print("Reading Epicast data from", args.events_file)
    events_df, demog_df = read_events_bin(args.events_file)
    print(f"Read {len(events_df):,} events, {len(demog_df)} Census tracts")

    plot_dirs = expand_plot_dirs(args.plot_dirs)
    print(f"Comparing {len(plot_dirs)} days:", ", ".join(os.path.basename(d.rstrip("/")) for d in plot_dirs))

    rows = []
    scatter_panels = []
    for plot_dir in plot_dirs:
        day = _parse_day_from_plot_dir(plot_dir)
        if day is None:
            raise SystemExit(f"Could not parse a day number from plot directory name: {plot_dir}")

        exaepi_df = load_exaepi_grid_stats(
            plot_dir, tract_level=not args.county_level, county_level=args.county_level
        )
        epicast_day = day + args.epicast_day_offset
        epicast_df, resolved_day = reconstruct_epicast_snapshot(
            events_df, demog_df, day=epicast_day, county_level=args.county_level
        )
        if resolved_day != epicast_day:
            print(
                f"WARNING: {plot_dir} requested Epicast day {epicast_day} (ExaEpi day {day} + offset "
                f"{args.epicast_day_offset}), but Epicast clamped it to day {resolved_day}"
            )

        rho, pval, r, rmse, n, merged_df = compare_day(exaepi_df, epicast_df)
        if rho is None:
            print(f"WARNING: day {day}: fewer than 2 matching communities ({n}), skipping")
            continue
        # Purely a display convenience: express the (unitless) RMSE of infection RATE as a
        # people-equivalent, by scaling it up by the total matched population -- "if this RMSE
        # applied uniformly across everyone being compared, that's about how many people it'd be."
        # The RMSE itself is still computed on rates, weighted by infection level (see compare_day)
        # -- this is just a more human-readable way to report that same number, not a different
        # metric.
        total_pop = ((merged_df.pop_exaepi + merged_df.pop_epicast) / 2.0).sum()
        rmse_people = rmse * total_pop
        print(
            f"Day {day}: Spearman rho = {rho:.4f}, weighted Pearson r = {r:.4f}, "
            f"weighted RMSE = {rmse:.4f} (~{rmse_people:,.0f} people) (p = {pval:.3g}, n = {n})"
        )
        rows.append(
            {"day": day, "rho": rho, "pval": pval, "pearson_r": r, "rmse": rmse, "rmse_people": rmse_people, "n": n}
        )
        scatter_panels.append(
            (merged_df, f"Day {day}\nρ = {rho:.2f}, r = {r:.2f}, RMSE = {rmse:.3f} (~{rmse_people:,.0f} people)")
        )

    if not rows:
        raise SystemExit("No day had enough matching communities to compute a correlation")

    result_df = pd.DataFrame(rows).sort_values("day")
    if args.csv:
        result_df.to_csv(args.csv, index=False)
        print("Wrote table to", args.csv)

    fig, ax = plt.subplots(figsize=(16, 10))
    l1 = ax.plot(result_df.day, result_df.rho, label="Spearman's rho (rank)", lw=2)
    l2 = ax.plot(result_df.day, result_df.pearson_r, label="Pearson's r (infection-weighted, magnitude)", lw=2)
    ax.set_xlabel("Day", fontsize=20)
    ax.set_ylabel("Correlation (ExaEpi vs Epicast infection rate)", fontsize=20)
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.tick_params(axis="both", labelsize=16)

    ax_rmse = ax.twinx()
    l3 = ax_rmse.plot(
        result_df.day,
        result_df.rmse,
        color="tab:green",
        ls="--",
        lw=2,
        label="RMSE (infection-weighted, right axis)",
    )
    ax_rmse.set_ylabel("RMSE of infection rate (ExaEpi vs Epicast)", fontsize=20)
    ax_rmse.set_ylim(bottom=0)
    ax_rmse.tick_params(axis="y", labelsize=16)

    lines = l1 + l2 + l3
    ax.legend(lines, [line.get_label() for line in lines], fontsize=16)
    ax.set_title("Spatial agreement between ExaEpi and Epicast", fontsize=24)
    fig.tight_layout()
    print("Plotting results to", args.output)
    fig.savefig(args.output, bbox_inches="tight")

    if not args.scatter_output:
        return

    # Community-by-community scatter: one panel per day, ExaEpi rate vs Epicast rate, one point per
    # matched community. All panels share the same axis range so panels are visually comparable
    # across days, and a y=x reference line marks perfect agreement.
    axis_max = max(max(df.rate_exaepi.max(), df.rate_epicast.max()) for df, _ in scatter_panels)
    axis_max = axis_max * 1.05 if axis_max > 0 else 1.0

    n = len(scatter_panels)
    panel_width = 5.0
    fig2, axes2 = plt.subplots(1, n, figsize=(panel_width * n, panel_width), squeeze=False)
    axes2 = axes2[0]
    for ax, (df, label) in zip(axes2, scatter_panels):
        ax.plot([0, axis_max], [0, axis_max], color="gray", lw=1, ls="--", zorder=1)
        ax.scatter(df.rate_exaepi, df.rate_epicast, s=12, alpha=0.5, zorder=2)
        ax.set_xlim(0, axis_max)
        ax.set_ylim(0, axis_max)
        ax.set_aspect("equal")
        ax.set_xlabel("ExaEpi infection rate")
        ax.set_ylabel("Epicast infection rate")
        ax.set_title(label)
    fig2.tight_layout()
    print("Plotting community-by-community scatter to", args.scatter_output)
    fig2.savefig(args.scatter_output, bbox_inches="tight")


if __name__ == "__main__":
    main()
