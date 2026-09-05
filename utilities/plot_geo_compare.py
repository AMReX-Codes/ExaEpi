#!/usr/bin/env python

"""Compare an ExaEpi run against an Epicast run over a sequence of days, at the level of individual
communities (see plot_geo.py for the single-choropleth-per-day view this builds on, and for the
loaders and compare_day() this reuses). Two outputs are produced:
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
import argparse
import pandas as pd
import matplotlib

# This script only ever saves figures to a file, never displays them -- force the non-interactive
# Agg backend so rendering never touches an X server. Must happen before pyplot is imported.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_geo import (  # noqa: E402
    load_exaepi_grid_stats,
    _parse_day_from_plot_dir,
    reconstruct_epicast_snapshot,
    expand_plot_dirs,
    compare_day,
)
from read_epicast_events import read_events_bin  # noqa: E402
from plos_compbio_style import apply_style, HALF_PAGE_WIDTH_IN, HALF_PAGE_HEIGHT_IN  # noqa: E402


def main():
    apply_style()

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
    parser.add_argument(
        "--show_rmse",
        action="store_true",
        default=False,
        help="Also plot the infection-weighted RMSE as a third series (on a secondary axis, since "
        "it's in rate units rather than the -1..1 range of the two correlations). Off by default; "
        "the RMSE is still computed and printed/written to --csv either way.",
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

    # Labels are kept short (the "infection-weighted"/"rank"/"magnitude" detail belongs in the
    # figure caption, not the plot itself) since this whole figure is only ~3.1in wide in the
    # paper -- a long label/legend string simply has no room to fit at PLOS's 8-12pt font floor,
    # regardless of layout engine.
    fig, ax = plt.subplots(figsize=(HALF_PAGE_WIDTH_IN, HALF_PAGE_HEIGHT_IN), layout="constrained")
    l1 = ax.plot(result_df.day, result_df.rho, label="Spearman's ρ", lw=1)
    l2 = ax.plot(result_df.day, result_df.pearson_r, label="Pearson's r", lw=1)
    ax.set_xlabel("Day")
    ax.set_ylabel("Correlation")
    ax.set_xlim(left=0)
    ymin = min(result_df.rho.min(), result_df.pearson_r.min())
    ax.set_ylim(ymin - 0.05, 1.05)
    ax.axhline(0, color="gray", lw=0.5, ls="--")

    lines = l1 + l2
    if args.show_rmse:
        ax_rmse = ax.twinx()
        l3 = ax_rmse.plot(
            result_df.day,
            result_df.rmse,
            color="tab:green",
            ls="--",
            lw=1,
            label="RMSE",
        )
        ax_rmse.set_ylabel("RMSE")
        ax_rmse.set_ylim(bottom=0)
        lines = lines + l3

    ax.legend(lines, [line.get_label() for line in lines])
    print("Plotting results to", args.output)
    fig.savefig(args.output)

    if not args.scatter_output:
        return

    # Community-by-community scatter: one panel per day, ExaEpi rate vs Epicast rate, one point per
    # matched community. All panels share the same axis range so panels are visually comparable
    # across days, and a y=x reference line marks perfect agreement. Total width is fixed at the
    # paper's half-page width regardless of how many days there are (see paper_style.py) -- each
    # panel just gets narrower as more days are added.
    axis_max = max(max(df.rate_exaepi.max(), df.rate_epicast.max()) for df, _ in scatter_panels)
    axis_max = axis_max * 1.05 if axis_max > 0 else 1.0

    n = len(scatter_panels)
    panel_width = HALF_PAGE_WIDTH_IN / n
    fig2, axes2 = plt.subplots(1, n, figsize=(HALF_PAGE_WIDTH_IN, panel_width), squeeze=False, layout="constrained")
    axes2 = axes2[0]
    for ax, (df, label) in zip(axes2, scatter_panels):
        ax.plot([0, axis_max], [0, axis_max], color="gray", lw=0.5, ls="--", zorder=1)
        ax.scatter(df.rate_exaepi, df.rate_epicast, s=6, alpha=0.5, linewidths=0, zorder=2)
        ax.set_xlim(0, axis_max)
        ax.set_ylim(0, axis_max)
        ax.set_aspect("equal")
        ax.set_xlabel("ExaEpi infection rate")
        ax.set_ylabel("Epicast infection rate")
        ax.set_title(label)
    print("Plotting community-by-community scatter to", args.scatter_output)
    fig2.savefig(args.scatter_output)


if __name__ == "__main__":
    main()
