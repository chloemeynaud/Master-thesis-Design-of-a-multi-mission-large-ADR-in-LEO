"""
run_client_selection.py
=======================
Stage 1 of the thesis: client choice and categorisation.

This is the only file to run for this stage. It contains no analysis logic, only
the sequence of calls, so that the order of the steps mirrors the order of the
corresponding section of the thesis. Set the STEPS flags below to run a subset.

    python scripts/run_client_selection.py
"""

import sys
from pathlib import Path

# Make src/ importable. Works both when the files sit in src/ + scripts/ and
# when they are all in one folder (as when opened directly in Spyder).
_HERE = Path(__file__).resolve().parent
for _candidate in (_HERE.parent / "src", _HERE):
    if (_candidate / "config.py").is_file():
        sys.path.insert(0, str(_candidate))
        break

import pandas as pd

import config as cfg
import figures as fg
from config import by_name
from catalogue import (load_catalogue, load_targets, query_objects,
                       select_cluster, subcluster_summary, summarise_clusters)
from scoring import score_clusters

# ---------------------------------------------------------------------------
STEPS = {
    "overview": True,        # 2.1 population of LEO by altitude and inclination
    "scoring": True,         # 2.2 cluster scores feeding the AHP matrix
    "refinement": True,      # 2.3 narrowing each cluster to an inclination band
    "targets": True,         # 2.4 the selected objects inside each cluster
    "raan": True,            # 2.5 RAAN refinement into sub-clusters
    "export": True,          # write the tables used in the thesis
}
SHOW = True                  # False to only write PNGs, useful for a batch run

# Clusters used for the mass-weighted inclination histograms: the wide,
# non-refined windows, so the narrow bands stand out against their surroundings.
INC_HIST_CLUSTERS = [
    by_name(cfg.CLUSTERS_WIDE, "PL_SSO"),   # 600-900 km
    by_name(cfg.CLUSTERS_WIDE, "RB_700_74"),
    by_name(cfg.CLUSTERS_WIDE, "RB_800_83"),
    by_name(cfg.CLUSTERS_WIDE, "RB_SSO"),
]

# Clusters that get a target scatter. Only the two SSO clusters are kept: the
# others hold 50 to 139 objects that are all SL-8 R/B of the same mass and the
# same shape, so the scatter has nothing to distinguish between them. Their
# content is in the Excel export instead.
SCATTER_CLUSTERS = [cfg.by_name(cfg.CLUSTERS_REFINED, name)
                    for name in ("RB_SSO", "PL_SSO")]

# Clusters that get a RAAN wheel: the two SSO clusters again. The SL-8 and
# SL-16 clusters hold 50 to 142 stages spread over the whole 0-360 deg range,
# so their sub-clusters are chosen when the mission scenarios are built.
RAAN_CLUSTERS = [cfg.by_name(cfg.CLUSTERS_REFINED, name)
                 for name in ("PL_SSO", "RB_SSO")]

# The Ariane stages all share the name "ARIANE 40 R/B", so the RAAN wheel of
# the rocket bodies is labelled with the COSPAR ID instead.
RAAN_LABEL_COL = {"RB_SSO": "OBJECT_ID"}

# Manual label nudges for target names that would overlap a legend.
LABEL_OFFSETS = {
    "2007-059A": (40, 10),    # SKYMED 2, same point as SKYMED 4
    "2010-060A": (40, -2),    # SKYMED 4
    "2008-054A": (30, 0),     # SKYMED 3, too close to the left axis
    "1986-019C": (38, 8),     # ARIANE 1 R/B
    "2004-049H": (42, 4),     # ARIANE 5 R/B
}
# ---------------------------------------------------------------------------


def main():
    fg.apply_style()
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    catalogue = load_catalogue()

    # -- 2.1 Population overview ------------------------------------------
    # One heatmap per object type, from 400 km up, coloured by the total mass
    # in each 100 km x 5 deg cell. The number printed in a cell is still the
    # object count.
    if STEPS["overview"]:
        for obj_type in ("Payload", "Rocket Body"):
            tag = obj_type.split()[0].lower()
            fg.heatmap_alt_inc(catalogue, obj_type, colour_by="mass",
                               save_as=f"heatmap_mass_{tag}.png", show=SHOW)

    # -- 2.2 Cluster scoring ----------------------------------------------
    # The AHP matrix itself lives in Client_prioritization_AHP_method.xlsx and
    # is filled in by hand with the _AHP levels printed here. The scores are
    # also exported with the other tables (see "export").
    scores =  None
    if STEPS["scoring"]:
        scores = score_clusters(catalogue, cfg.CLUSTERS_WIDE)
        print("\nCluster scores")
        print(scores.to_string(index=False))


    # -- 2.3 Cluster refinement -------------------------------------------
    # Mass-weighted inclination distribution over the wide cluster windows: bar
    # height is the object count, bar colour is the total mass in the bin. This
    # is the figure the narrow bands of CLUSTERS_REFINED are read off.
    if STEPS["refinement"]:
        for cluster in INC_HIST_CLUSTERS:
            fg.inclination_histogram(catalogue, cluster, weight_by_mass=True,
                                     save_as=f"inc_hist_{cluster.name}.png",
                                     show=SHOW)

        fg.mass_class_panel(catalogue, cfg.CLUSTERS_MASS_PANEL,
                            save_as="mass_classes_per_cluster.png", show=SHOW)

    # -- 2.4 Selected targets ---------------------------------------------
    # One scatter per SSO cluster, with the objects listed in Targets.xlsx
    # ringed and labelled with their name.
    if STEPS["targets"]:
        for cluster in SCATTER_CLUSTERS:
            print(f"\n{cluster.label}")
            targets = (load_targets(cluster.target_sheet)
                       if cluster.target_sheet else None)
            fg.cluster_scatter(catalogue, cluster, targets=targets,
                               label_offsets=LABEL_OFFSETS,
                               save_as=f"scatter_{cluster.name}.png",
                               show=SHOW)

    # -- 2.5 RAAN refinement ----------------------------------------------
    # Orientation of the orbital planes of the candidate targets. Objects that
    # sit close together on the wheel share a plane orientation; the groups
    # themselves are defined in config.SUBCLUSTERS, on RAAN and on platform
    # commonality together.
    raan_tables = []
    if STEPS["raan"]:
        for cluster in RAAN_CLUSTERS:
            targets = load_targets(cluster.target_sheet)
            if targets is None or "RAAN_DEG" not in targets.columns:
                print(f"{cluster.label}: no RAAN in the target sheet, skipped")
                continue
            groups = cfg.SUBCLUSTERS.get(cluster.name, {})
            fg.raan_wheel(catalogue, cluster, targets, groups=groups,
                          label_col=RAAN_LABEL_COL.get(cluster.name,
                                                       "SATNAME"),
                          save_as=f"raan_{cluster.name}.png", show=SHOW)

            print(f"\nRAAN gaps - {cluster.label}")
            print(fg.raan_gaps(targets).to_string(index=False))

            summary = subcluster_summary(catalogue, cluster, targets, groups)
            print(f"\nSub-clusters - {cluster.label}")
            print(summary.to_string(index=False))
            raan_tables.append(summary)

    # -- Tables for the thesis --------------------------------------------
    if STEPS["export"]:
        out = cfg.OUTPUT_DIR / "client_selection_tables.xlsx"
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            summarise_clusters(catalogue, cfg.CLUSTERS_WIDE).to_excel(
                writer, sheet_name="wide_clusters", index=False)
            summarise_clusters(catalogue, cfg.CLUSTERS_REFINED).to_excel(
                writer, sheet_name="refined_clusters", index=False)
            if scores is not None:
                scores.to_excel(writer, sheet_name="cluster_scores",
                                index=False)
            if raan_tables:
                pd.concat(raan_tables, ignore_index=True).to_excel(
                    writer, sheet_name="subclusters", index=False)
            for cluster in cfg.CLUSTERS_REFINED:
                select_cluster(catalogue, cluster).to_excel(
                    writer, sheet_name=cluster.name[:31], index=False)
        print(f"\nTables written to {out}")


if __name__ == "__main__":
    main()
