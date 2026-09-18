"""
fetch_external_data.py
======================
Downloads the two external datasets. Run this only when the catalogue needs to
be refreshed; the analysis itself reads the Excel files in data/.

    export DISCOS_TOKEN="..."
    export SPACETRACK_USER="..." SPACETRACK_PASS="..."
    python scripts/fetch_external_data.py --discos
    python scripts/fetch_external_data.py --raan
"""

import argparse
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
from catalogue import load_catalogue, select_cluster
from external_data import fetch_discos, fetch_raan


def run_discos(object_class):
    df = fetch_discos(object_class=object_class)
    out = cfg.DATA_DIR / f"discos_{object_class.lower().replace(' ', '_')}.csv"
    df.to_csv(out, index=False)
    print(f"{len(df)} objects written to {out}")


def run_raan():
    """RAAN of every object of the refined clusters, one sheet per cluster.

    The RAAN is not in the base catalogue and is needed at the mission-design
    stage to evaluate the plane changes between successive targets.
    """
    catalogue = load_catalogue()

    frames = []
    for cluster in cfg.CLUSTERS_REFINED:
        subset = select_cluster(catalogue, cluster)
        subset = subset[subset["OBJECT_NUMBER"].notna()].copy()
        subset["CLUSTER"] = cluster.name
        frames.append(subset[["CLUSTER", "OBJ_TYPE", "OBJECT_NUMBER",
                              "OBJECT_ID", "SATNAME", "INCLINATION",
                              "ALT_MEAN", "MASS_KG"]])
    base = pd.concat(frames, ignore_index=True).drop_duplicates("OBJECT_NUMBER")
    base["NORAD_ID"] = base["OBJECT_NUMBER"].astype(int)
    print(f"{len(base)} unique objects across {len(cfg.CLUSTERS_REFINED)} "
          f"clusters")

    raan = fetch_raan(base["NORAD_ID"].tolist(),
                      cache_file=cfg.DATA_DIR / "tle_cache.csv")
    result = base.merge(raan, on="NORAD_ID", how="left")

    out = cfg.DATA_DIR / "RAAN_by_cluster.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="All_Objects", index=False)
        for cluster in cfg.CLUSTERS_REFINED:
            (result[result["CLUSTER"] == cluster.name]
             .to_excel(writer, sheet_name=cluster.name[:31], index=False))
        (result.groupby("CLUSTER")["RAAN_DEG"]
         .agg(N="count", mean="mean", std="std", min="min", max="max")
         .reset_index()
         .to_excel(writer, sheet_name="RAAN_statistics", index=False))
    print(f"Written to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discos", action="store_true",
                        help="download the DISCOS physical properties")
    parser.add_argument("--object-class", default="Payload",
                        help="DISCOS object class (default: Payload)")
    parser.add_argument("--raan", action="store_true",
                        help="download the RAAN of the cluster objects")
    args = parser.parse_args()

    if args.discos:
        run_discos(args.object_class)
    if args.raan:
        run_raan()
    if not (args.discos or args.raan):
        parser.print_help()
