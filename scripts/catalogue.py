"""
catalogue.py
============
Loading and preparation of the object catalogue.

This module replaces the block of code that was copy-pasted at the top of
`plot_clients.py`, `cluster_refinement.py`, `mass_distribution.py`,
`recup_code_clientdoc.py` and `get_RAAN.py`.

The key change is that rocket bodies and payloads are held in ONE DataFrame
with an `OBJ_TYPE` column, instead of two DataFrames (`rb_leo`, `pl_leo`)
carried side by side. Every downstream function then takes a single catalogue
and a Cluster, which removes the need to duplicate each analysis twice.

Typical use
-----------
    from catalogue import load_catalogue, select_cluster
    from config import CLUSTERS_REFINED

    cat = load_catalogue()
    subset = select_cluster(cat, CLUSTERS_REFINED[0])
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg


# ---------------------------------------------------------------------------
# Derived columns
# ---------------------------------------------------------------------------
def _mean_altitude(df: pd.DataFrame) -> pd.Series:
    """Mean altitude in km, as the average of apogee and perigee altitudes.

    Computed rather than read from the MEAN ALTITUDE / MEAN_ALTITUDE column,
    because that column is named differently in the two sheets of the source
    workbook and is not always populated.
    """
    return (df["APOGEE"] + df["PERIGEE"]) / 2.0


def _cross_section(df: pd.DataFrame) -> pd.Series:
    """Geometric cross-sectional area in m^2.

    Disc of the given diameter when a diameter is known, otherwise the
    width x depth rectangle. This is the definition used for the marker sizes
    in the cluster scatter plots.
    """
    return pd.Series(
        np.where(
            df["DIAMETER_M"].notna(),
            np.pi * (df["DIAMETER_M"] / 2.0) ** 2,
            df["WIDTH_M"] * df["DEPTH_M"],
        ),
        index=df.index,
    )


def _mean_area(df: pd.DataFrame) -> pd.Series:
    """Average of the available area estimates, in m^2.

    Averages width x height, the disc of the diameter, and span x width,
    ignoring the ones that cannot be computed. This is a rough proxy for the
    area an object presents to a random impactor, and is the quantity used in
    the collision score.

    NOTE: this is a different definition from _cross_section above. The
    original scripts used the averaged version for the collision score and the
    geometric version for the scatter plots. Both are kept so the difference is
    explicit, but the thesis should state which one is used where.
    """
    parts = []
    if {"WIDTH_M", "HEIGHT_M"} <= set(df.columns):
        parts.append(df["WIDTH_M"] * df["HEIGHT_M"])
    if "DIAMETER_M" in df.columns:
        parts.append(np.pi * (df["DIAMETER_M"] / 2.0) ** 2)
    if {"SPAN_M", "WIDTH_M"} <= set(df.columns):
        parts.append(df["SPAN_M"] * df["WIDTH_M"])
    if not parts:
        return pd.Series(np.nan, index=df.index)
    return pd.concat(parts, axis=1).mean(axis=1)


def _volume(df: pd.DataFrame) -> pd.Series:
    """Bounding-volume approximation in m^3: cylinder if possible, else box."""
    return pd.Series(
        np.where(
            df["DIAMETER_M"].notna() & df["HEIGHT_M"].notna(),
            np.pi * (df["DIAMETER_M"] / 2.0) ** 2 * df["HEIGHT_M"],
            df["WIDTH_M"] * df["DEPTH_M"] * df["HEIGHT_M"],
        ),
        index=df.index,
    )


def simplify_shape(shape) -> str:
    """Collapse the free-text DISCOS shape string into one of five families."""
    if pd.isna(shape):
        return "Unknown"
    if "Cyl" in shape:
        return "Cylinder"
    if "Sphere" in shape:
        return "Sphere"
    if "Box" in shape:
        return "Box"
    return "Other"


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add every column the analysis needs. Returns a new DataFrame."""
    out = df.copy()

    out["MASS_KG"] = pd.to_numeric(out["MASS_KG"], errors="coerce")
    out["ALT_MEAN"] = _mean_altitude(out)
    out["CROSS_SECTION"] = _cross_section(out)
    out["AREA_M2"] = _mean_area(out)
    out["VOLUME_M3"] = _volume(out)
    out["CHAR_LENGTH"] = out[["WIDTH_M", "HEIGHT_M", "DEPTH_M",
                              "DIAMETER_M"]].max(axis=1)
    out["SHAPE_SIMPLE"] = out["SHAPE"].apply(simplify_shape)

    # Categorical bands, used by the heatmaps and the mass-class bar charts
    out["MASS_CLASS"] = pd.cut(
        out["MASS_KG"].where(out["MASS_KG"] > 0),
        bins=cfg.MASS_BINS, labels=cfg.MASS_LABELS, right=False,
    )
    out["ALT_BAND"] = pd.cut(
        out["ALT_MEAN"], bins=cfg.ALT_BAND_EDGES,
        labels=cfg.ALT_BAND_LABELS, right=False,
    )
    out["INC_BAND"] = pd.cut(
        out["INCLINATION"], bins=cfg.INC_BAND_EDGES,
        labels=cfg.INC_BAND_LABELS, right=False,
    )
    return out


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_catalogue(path=None, alt_min_km=None, period_max_min=None,
                   verbose=True) -> pd.DataFrame:
    """Load both sheets of the catalogue, filter to LEO, add derived columns.

    Parameters
    ----------
    path : path to the workbook. Defaults to config.CATALOGUE_FILE.
    alt_min_km : lower bound on mean altitude. Defaults to config.ALT_MIN_KM.
                 Pass config.ALT_MIN_HEATMAP_KM to reproduce the wide
                 exploratory heatmaps, which started at 400 km.
    period_max_min : LEO cut on orbital period. Defaults to
                     config.PERIOD_MAX_MIN.

    Returns
    -------
    DataFrame with one row per object and an OBJ_TYPE column set to
    'Rocket Body' or 'Payload'.
    """
    path = path or cfg.CATALOGUE_FILE
    alt_min_km = cfg.ALT_MIN_KM if alt_min_km is None else alt_min_km
    period_max_min = (cfg.PERIOD_MAX_MIN if period_max_min is None
                      else period_max_min)

    frames = []
    for sheet, obj_type in cfg.CATALOGUE_SHEETS.items():
        df = pd.read_excel(path, sheet_name=sheet)
        df["OBJ_TYPE"] = obj_type
        frames.append(df)
    cat = pd.concat(frames, ignore_index=True)

    cat = add_derived_columns(cat)

    n_before = len(cat)
    cat = cat[
        (cat["PERIOD"] < period_max_min)
        & cat["INCLINATION"].notna()
        & (cat["ALT_MEAN"] >= alt_min_km)
    ].copy()

    if verbose:
        counts = cat["OBJ_TYPE"].value_counts()
        print(f"Catalogue: {n_before} objects loaded, {len(cat)} kept "
              f"(period < {period_max_min} min, alt >= {alt_min_km:.0f} km)")
        for obj_type, n in counts.items():
            with_mass = cat.loc[cat["OBJ_TYPE"] == obj_type,
                                "MASS_KG"].notna().sum()
            print(f"  {obj_type:<12} {n:>6}  ({with_mass} with a known mass)")
    return cat


def load_targets(sheet: str, path=None):
    """Load one sheet of the ADR target list.

    Blank separator rows are dropped and duplicate COSPAR IDs removed.

    Returns None, with a warning, if the workbook or the sheet is missing,
    rather than raising. A missing target list must not stop the rest of the
    run: the scatter plots are still worth producing without the rings.
    """
    path = Path(path or cfg.TARGETS_FILE)
    if not path.is_file():
        print(f"  WARNING: {path} not found, targets will not be shown. "
              f"Place Targets.xlsx in {cfg.DATA_DIR}")
        return None
    try:
        t = pd.read_excel(path, sheet_name=sheet)
    except ValueError:
        print(f"  WARNING: no sheet named {sheet!r} in {path.name}, "
              f"targets will not be shown")
        return None

    keep = [c for c in ("OBJECT_ID", "SATNAME", "MASS_KG", "RAAN_DEG")
            if c in t.columns]
    return (t.dropna(subset=["OBJECT_ID"])[keep]
             .drop_duplicates("OBJECT_ID")
             .reset_index(drop=True))


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
def select_cluster(cat: pd.DataFrame, cluster: cfg.Cluster) -> pd.DataFrame:
    """Return the objects of `cat` that fall inside `cluster`.

    Bounds are inclusive on the low side and exclusive on the high side, which
    is the convention used throughout the original scripts.
    """
    return cat[
        (cat["OBJ_TYPE"] == cluster.obj_type)
        & (cat["ALT_MEAN"] >= cluster.alt_lo)
        & (cat["ALT_MEAN"] < cluster.alt_hi)
        & (cat["INCLINATION"] >= cluster.inc_lo)
        & (cat["INCLINATION"] < cluster.inc_hi)
    ].copy()


def query_objects(cat: pd.DataFrame, alt_lo, alt_hi, inc_lo, inc_hi,
                  mass_lo=None, mass_hi=None, obj_type=None, country=None,
                  sort_by="MASS_KG") -> pd.DataFrame:
    """Free-form query over the catalogue, for exploratory work.

    Returns the matching objects sorted by decreasing mass. Use
    `figures.object_table` to render the result as a figure, or
    `.to_excel(...)` to export it.
    """
    mask = (
        (cat["ALT_MEAN"] >= alt_lo) & (cat["ALT_MEAN"] < alt_hi)
        & (cat["INCLINATION"] >= inc_lo) & (cat["INCLINATION"] < inc_hi)
    )
    if obj_type is not None:
        mask &= cat["OBJ_TYPE"] == obj_type
    if mass_lo is not None:
        mask &= cat["MASS_KG"] >= mass_lo
    if mass_hi is not None:
        mask &= cat["MASS_KG"] < mass_hi
    if country is not None:
        mask &= cat["COUNTRY"].str.upper() == country.upper()

    cols = ["OBJECT_ID", "SATNAME", "COUNTRY", "OBJ_TYPE", "SHAPE", "RCS_SIZE",
            "WIDTH_M", "HEIGHT_M", "DIAMETER_M", "SPAN_M", "INCLINATION",
            "ALT_MEAN", "MASS_KG", "LAUNCH_YEAR"]
    out = cat.loc[mask, [c for c in cols if c in cat.columns]]
    return out.sort_values(sort_by, ascending=False).reset_index(drop=True)


def summarise_clusters(cat: pd.DataFrame,
                       clusters: list[cfg.Cluster]) -> pd.DataFrame:
    """One row per cluster: object count, total mass, mean mass, mean area."""
    rows = []
    for c in clusters:
        s = select_cluster(cat, c)
        rows.append({
            "CLUSTER": c.name,
            "LABEL": c.label,
            "OBJ_TYPE": c.obj_type,
            "N_OBJECTS": len(s),
            "N_WITH_MASS": int(s["MASS_KG"].notna().sum()),
            "TOTAL_MASS_T": s["MASS_KG"].sum() / 1000.0,
            "MEAN_MASS_KG": s["MASS_KG"].mean(),
            "MEAN_AREA_M2": s["AREA_M2"].mean(),
        })
    return pd.DataFrame(rows)


def subcluster_summary(cat: pd.DataFrame, cluster: cfg.Cluster,
                       targets: pd.DataFrame, groups: dict) -> pd.DataFrame:
    """One row per RAAN sub-cluster: extent in altitude, inclination and RAAN.

    This is the table the summary table of the thesis is filled from. Targets
    listed in no group are collected in a final 'isolated' row.
    """
    cols = ["OBJECT_ID", "ALT_MEAN", "INCLINATION"]
    df = targets.merge(cat[cols], on="OBJECT_ID", how="left")
    member_of = {oid: name for name, ids in (groups or {}).items()
                 for oid in ids}
    df["SUBCLUSTER"] = df["OBJECT_ID"].map(member_of).fillna("isolated")

    out = (df.groupby("SUBCLUSTER")
             .agg(N_OBJECTS=("OBJECT_ID", "size"),
                  ALT_LO=("ALT_MEAN", "min"), ALT_HI=("ALT_MEAN", "max"),
                  INC_LO=("INCLINATION", "min"), INC_HI=("INCLINATION", "max"),
                  RAAN_LO=("RAAN_DEG", "min"), RAAN_HI=("RAAN_DEG", "max"),
                  TOTAL_MASS_T=("MASS_KG", lambda m: m.sum() / 1000.0))
             .reset_index())
    out.insert(0, "CLUSTER", cluster.name)
    out["RAAN_SPREAD"] = out["RAAN_HI"] - out["RAAN_LO"]
    return out.round(2)
