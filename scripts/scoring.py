"""
scoring.py
==========
Cluster-level scores feeding the AHP client-prioritisation matrix.

Each criterion is a function of (subset, cluster), where `subset` holds the
objects of the cluster and `cluster` its definition (bounds, spatial density).
A new criterion is added by writing one function and adding it to SCORE_FUNCS.

Collision probability
---------------------
The collision criterion is evaluated PER OBJECT. For every object i of the
cluster, the kinetic gas approach (Kessler & Cour-Palais, 1978) gives its
expected number of collisions over dt:

    P_i = A_i * S * v_rel * dt

with A_i its cross-sectional area, S the spatial density of objects > 10 cm in
the altitude range of the cluster, and v_rel the mean relative velocity. The
cluster score is the mean of P_i over the objects of the cluster, i.e. the
collision probability of a typical object that a removal mission would target.

Neither the mass nor the number of objects enters this score: both are already
assessed by the mass and repeat-business sub-criteria.
"""

from __future__ import annotations

import pandas as pd

import config as cfg
from catalogue import select_cluster

SECONDS_PER_YEAR = 365.25 * 86400.0


# ---------------------------------------------------------------------------
# Collision probability
# ---------------------------------------------------------------------------
def collision_probability_per_object(subset: pd.DataFrame,
                                     cluster: cfg.Cluster) -> pd.Series:
    """Expected number of collisions per year of each object of the cluster.

    Returns a Series aligned on `subset` (one value per object). Objects whose
    area is unknown get NaN, and are therefore ignored in the cluster mean.
    """
    if cluster.spatial_density is None:
        return pd.Series(float("nan"), index=subset.index)
    area_km2 = subset["AREA_M2"] * 1e-6
    dt_s = cfg.DT_YEARS * SECONDS_PER_YEAR
    return area_km2 * cluster.spatial_density * cfg.V_REL_KMS * dt_s


def collision_score(subset: pd.DataFrame, cluster: cfg.Cluster) -> float:
    """Mean collision probability per object of the cluster [1/yr]."""
    if subset.empty:
        return 0.0
    return float(collision_probability_per_object(subset, cluster).mean())


def cluster_collision_rate(subset: pd.DataFrame, cluster: cfg.Cluster) -> float:
    """Sum of P_i over the cluster: expected collisions/yr of the whole cluster.

    Not used in the AHP. Kept for the sensitivity analysis only.
    """
    if subset.empty:
        return 0.0
    p = collision_probability_per_object(subset, cluster)
    # Objects with unknown area are counted with the mean of the cluster
    return float(p.mean() * len(subset))


def legacy_collision_proxy(subset: pd.DataFrame,
                           cluster: cfg.Cluster = None) -> float:
    """Previous proxy, N * mean(area) * mean(mass) / 1000.

    Not used in the AHP. Kept for the sensitivity analysis only.
    """
    if subset.empty:
        return 0.0
    return (len(subset) * subset["AREA_M2"].mean()
            * subset["MASS_KG"].mean() / 1000.0)


# ---------------------------------------------------------------------------
# Mass
# ---------------------------------------------------------------------------
def mass_score(subset: pd.DataFrame, cluster: cfg.Cluster = None) -> float:
    """Weighted count of heavy objects.

    Objects above each threshold of config.MASS_SCORE_THRESHOLDS_KG are counted
    and weighted by config.MASS_SCORE_WEIGHTS. The thresholds are cumulative:
    with the default values, a 6 t object contributes 1 + 2 + 3 = 6.
    """
    if subset.empty:
        return 0.0
    m = subset["MASS_KG"]
    return float(sum(w * (m >= t).sum() for t, w in
                     zip(cfg.MASS_SCORE_THRESHOLDS_KG,
                         cfg.MASS_SCORE_WEIGHTS)))


def total_mass_score(subset: pd.DataFrame, cluster: cfg.Cluster = None) -> float:
    """Total mass in the cluster, in tonnes. Kept as an alternative criterion."""
    if subset.empty:
        return 0.0
    return subset["MASS_KG"].sum() / 1000.0


# Criteria used in the AHP. Add one here and it appears in the output table.
SCORE_FUNCS = {
    "COLLISION_SCORE": collision_score,
    "MASS_SCORE": mass_score,
    "TOTAL_MASS_T": total_mass_score,
}

# Alternative collision metrics compared in the sensitivity analysis.
COLLISION_VARIANTS = {
    "PER_OBJECT": collision_score,
    "CLUSTER_TOTAL": cluster_collision_rate,
    "LEGACY_PROXY": legacy_collision_proxy,
}


# ---------------------------------------------------------------------------
# AHP normalisation
# ---------------------------------------------------------------------------
def to_ahp_levels(values: pd.Series, n_levels: int = 5) -> pd.Series:
    """Map raw scores onto the 1..n_levels scale used in the AHP matrix.

    Rank-based quantile binning: the result depends only on the ordering of
    the clusters. Clusters with a NaN score get a missing level (<NA>).
    """
    levels = pd.Series(pd.NA, index=values.index, dtype="Int64")
    valid = values.dropna()
    if valid.empty:
        return levels
    q = min(n_levels, len(valid))
    binned = pd.qcut(valid.rank(method="first"), q=q,
                     labels=list(range(1, q + 1))).astype(int)
    levels.loc[valid.index] = binned
    return levels


def score_clusters(cat: pd.DataFrame, clusters: list[cfg.Cluster],
                   n_levels: int = 5) -> pd.DataFrame:
    """Score every cluster on every criterion and add the AHP levels."""
    rows = []
    for c in clusters:
        subset = select_cluster(cat, c)
        row = {"CLUSTER": c.name, "LABEL": c.label,
               "N_OBJECTS": len(subset),
               "MEAN_AREA_M2": subset["AREA_M2"].mean(),
               "SPATIAL_DENSITY_KM3": c.spatial_density}
        for score_name, func in SCORE_FUNCS.items():
            row[score_name] = func(subset, c)
        rows.append(row)

    df = pd.DataFrame(rows)
    for score_name in SCORE_FUNCS:
        df[f"{score_name}_AHP"] = to_ahp_levels(df[score_name], n_levels)
    return df


def collision_variants(cat: pd.DataFrame, clusters: list[cfg.Cluster],
                       n_levels: int = 5) -> pd.DataFrame:
    """Raw value and AHP level of every collision metric, for every cluster."""
    rows = []
    for c in clusters:
        subset = select_cluster(cat, c)
        row = {"CLUSTER": c.name}
        for name, func in COLLISION_VARIANTS.items():
            row[name] = func(subset, c)
        rows.append(row)
    df = pd.DataFrame(rows)
    for name in COLLISION_VARIANTS:
        df[f"{name}_AHP"] = to_ahp_levels(df[name], n_levels)
    return df
