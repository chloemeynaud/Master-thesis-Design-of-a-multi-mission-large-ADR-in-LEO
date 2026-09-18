"""
config.py
=========
Single source of truth for every constant and every cluster definition used in
the client-selection stage.

Nothing in this file computes anything. Every number that appears in the thesis
text is defined here and nowhere else, so that changing a value here propagates
to every figure and every score automatically.
"""

from __future__ import annotations  # allows "int | None" on Python < 3.10

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# 1. PATHS
# ---------------------------------------------------------------------------
# Resolved relative to this file, so that the scripts run from any working
# directory. Two layouts are supported:
#   repository : <root>/scripts/config.py  -> ROOT = <root>
#   flat       : <folder>/config.py        -> ROOT = <folder>
# The repository layout is detected by the presence of a sibling data/ folder.
_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent if (_HERE.parent / "data").is_dir() else _HERE

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

CATALOGUE_FILE = DATA_DIR / "RB_Payloads_notdecay_LEO_with_DISCOS.xlsx"
TARGETS_FILE = DATA_DIR / "Targets.xlsx"

# Sheet names inside CATALOGUE_FILE, mapped to the object-type label used
# everywhere downstream in the OBJ_TYPE column.
CATALOGUE_SHEETS = {
    "Rocket body": "Rocket Body",
    "Payload": "Payload",
}

# ---------------------------------------------------------------------------
# 2. LEO SELECTION FILTER
# ---------------------------------------------------------------------------
PERIOD_MAX_MIN = 119.0   # orbital period below which an object is treated as LEO

# Lower bound applied when the catalogue is loaded. Set to 400 km so that the
# altitude/inclination heatmaps cover the whole region of interest and so that
# no cluster is silently truncated by the global filter. Each cluster applies
# its own alt_lo on top of this, so the analysis floor is defined per cluster
# rather than globally.
ALT_MIN_KM = 400.0

# ---------------------------------------------------------------------------
# 3. BINNING
# ---------------------------------------------------------------------------
MASS_BINS = [0, 100, 500, 1000, 2000, 5000, float("inf")]
MASS_LABELS = ["< 100 kg", "100-500 kg", "500-1000 kg", "1-2 t", "2-5 t", "> 5 t"]
MASS_COLORS = ["m", "c", "b", "g", "r", "k"]

INC_BAND_EDGES = list(range(0, 155, 5))                       # 5 deg bands
INC_BAND_LABELS = [f"{i}-{i + 5}" for i in range(0, 150, 5)]

ALT_BAND_EDGES = list(range(400, 2100, 100))                  # 100 km bands
ALT_BAND_LABELS = [f"{a}-{a + 100}" for a in range(400, 2000, 100)]

# Shape strings in DISCOS are free text; these are the families kept.
SHAPE_MARKERS = {
    "Cylinder": "o",
    "Box": "P",
    "Sphere": "*",
    "Other": "s",
    "Unknown": "X",
}

MASS_CMAP = "plasma"
DENSITY_CMAP = "YlOrRd"


# ---------------------------------------------------------------------------
# 3b. COLLISION MODEL
# ---------------------------------------------------------------------------
# Expected number of collisions per year for the objects of a cluster
# (kinetic gas approach, Kessler & Cour-Palais 1978):
#     N_c = A_tot * S * v_rel * dt
# See scoring.collision_score.
V_REL_KMS =  7        # average relative velocity determined by Kessler & Cour-Palais [km/s]
DT_YEARS = 1.0            # exposure time [years]; same for every cluster

# Spatial densities of objects larger than 10 cm, read on the MASTER profile
# of ESA's Annual Space Environment Report 2025 (Issue 9.1), Fig. 2.9, p. 30,
# reference population 01/08/2024. Log scale: values are orders of magnitude.
S_600_1000 = 4e-8         # objects/km^3, 600-1000 km
S_1000_1300 = 2e-8        # objects/km^3, 1000-1300 km

# Mass score thresholds. Objects above 1 t, 2 t and 5 t are counted
# cumulatively, so a 6 t object contributes 1 + 2 + 3 = 6. This weights the
# heaviest objects of a cluster without letting a single one dominate the score.
MASS_SCORE_THRESHOLDS_KG = (1000, 2000, 5000)
MASS_SCORE_WEIGHTS = (1.0, 2.0, 3.0)


# ---------------------------------------------------------------------------
# 4. CLUSTER DEFINITIONS
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Cluster:
    """A rectangular region of the (mean altitude, inclination) plane.

    Attributes
    ----------
    name      : short identifier, safe for filenames and Excel sheet names
    obj_type  : 'Rocket Body' or 'Payload' - matches the OBJ_TYPE column
    alt_lo/hi : mean altitude bounds in km, lower bound inclusive
    inc_lo/hi : inclination bounds in degrees, lower bound inclusive
    rank      : priority rank from the AHP study (1 = highest), None if unranked
    target_sheet : sheet of Targets.xlsx listing the objects selected for
                   removal in this cluster, or None
    spatial_density : spatial density of catalogued-size objects (> 10 cm)
                      over the altitude range of the cluster, in objects/km^3.
                      Read from ESA MASTER (see section 3b). Only needed for
                      the clusters that are scored; None otherwise.
    """

    name: str
    obj_type: str
    alt_lo: float
    alt_hi: float
    inc_lo: float
    inc_hi: float
    rank: Optional[int] = None
    target_sheet: Optional[str] = None
    spatial_density: Optional[float] = None

    @property
    def label(self) -> str:
        """Human-readable label used in figure titles."""
        prefix = "RB" if self.obj_type == "Rocket Body" else "PL"
        return (f"{prefix} - {self.alt_lo:.0f}-{self.alt_hi:.0f} km / "
                f"{self.inc_lo}-{self.inc_hi} deg")


# --- 4a. Wide clusters: the six candidates scored by the AHP -----------------
# These are the coarse boxes read off the altitude/inclination heatmaps.
CLUSTERS_WIDE = [
    Cluster("RB_SSO",       "Rocket Body",  600,  900, 95, 100, rank=4,
            spatial_density=S_600_1000),
    # The 80-85 deg window captures the SL-8 stages around 83 deg.
    Cluster("RB_800_83",    "Rocket Body",  800, 1000, 80,  85, rank=2,
            spatial_density=S_600_1000),
    Cluster("RB_700_74",    "Rocket Body",  700,  900, 70,  75, rank=3,
            spatial_density=S_600_1000),

    # The wide SSO payload cluster starts at 600 km, which is where the AHP
    # scores are computed. The refined window below extends down to 500 km, so
    # that the Cosmo-SkyMed satellites sitting around 550 km are included.
    Cluster("PL_SSO",       "Payload",      600,  900, 95, 100, rank=1,
            spatial_density=S_600_1000),
    Cluster("PL_800_83",    "Payload",      800, 1000, 80,  85,
            spatial_density=S_600_1000),
    Cluster("PL_1000_88",   "Payload",     1000, 1300, 85,  90, rank=5,
            spatial_density=S_1000_1300),
]

# --- 4b. Refined clusters: narrow inclination bands ------------------------
# Obtained by zooming into the 0.1 deg inclination histograms of the wide
# clusters. These are the windows the ADR targets are drawn from.
CLUSTERS_REFINED = [
    # The six Ariane upper stages lie between 98.22 and 98.82 deg, and between
    # 612 and 783 km.
    Cluster("RB_SSO",     "Rocket Body",  600,  900, 98.2, 98.9,
            rank=4, target_sheet="RB SSO"),
    # The lower bound is set at 82.83 deg so that the whole SL-8 population is
    # captured: three of the stages lie just below 82.9 deg.
    Cluster("RB_800_83",  "Rocket Body",  800, 1000, 82.83, 83.0,
            rank=2, target_sheet="SL8 800-1000"),
    Cluster("RB_700_74",  "Rocket Body",  700,  900, 74.0, 74.1,
            rank=3, target_sheet="1 - SL8 700-900"),
    Cluster("PL_1000_88", "Payload",     1000, 1300, 87.8, 88.0,
            rank=5),
    # The sixteen candidate payloads lie between 97.83 and 98.86 deg, and
    # between 547 and 825 km.
    Cluster("PL_SSO",     "Payload",      500,  900, 97.8, 98.9,
            rank=1, target_sheet="PL SSO"),
]


def by_name(clusters: List[Cluster], name: str) -> Cluster:
    """Return the cluster with the given short name, for one-off calls."""
    for c in clusters:
        if c.name == name:
            return c
    raise KeyError(f"No cluster named {name!r}. Available: "
                   f"{[c.name for c in clusters]}")


# --- 4c. Clusters shown on the mass-distribution panel ----------------------
# The six candidate clusters, ordered payloads first then rocket bodies. Built
# by selecting from CLUSTERS_WIDE rather than as a second list, so that the
# bounds stay defined in exactly one place.
MASS_PANEL_ORDER = [
    "PL_800_83",    # Payloads,      80-85 deg,   800-1000 km
    "PL_1000_88",   # Payloads,      85-90 deg,  1000-1300 km
    "PL_SSO",       # Payloads,      95-100 deg,  600-900 km
    "RB_700_74",    # Rocket bodies, 70-75 deg,   700-900 km
    "RB_800_83",    # Rocket bodies, 80-85 deg,  800-1000 km
    "RB_SSO",       # Rocket bodies, 95-100 deg,  600-900 km
]
CLUSTERS_MASS_PANEL = [by_name(CLUSTERS_WIDE, n) for n in MASS_PANEL_ORDER]

# --- 4d. RAAN sub-clusters --------------------------------------------------
# Groups of targets that can realistically be visited by the same servicer.
# Two criteria, not one:
#   * RAAN, so that the plane changes between successive targets stay
#     affordable (see figures.raan_wheel and figures.raan_gaps);
#   * platform, because targets of the same family share a structure, a
#     capture interface and an attitude behaviour, so that a single servicer
#     design covers the whole group.
# The second criterion is why the Spot group is kept despite a RAAN spread of
# 59 deg, wider than the 37 deg of the Cosmo-SkyMed group.
# Objects listed in no group are isolated on both counts.
SUBCLUSTERS = {
    "PL_SSO": {
        # Same mission class, three agencies, the tightest group in RAAN
        "Metop-A / ERS-1 / Radarsat-1": ["2006-044A", "1991-050A",
                                         "1995-059A"],
        # Spot 3, 4 and 5: same Spot Mk.2/Mk.3 bus
        "Spot 3-5": ["1993-061A", "1998-017A", "2002-021A"],
        # Cosmo-SkyMed 1 to 4: four identical satellites, 2 and 4 coplanar
        "Cosmo-SkyMed 1-4": ["2007-023A", "2007-059A", "2008-054A",
                             "2010-060A"],
    },
    "RB_SSO": {
        # Ariane 40 and 40+ H10 stages: identical design
        "Ariane 4 upper stages": ["1998-017B", "1990-005H",
                                  "1993-061H", "1995-021B"],
    },
}
