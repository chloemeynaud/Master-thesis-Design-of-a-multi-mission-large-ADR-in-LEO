"""
config.py
=========
Single source of truth for every constant and every cluster definition used in
the client-selection stage.

Nothing in this file computes anything. If a number appears in the thesis text,
it should appear here and nowhere else, so that a change made here propagates to
every figure and every score automatically.

------------------------------------------------------------------------------
INCONSISTENCIES FOUND IN THE ORIGINAL SCRIPTS  (resolve before the final run)
------------------------------------------------------------------------------
The five original scripts did not agree with each other. The values kept below
are the ones from `cluster_refinement.py`, which appears to be the most recent.
Every disagreement is marked with a [CHECK] comment; decide which value is
correct, fix it here, and rerun. See README.md for the full list.
"""

from __future__ import annotations  # allows "int | None" on Python < 3.10

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# 1. PATHS
# ---------------------------------------------------------------------------
# Resolved relative to this file, so the scripts run from anywhere and do not
# depend on the working directory Spyder happens to be in.
#
# Two layouts are supported:
#   repo layout : adr-thesis/src/config.py  -> ROOT = adr-thesis/
#   flat layout : New codes/config.py       -> ROOT = New codes/
# The repo layout is detected by the presence of a sibling data/ directory.
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
ESA_MASTER_FIGURE = "ESA Space Environment Report 2025, Fig. 2.9"
S_600_1000 = 4e-8         # objects/km^3, 600-1000 km
S_1000_1300 = 2e-8        # objects/km^3, 1000-1300 km

# [CHECK] Mass score thresholds. The code counts objects above 1 t, 2 t and
# 5 t CUMULATIVELY (a 6 t object scores 1 + 2 + 3 = 6), whereas the Client List
# Document describes non-cumulative classes 1-2 t, 2-3 t and > 3 t. Align the
# thesis text with whichever version produced the AHP scores.
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
                      Read from ESA MASTER (see section 5). Only needed for the
                      clusters that are scored; None otherwise.
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
    # Note: the original plot_clients.py scored this cluster at 70-75 deg in
    # the collision score but at 80-85 deg in the mass score. 80-85 deg is the
    # correct window and is now used for every score.
    Cluster("RB_800_83",    "Rocket Body",  800, 1000, 80,  85, rank=2,
            spatial_density=S_600_1000),
    Cluster("RB_700_74",    "Rocket Body",  700,  900, 70,  75, rank=3,
            spatial_density=S_600_1000),

    # The wide PL SSO cluster used in the AHP runs from 600 km. The refined
    # window (CLUSTERS_REFINED) extends down to 500 km, because four of the
    # candidate targets sit below 600 km (HELIOS 1A, HELIOS 2A, SKYMED 1 and
    # SKYMED 3).
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
    # Window verified against the target list: the six Ariane R/B lie between
    # 98.22 and 98.82 deg, and between 612 and 783 km.
    Cluster("RB_SSO",     "Rocket Body",  600,  900, 98.2, 98.9,
            rank=4, target_sheet="RB SSO"),
    # [CHECK] with a lower bound of 82.9 deg, three of the SL-8 R/B listed as
    # targets fall outside the cluster: 1975-028B (82.83 deg), 1982-012B
    # (82.89 deg), 1991-013B (82.83 deg). Widening to 82.8 deg captures them.
    Cluster("RB_800_83",  "Rocket Body",  800, 1000, 82.83, 83.0,
            rank=2, target_sheet="SL8 800-1000"),
    # [CHECK] get_RAAN.py used 71.0-71.1 while keeping the label "74-74.1".
    Cluster("RB_700_74",  "Rocket Body",  700,  900, 74.0, 74.1,
            rank=3, target_sheet="1 - SL8 700-900"),
    Cluster("PL_1000_88", "Payload",     1000, 1300, 87.8, 88.0,
            rank=5),
    # Window verified against the target list: the sixteen payloads lie between
    # 97.83 and 98.86 deg, and between 547 and 825 km.
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
# The six candidate clusters, ordered payloads first then rocket bodies.
# Defined as a selection of CLUSTERS_WIDE rather than a second list, so the
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

# --- 4d. AHP workbook ---------------------------------------------------------
# Workbook holding the AHP weights and the hand-filled grades of the other
# sub-criteria. AHP_COLUMNS maps each cluster to its grade column in the
# "scores" sheet.
AHP_FILE = DATA_DIR / "Client_prioritization_-_AHP_method.xlsx"
AHP_COLUMNS = {
    "RB_SSO": "D", "RB_800_83": "F", "RB_700_74": "H",
    "PL_SSO": "J", "PL_800_83": "L", "PL_1000_88": "N",
}


# --- 4e. RAAN sub-clusters --------------------------------------------------
# Groups of targets that can realistically be visited by the same servicer.
# Two criteria, not one:
#   * RAAN, so that the plane changes between successive targets stay
#     affordable (see figures.raan_wheel and figures.raan_gaps);
#   * platform, because targets of the same family share a structure, a
#     capture interface and an attitude behaviour, so one capture and
#     servicing strategy covers the whole group.
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
