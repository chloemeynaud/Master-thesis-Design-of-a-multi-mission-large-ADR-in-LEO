"""
figures.py
==========
Every figure of the client-selection stage, one function per figure type.

Each function takes the catalogue (and a Cluster where relevant), returns the
matplotlib Figure, and never loads or filters data itself. Consolidations with
respect to the original scripts:

  * `heatmap_alt_inc`        replaces the two nearly identical heatmap blocks of
                             plot_clients.py (count-coloured and mass-coloured)
  * `inclination_histogram`  replaces three implementations spread over
                             cluster_refinement.py and recup_code_clientdoc.py
  * `mass_class_panel`       replaces mass_distribution.py in full, which was a
                             duplicate of one cell of cluster_refinement.py
  * `cluster_scatter`        replaces the two shape/mass/cross-section scatters,
                             with the ADR target rings as an option

All functions accept `save_as=` to write the figure to config.OUTPUT_DIR
instead of, or in addition to, showing it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

import config as cfg
from catalogue import select_cluster

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
STYLE = {
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
}


# Arc and marker colours of the RAAN sub-clusters, used by raan_wheel.
GROUP_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]


def apply_style() -> None:
    """Apply the thesis figure style. Call once at the start of a run."""
    plt.rcParams.update(STYLE)


def _finish(fig, save_as=None, show=True):
    """Save and/or show a figure, then return it."""
    if save_as is not None:
        cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(cfg.OUTPUT_DIR / save_as)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# 1. Population overview
# ---------------------------------------------------------------------------
def band_distribution(cat, obj_type, axis="ALT_BAND", save_as=None, show=True):
    """Object count per altitude band or per inclination band."""
    labels = (cfg.ALT_BAND_LABELS if axis == "ALT_BAND"
              else cfg.INC_BAND_LABELS)
    xlabel = ("Mean altitude band (km)" if axis == "ALT_BAND"
              else "Inclination band (deg)")
    colour = "steelblue" if axis == "ALT_BAND" else "darkorange"

    sub = cat[cat["OBJ_TYPE"] == obj_type]
    counts = (sub.dropna(subset=[axis]).groupby(axis, observed=False).size()
                 .reindex(labels, fill_value=0))

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(counts.index, counts.values, color=colour)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of objects")
    ax.set_title(f"{obj_type} - distribution by {xlabel.lower()}")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _finish(fig, save_as, show)


def heatmap_alt_inc(cat, obj_type, colour_by="count", save_as=None, show=True):
    """Altitude x inclination heatmap, 100 km x 5 deg bins.

    Parameters
    ----------
    colour_by : 'count' colours the cells by the number of objects,
                'mass' colours them by the total mass in tonnes.
                In both cases the number printed in each cell is the object
                count, and empty cells are left white.

    This single function covers both heatmaps of the original plot_clients.py.
    """
    sub = cat[cat["OBJ_TYPE"] == obj_type].dropna(subset=["ALT_BAND",
                                                          "INC_BAND"])
    if colour_by == "mass":
        sub = sub.dropna(subset=["MASS_KG"])

    grouped = sub.groupby(["ALT_BAND", "INC_BAND"], observed=False)
    counts = (grouped.size().unstack(fill_value=0)
              .reindex(index=cfg.ALT_BAND_LABELS,
                       columns=cfg.INC_BAND_LABELS, fill_value=0))

    if colour_by == "mass":
        colour_data = (grouped["MASS_KG"].sum().unstack(fill_value=0)
                       .reindex(index=cfg.ALT_BAND_LABELS,
                                columns=cfg.INC_BAND_LABELS,
                                fill_value=0) / 1000.0)
        cbar_label = "Total mass (tonnes, log scale)"
        vmin = 0.1
    else:
        colour_data = counts
        cbar_label = "Number of objects (log scale)"
        vmin = 1

    values = colour_data.values.astype(float)
    values[values == 0] = np.nan          # empty bins stay white
    count_values = counts.values.astype(int)

    fig, ax = plt.subplots(figsize=(16, 10))
    im = ax.imshow(values, aspect="auto", origin="lower",
                   cmap=cfg.DENSITY_CMAP, interpolation="nearest",
                   norm=LogNorm(vmin=vmin, vmax=np.nanmax(values)))

    for i in range(count_values.shape[0]):
        for j in range(count_values.shape[1]):
            n = count_values[i, j]
            if n > 0:
                shade = im.norm(values[i, j])
                ax.text(j, i, f"{n}", ha="center", va="center",
                        fontsize=6, fontweight="bold",
                        color="white" if shade > 0.6 else "black")

    ax.set_xticks(range(len(cfg.INC_BAND_LABELS)))
    ax.set_xticklabels(cfg.INC_BAND_LABELS, rotation=90, fontsize=8)
    ax.set_yticks(range(len(cfg.ALT_BAND_LABELS)))
    ax.set_yticklabels(cfg.ALT_BAND_LABELS, fontsize=8)
    ax.set_xlabel("Inclination (deg)")
    ax.set_ylabel("Mean altitude (km)")
    ax.set_title(f"{obj_type} - object count per bin\n"
                 f"colour = {'total mass' if colour_by == 'mass' else 'count'}"
                 f", bin = 5 deg x 100 km")
    fig.colorbar(im, ax=ax, shrink=0.7, label=cbar_label)
    fig.tight_layout()
    return _finish(fig, save_as, show)


# ---------------------------------------------------------------------------
# 2. Cluster refinement
# ---------------------------------------------------------------------------
_NICE_STEPS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]


def _nice_step(raw):
    """Round a step up to the next value in _NICE_STEPS."""
    for step in _NICE_STEPS:
        if raw <= step:
            return step
    return _NICE_STEPS[-1]


def inclination_histogram(cat, cluster, bin_width=None, tick_step=None,
                          weight_by_mass=True, save_as=None, show=True):
    """Inclination distribution inside a cluster.

    Bar height is always the object count. When `weight_by_mass` is True the
    bars are coloured by the total mass they contain, which is how the narrow
    inclination bands of the refined clusters were identified.

    `bin_width` and `tick_step` default to values derived from the width of the
    cluster window, so the same call works on a wide cluster spanning 5 deg and
    on a refined one spanning 0.1 deg. A fixed 0.1 deg bin would leave a
    refined cluster with a single bar.
    """
    subset = select_cluster(cat, cluster)
    if weight_by_mass:
        subset = subset.dropna(subset=["MASS_KG"])
    if subset.empty:
        print(f"{cluster.label}: no data")
        return None

    span = cluster.inc_hi - cluster.inc_lo
    if bin_width is None:
        # span/50 gives 0.1 deg on a 5 deg wide cluster, which is the width the
        # narrow inclination spikes show up at, and scales down sensibly on a
        # refined cluster.
        bin_width = _nice_step(span / 50)
    if tick_step is None:
        tick_step = _nice_step(span / 12)

    bins = np.arange(cluster.inc_lo, cluster.inc_hi + bin_width, bin_width)
    counts, edges = np.histogram(subset["INCLINATION"], bins=bins)

    fig, ax = plt.subplots(figsize=(12, 6))
    if weight_by_mass:
        masses, _ = np.histogram(subset["INCLINATION"], bins=bins,
                                 weights=subset["MASS_KG"])
        norm = Normalize(vmin=0, vmax=masses.max() if masses.max() else 1)
        cmap = plt.get_cmap(cfg.MASS_CMAP)
        ax.bar(edges[:-1], counts, width=bin_width, align="edge",
               color=cmap(norm(masses)), edgecolor="white", linewidth=0.2)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, pad=0.02).set_label("Total mass in bin (kg)")
        subtitle = (f"object count per {bin_width} deg bin, "
                    f"colour = total mass in bin  (N = {len(subset)}, "
                    f"{subset['MASS_KG'].sum() / 1000:.1f} t)")
    else:
        ax.bar(edges[:-1], counts, width=bin_width, align="edge",
               color="steelblue", edgecolor="white", linewidth=0.2)
        subtitle = f"N = {len(subset)}"

    ax.set_xticks(np.arange(cluster.inc_lo,
                            cluster.inc_hi + tick_step / 2, tick_step))
    ax.set_xlabel("Inclination (deg)")
    ax.set_ylabel("Object count")
    ax.set_title(f"{cluster.label}\n{subtitle}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _finish(fig, save_as, show)


def inclination_overlay(cat, clusters, bin_width=0.1, tick_step=0.2,
                        save_as=None, show=True):
    """Normalised inclination distributions of several clusters, overlaid."""
    fig, ax = plt.subplots(figsize=(9, 5))
    inc_lo = min(c.inc_lo for c in clusters)
    inc_hi = max(c.inc_hi for c in clusters)
    bins = np.arange(inc_lo, inc_hi + bin_width, bin_width)

    for c in clusters:
        subset = select_cluster(cat, c)
        ax.hist(subset["INCLINATION"], bins=bins, alpha=0.45,
                label=f"{c.label} (N={len(subset)})", density=True)

    ax.set_xticks(np.arange(inc_lo, inc_hi, tick_step))
    ax.set_xlabel("Inclination (deg)")
    ax.set_ylabel("Density")
    ax.set_title("Inclination distribution by cluster")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _finish(fig, save_as, show)


def altitude_histogram(cat, cluster, bin_width=20, save_as=None, show=True):
    """Mean-altitude distribution inside a cluster, in bins of `bin_width` km."""
    subset = select_cluster(cat, cluster)
    if subset.empty:
        print(f"{cluster.label}: no data")
        return None

    bins = np.arange(cluster.alt_lo, cluster.alt_hi + bin_width, bin_width)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(subset["ALT_MEAN"], bins=bins, alpha=0.75, color="steelblue",
            edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Mean altitude (km)")
    ax.set_ylabel("Object count")
    ax.set_title(f"{cluster.label}\nN = {len(subset)}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _finish(fig, save_as, show)


def mass_class_panel(cat, clusters, n_cols=3, save_as=None, show=True):
    """Stacked mass-class bar chart for each cluster, on one panel figure.

    The legend goes into a spare cell of the grid when there is one, and below
    the panel otherwise, so no empty row is ever left at the bottom.
    """
    n = len(clusters)
    n_rows = int(np.ceil(n / n_cols))
    spare = n_rows * n_cols - n
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, n_rows * 4.5))
    axes = np.atleast_1d(axes).flatten()

    for ax, cluster in zip(axes, clusters):
        subset = select_cluster(cat, cluster)
        subset = subset[subset["MASS_CLASS"].notna()]
        counts = (subset["MASS_CLASS"].value_counts()
                  .reindex(cfg.MASS_LABELS, fill_value=0))

        bottom = 0
        for label, colour in zip(cfg.MASS_LABELS, cfg.MASS_COLORS):
            value = counts[label]
            ax.bar(cluster.label, value, bottom=bottom, color=colour,
                   edgecolor="white", linewidth=0.4)
            if value > 0:
                ax.text(0, bottom + value / 2, str(value),
                        ha="center", va="center", fontsize=9,
                        fontweight="bold",
                        color="white" if colour in "bgrk" else "black")
            bottom += value

        ax.set_title(cluster.label, fontsize=10, fontweight="bold", pad=6)
        ax.set_ylabel("Object count", fontsize=9)
        ax.set_xticks([])
        ax.grid(axis="y", alpha=0.3)
        ax.text(0.98, 0.97,
                f"n = {counts.sum()}\ntotal: "
                f"{subset['MASS_KG'].sum() / 1000:.1f} t",
                transform=ax.transAxes, fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          edgecolor="#AAAAAA", alpha=0.9))

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=l)
               for l, c in zip(cfg.MASS_LABELS, cfg.MASS_COLORS)]
    if spare:
        axes[n].axis("off")
        axes[n].legend(handles=handles, title="Mass range", loc="center",
                       fontsize=11, title_fontsize=12, framealpha=0.9)
        for ax in axes[n + 1:]:
            ax.axis("off")
        fig.suptitle("Mass distribution within each cluster",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
    else:
        fig.suptitle("Mass distribution within each cluster",
                     fontsize=14, fontweight="bold")
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        fig.legend(handles=handles, title="Mass range", loc="lower center",
                   ncol=len(cfg.MASS_LABELS), fontsize=10, title_fontsize=11,
                   framealpha=0.9, bbox_to_anchor=(0.5, 0.005))
    return _finish(fig, save_as, show)


# ---------------------------------------------------------------------------
# 3. Cluster content: shape, mass, size, and the selected targets
# ---------------------------------------------------------------------------
def cluster_scatter(cat, cluster, targets=None, label_offsets=None,
                    size_scale=250, size_refs=(5, 10, 25, 50), ring_pad=8,
                    size_legend_loc="lower right", max_labels=25,
                    save_as=None, show=True):
    """Altitude vs inclination inside a cluster.

    Marker shape encodes the object family, colour encodes mass, marker area
    encodes cross-section (square-root scaled so that one very large object does
    not flatten the rest).

    Parameters
    ----------
    targets : DataFrame with an OBJECT_ID column, typically from
              `catalogue.load_targets`. Matching objects get a black ring and a
              name label. Pass None to omit.
    label_offsets : {COSPAR ID or name: (dx, dy) in points} to move labels that
              would otherwise overlap a legend or another label.
    max_labels : names are written only when the cluster holds at most this many
              targets, and only when they are not all the same name. Beyond
              that the labels overlap into an unreadable block: the SL-8
              clusters hold 50 and 139 targets that are all called "SL-8 R/B",
              so the rings alone carry the information. Raise it to force
              labels back on.
    """
    label_offsets = label_offsets or {}
    subset = select_cluster(cat, cluster)
    plot_data = subset[["OBJECT_ID", "SATNAME", "INCLINATION", "ALT_MEAN",
                        "MASS_KG", "CROSS_SECTION", "SHAPE_SIMPLE"]].dropna(
        subset=["INCLINATION", "ALT_MEAN", "MASS_KG", "CROSS_SECTION"])
    if plot_data.empty:
        print(f"{cluster.label}: no data")
        return None

    cs_max = plot_data["CROSS_SECTION"].max()

    def scale(cs):
        return np.sqrt(cs / cs_max) * size_scale

    norm = Normalize(vmin=plot_data["MASS_KG"].min(),
                     vmax=plot_data["MASS_KG"].max())
    fig, ax = plt.subplots(figsize=(12, 6))

    shape_handles = []
    for shape, group in plot_data.groupby("SHAPE_SIMPLE"):
        marker = cfg.SHAPE_MARKERS.get(shape, "s")
        ax.scatter(group["INCLINATION"], group["ALT_MEAN"],
                   s=scale(group["CROSS_SECTION"]), c=group["MASS_KG"],
                   cmap=cfg.MASS_CMAP, norm=norm, marker=marker,
                   alpha=0.85, edgecolors="none")
        shape_handles.append(Line2D([], [], linestyle="none", marker=marker,
                                    color="grey", markersize=6, label=shape))

    target_handle = []
    if targets is not None and len(targets):
        hit = plot_data[plot_data["OBJECT_ID"].isin(targets["OBJECT_ID"])]
        missing = targets[~targets["OBJECT_ID"].isin(hit["OBJECT_ID"])]
        print(f"{cluster.label}: {len(targets)} targets listed, "
              f"{len(hit)} inside the window")
        for _, row in missing.iterrows():
            print(f"    outside: {row['OBJECT_ID']}  {row.get('SATNAME', '')}")

        if len(hit):
            ring_s = (np.sqrt(scale(hit["CROSS_SECTION"])) + ring_pad) ** 2
            for colour, lw, z in [("white", 3.0, 4), ("black", 1.4, 5)]:
                ax.scatter(hit["INCLINATION"], hit["ALT_MEAN"], s=ring_s,
                           facecolors="none", edgecolors=colour,
                           linewidths=lw, zorder=z)

            distinct = hit["SATNAME"].nunique()
            write_labels = len(hit) <= max_labels and distinct > 1
            if not write_labels:
                reason = ("all targets share the same name"
                          if distinct <= 1 else
                          f"more than {max_labels} targets")
                print(f"    names not written: {reason}")

            if write_labels:
                seen = {}
                for _, r in hit.sort_values("SATNAME").iterrows():
                    key = (round(r["INCLINATION"], 3), round(r["ALT_MEAN"], 1))
                    k = seen.get(key, 0)
                    seen[key] = k + 1
                    off = label_offsets.get(r["OBJECT_ID"],
                                            label_offsets.get(r["SATNAME"]))
                    if off is None:
                        off = (0, 9 + np.sqrt(ring_s.max()) / 2 - 10 * k)
                    ha = "center" if off[0] == 0 else ("left" if off[0] > 0
                                                       else "right")
                    ax.annotate(r["SATNAME"],
                                (r["INCLINATION"], r["ALT_MEAN"]),
                                textcoords="offset points", xytext=off, ha=ha,
                                va="center", fontsize=7, color="black",
                                zorder=6,
                                path_effects=[pe.withStroke(linewidth=2.5,
                                                            foreground="white")])
            label = (f"ADR target ({len(hit)})" if not write_labels
                     else "ADR target")
            target_handle = [Line2D([], [], linestyle="none", marker="o",
                                    markerfacecolor="none",
                                    markeredgecolor="black", markersize=9,
                                    markeredgewidth=1.4, label=label)]

    sm = ScalarMappable(cmap=cfg.MASS_CMAP, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.02).set_label("Mass (kg)", fontsize=9)

    leg1 = ax.legend(handles=shape_handles, title="Shape", loc="upper left",
                     fontsize=8, title_fontsize=9, framealpha=0.85)
    ax.add_artist(leg1)

    size_handles = [Line2D([], [], linestyle="none", marker="o", color="grey",
                           markersize=np.sqrt(scale(a)), label=f"{a} m2")
                    for a in size_refs if a <= cs_max]
    if size_handles:
        leg2 = ax.legend(handles=size_handles, title="Cross-section",
                         loc=size_legend_loc, fontsize=8, title_fontsize=9,
                         framealpha=0.85, labelspacing=0.6, borderpad=0.6)
        ax.add_artist(leg2)
    if target_handle:
        ax.legend(handles=target_handle, loc="upper center", fontsize=8,
                  framealpha=0.85)

    ax.set_xlabel("Inclination (deg)")
    ax.set_ylabel("Mean altitude (km)")
    ax.set_title(f"{cluster.label}\n"
                 f"shape (marker) - mass (colour) - cross-section (size)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _finish(fig, save_as, show)


# ---------------------------------------------------------------------------
# 4. Tables
# ---------------------------------------------------------------------------
TABLE_HEADERS = {
    "OBJECT_ID": "COSPAR ID", "SATNAME": "Name", "COUNTRY": "Country",
    "OBJ_TYPE": "Type", "SHAPE": "Shape", "RCS_SIZE": "RCS",
    "WIDTH_M": "Width (m)", "HEIGHT_M": "Height (m)",
    "DIAMETER_M": "Diameter (m)", "SPAN_M": "Span (m)",
    "INCLINATION": "Inc (deg)", "ALT_MEAN": "Alt (km)",
    "MASS_KG": "Mass (kg)", "LAUNCH_YEAR": "Year",
}


def object_table(df, title="", rows_per_page=30, save_as=None, show=True):
    """Render a query result as one or more paginated matplotlib tables.

    Only useful to paste a short list into a slide. For anything longer, export
    the DataFrame with `df.to_excel(...)` and typeset it in LaTeX instead.
    """
    if df.empty:
        print("No matching objects.")
        return []

    fmt = df.copy()
    for c in ["INCLINATION", "ALT_MEAN", "WIDTH_M", "HEIGHT_M", "DIAMETER_M",
              "SPAN_M"]:
        if c in fmt:
            fmt[c] = fmt[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    if "MASS_KG" in fmt:
        fmt["MASS_KG"] = fmt["MASS_KG"].map(
            lambda x: f"{x:.0f}" if pd.notna(x) else "-")
    if "LAUNCH_YEAR" in fmt:
        fmt["LAUNCH_YEAR"] = fmt["LAUNCH_YEAR"].map(
            lambda x: f"{int(x)}" if pd.notna(x) else "-")
    fmt = fmt.fillna("-").astype(str)

    headers = [TABLE_HEADERS.get(c, c) for c in fmt.columns]
    rows = fmt.values.tolist()
    n_pages = max(1, (len(rows) + rows_per_page - 1) // rows_per_page)
    figs = []

    total_mass = pd.to_numeric(df.get("MASS_KG"), errors="coerce").sum()
    for page in range(n_pages):
        chunk = rows[page * rows_per_page:(page + 1) * rows_per_page]
        fig, ax = plt.subplots(figsize=(20, max(4, 0.4 * len(chunk) + 2.5)))
        ax.axis("off")
        page_label = f" - page {page + 1}/{n_pages}" if n_pages > 1 else ""
        ax.set_title(f"{title}{page_label}\n({len(df)} objects, "
                     f"{total_mass / 1000:.1f} t total)",
                     fontsize=13, fontweight="bold", loc="left")

        table = ax.table(cellText=chunk, colLabels=headers, loc="center",
                         cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.3)
        for j in range(len(headers)):
            table[0, j].set_facecolor("#1a3e5c")
            table[0, j].set_text_props(color="white", fontweight="bold",
                                       fontsize=9)
        for i in range(1, len(chunk) + 1):
            bg = "#f0f4f8" if i % 2 == 0 else "white"
            for j in range(len(headers)):
                table[i, j].set_facecolor(bg)
        fig.tight_layout()
        name = None if save_as is None else save_as.replace(
            ".png", f"_p{page + 1}.png")
        figs.append(_finish(fig, name, show))
    return figs


# ---------------------------------------------------------------------------
# 5. RAAN-based refinement
# ---------------------------------------------------------------------------
def raan_wheel(cat, cluster, targets, groups=None, label_col="SATNAME",
               min_sep=4.0, r_point=1.0, r_label=1.10, r_arc=0.88,
               save_as=None, show=True):
    """Orientation of the orbital planes of the candidate targets.

    The RAAN is an angle, so it is drawn as one: each target is a point on a
    circle at its own RAAN, seen from the north pole, with the vernal equinox
    direction to the right. Targets that sit close together on the circle share
    a plane orientation. Each sub-cluster is underlined by an arc labelled with
    its total spread, and the two widest empty sectors are where the groups are
    cut.

    Marker area encodes the mass, as in the other target figures.

    Parameters
    ----------
    groups   : {group name: [COSPAR ID, ...]}, from config.SUBCLUSTERS.
    min_sep  : two targets closer than this many degrees have their labels
               pushed outwards one after the other, so that near-coplanar
               objects such as Cosmo-SkyMed 2 and 4 stay readable.
    """
    groups = groups or {}
    subset = cat[["OBJECT_ID", "ALT_MEAN", "INCLINATION"]]
    df = (targets.merge(subset, on="OBJECT_ID", how="left")
                 .dropna(subset=["RAAN_DEG"])
                 .sort_values("RAAN_DEG")
                 .reset_index(drop=True))
    if df.empty:
        print(f"{cluster.label}: no RAAN data")
        return None

    member_of = {oid: name for name, ids in groups.items() for oid in ids}
    df["GROUP"] = df["OBJECT_ID"].map(member_of).fillna("Isolated")

    m_max = df["MASS_KG"].max()
    theta = np.deg2rad(df["RAAN_DEG"].to_numpy())

    fig, ax = plt.subplots(figsize=(9, 9),
                           subplot_kw={"projection": "polar"})
    colours = {name: GROUP_COLORS[i % len(GROUP_COLORS)]
               for i, name in enumerate(groups)}

    # Sub-cluster arcs, inside the ring of points
    for name, ids in groups.items():
        g = df[df["OBJECT_ID"].isin(ids)]
        if g.empty:
            continue
        lo, hi = g["RAAN_DEG"].min(), g["RAAN_DEG"].max()
        span = np.deg2rad(np.linspace(lo, hi, 60))
        ax.plot(span, np.full_like(span, r_arc), color=colours[name], lw=3,
                solid_capstyle="round", zorder=2)
        mid = np.deg2rad((lo + hi) / 2)
        ax.text(mid, r_arc - 0.17, f"$\\Delta\\Omega$ = {hi - lo:.0f}°",
                ha="center", va="center", fontsize=9, fontweight="bold",
                color=colours[name], zorder=6,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    handles = []
    for name in list(groups) + ["Isolated"]:
        g = df[df["GROUP"] == name]
        if g.empty:
            continue
        colour = colours.get(name, "0.55")
        ax.scatter(np.deg2rad(g["RAAN_DEG"]),
                   np.full(len(g), r_point),
                   s=np.sqrt(g["MASS_KG"] / m_max) * 300,
                   color=colour, edgecolors="white", linewidths=0.8,
                   alpha=0.9, zorder=4)
        handles.append(Line2D([], [], linestyle="none", marker="o",
                              color=colour, markersize=9, label=name))

    # Labels, laid end to end along the radius when two targets nearly share
    # a plane, so that a near-coplanar pair does not print on itself
    last_deg, end, r_max = -999.0, r_label, r_label
    for _, r in df.iterrows():
        deg, text = r["RAAN_DEG"], str(r[label_col])
        start = end + 0.05 if deg - last_deg < min_sep else r_label
        last_deg, end = deg, start + 0.048 * len(text)
        r_max = max(r_max, end)
        rot = deg if -90 < ((deg + 180) % 360) - 180 < 90 else deg + 180
        ha = "left" if rot == deg else "right"
        ax.text(np.deg2rad(deg), start, text,
                rotation=rot, rotation_mode="anchor", ha=ha, va="center",
                fontsize=8, zorder=6,
                path_effects=[pe.withStroke(linewidth=2.5,
                                            foreground="white")])

    ax.set_ylim(0, r_max + 0.10)
    ax.set_yticks([])
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 30)))
    ax.set_xticklabels([f"{d}°" for d in range(0, 360, 30)], fontsize=9)
    ax.set_rlabel_position(0)
    ax.grid(True, alpha=0.25)
    ax.spines["polar"].set_visible(False)
    ax.annotate("$\\gamma$", xy=(0, 0), xytext=(0, r_arc - 0.25),
                fontsize=13, ha="center", va="center")
    ax.plot([0, 0], [0, r_arc - 0.32], color="0.3", lw=1.2, zorder=3)
    ax.legend(handles=handles, title="Sub-cluster", loc="center",
              fontsize=8, title_fontsize=9, framealpha=0.9)
    ax.set_title(f"{cluster.label}\nOrientation of the orbital planes, "
                 f"RAAN at the TLE epoch of May 2026", pad=24)
    fig.tight_layout()
    return _finish(fig, save_as, show)


def raan_gaps(targets, n=None):
    """Sorted RAAN list with the gap to the next object, wrapping at 360 deg.

    Printed rather than plotted: it is the table the sub-cluster boundaries are
    chosen from, the natural cuts being the largest gaps.
    """
    t = targets.dropna(subset=["RAAN_DEG"]).sort_values("RAAN_DEG").copy()
    raan = t["RAAN_DEG"].to_numpy()
    t["GAP_TO_NEXT"] = np.append(np.diff(raan), 360.0 - raan[-1] + raan[0])
    cols = [c for c in ("OBJECT_ID", "SATNAME", "RAAN_DEG", "GAP_TO_NEXT")
            if c in t.columns]
    out = t[cols].reset_index(drop=True)
    return out.nlargest(n, "GAP_TO_NEXT") if n else out
