"""
external_data.py
================
Retrieval of the two external data sources: ESA DISCOSweb (physical properties)
and Space-Track (orbital elements, used here for the RAAN of the cluster
objects).

Merges `discos_getdata.py` and the download half of `get_RAAN.py`. Both were
one-shot scripts; they are functions here so they can be called from a runner
and, more importantly, so that the credentials are no longer written in the
source.

CREDENTIALS
-----------
Never hard-code a token. Set them in the environment before running:

    export DISCOS_TOKEN="..."
    export SPACETRACK_USER="you@example.com"
    export SPACETRACK_PASS="..."

or put them in a `.env` file at the repository root (which .gitignore excludes)
and load it with python-dotenv.
"""

from __future__ import annotations

import getpass
import os
import time
from pathlib import Path

import pandas as pd
import requests

DISCOS_URL = "https://discosweb.esoc.esa.int/api/objects"
SPACETRACK_URL = "https://www.space-track.org"


# ---------------------------------------------------------------------------
# DISCOSweb
# ---------------------------------------------------------------------------
DISCOS_FIELDS = {
    "satno": "satno", "name": "name", "objectClass": "objectClass",
    "mass_kg": "mass", "shape": "shape", "width_m": "width",
    "height_m": "height", "depth_m": "depth", "diameter_m": "diameter",
    "span_m": "span", "xSectAvg_m2": "xSectAvg", "xSectMin_m2": "xSectMin",
    "xSectMax_m2": "xSectMax",
}

_COSPAR_KEYS = ("cosparId", "cosparID", "cospar_id",
                "internationalDesignator", "international_designator")


def fetch_discos(object_class="Payload", token=None, page_size=100,
                 verbose=True) -> pd.DataFrame:
    """Download every DISCOS object of a given class.

    Parameters
    ----------
    object_class : 'Payload', 'Rocket Body', 'Debris', ...
    token : DISCOS API token. Defaults to the DISCOS_TOKEN environment
            variable.

    Handles the API rate limit by honouring the Retry-After header.
    """
    token = token or os.environ.get("DISCOS_TOKEN")
    if not token:
        raise RuntimeError("No DISCOS token. Set the DISCOS_TOKEN "
                           "environment variable.")

    headers = {"Authorization": f"Bearer {token}",
               "DiscosWeb-Api-Version": "2"}
    params = {"filter": f"eq(objectClass,'{object_class}')",
              "page[size]": page_size}

    records, page = [], 1
    while True:
        params["page[number]"] = page
        response = requests.get(DISCOS_URL, headers=headers, params=params,
                                timeout=60)

        if response.status_code == 429:
            wait = int(response.headers.get("Retry-After", 5))
            if verbose:
                print(f"  rate limit, waiting {wait} s")
            time.sleep(wait)
            continue
        response.raise_for_status()

        payload = response.json()
        objects = payload.get("data", [])
        if not objects:
            break

        for obj in objects:
            attr = obj.get("attributes", {})
            row = {"discos_id": obj.get("id")}
            row.update({out: attr.get(src)
                        for out, src in DISCOS_FIELDS.items()})
            row["cosparId"] = next(
                (attr[k] for k in _COSPAR_KEYS if attr.get(k)), None)
            records.append(row)

        if verbose:
            print(f"  page {page}: {len(records)} objects so far")
        if payload["links"].get("next") is None:
            break
        page += 1
        time.sleep(1)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Space-Track
# ---------------------------------------------------------------------------
def _parse_tle_text(text: str) -> dict:
    """Extract RAAN and epoch from a two-line-element listing.

    RAAN is columns 18-25 of line 2 and the epoch is columns 19-32 of line 1,
    per the TLE format definition.
    """
    found = {}
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    i = 0
    while i < len(lines) - 1:
        line1, line2 = lines[i], lines[i + 1]
        if line1.startswith("1 ") and line2.startswith("2 "):
            try:
                found[int(line2[2:7])] = {
                    "RAAN_DEG": float(line2[17:25]),
                    "TLE_EPOCH": line1[18:32].strip(),
                    "TLE_LINE1": line1,
                    "TLE_LINE2": line2,
                }
                i += 2
                continue
            except (ValueError, IndexError):
                pass
        i += 1
    return found


def fetch_raan(norad_ids, username=None, password=None,
               cache_file="tle_cache.csv", batch_size=50, sleep_s=3.0,
               verbose=True) -> pd.DataFrame:
    """Fetch the latest RAAN of each NORAD ID from Space-Track.

    Uses the GP class, which returns a single latest element set per object,
    with comma-separated IDs so that one request covers `batch_size` objects.
    The Space-Track limit is 30 requests/min; `sleep_s=3` stays under it.

    Results are appended to `cache_file` after every batch, so an interrupted
    run resumes where it stopped. Delete the cache to start over.

    Returns a DataFrame indexed by NORAD_ID with RAAN_DEG, TLE_EPOCH and the
    two TLE lines.
    """
    cache_path = Path(cache_file)
    records = {}
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if not cached.empty and "NORAD_ID" in cached:
            records = {int(r["NORAD_ID"]): {k: r[k] for k in
                       ("RAAN_DEG", "TLE_EPOCH", "TLE_LINE1", "TLE_LINE2")}
                       for _, r in cached.iterrows()}
            if verbose:
                print(f"Resumed from cache: {len(records)} TLEs already held")

    remaining = [int(n) for n in norad_ids if int(n) not in records]
    if not remaining:
        return pd.DataFrame([{"NORAD_ID": k, **v} for k, v in records.items()])

    username = username or os.environ.get("SPACETRACK_USER") or input(
        "Space-Track username: ").strip()
    password = (password or os.environ.get("SPACETRACK_PASS")
                or getpass.getpass("Space-Track password: "))

    session = requests.Session()
    session.post(f"{SPACETRACK_URL}/ajaxauth/login",
                 data={"identity": username,
                       "password": password}).raise_for_status()

    batches = [remaining[i:i + batch_size]
               for i in range(0, len(remaining), batch_size)]
    if verbose:
        print(f"Fetching {len(remaining)} objects in {len(batches)} batches")

    for k, batch in enumerate(batches, start=1):
        url = (f"{SPACETRACK_URL}/basicspacedata/query/class/gp"
               f"/NORAD_CAT_ID/{','.join(map(str, batch))}"
               f"/decay_date/null-val/orderby/NORAD_CAT_ID/format/tle")
        for attempt in range(5):
            try:
                response = session.get(url, timeout=30)
                if response.status_code == 429:
                    time.sleep(90)
                    continue
                response.raise_for_status()
                break
            except requests.RequestException as exc:
                print(f"  attempt {attempt + 1} failed: {exc}")
                time.sleep(15)
        else:
            continue

        records.update(_parse_tle_text(response.text))
        pd.DataFrame([{"NORAD_ID": i, **v} for i, v in records.items()]).to_csv(
            cache_path, index=False)
        if verbose:
            print(f"  batch {k}/{len(batches)}: {len(records)} TLEs total")
        time.sleep(sleep_s)

    missing = [n for n in norad_ids if int(n) not in records]
    if missing and verbose:
        print(f"No TLE for {len(missing)} objects (not in the current "
              f"catalogue): {missing[:10]}")

    return pd.DataFrame([{"NORAD_ID": k, **v} for k, v in records.items()])
