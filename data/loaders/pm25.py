"""Beijing PM2.5 (STMVL) dataset loader -- the project's analog of the paper's "Air" dataset.

Raw layout: datasets/PM25/STMVL-Release (1).zip contains
Code/STMVL/SampleData/pm25_ground.txt, a wide CSV: `datetime` + 36 station columns
(001001..001036), hourly readings from 2014-05-01 to 2015-04-30. A handful of cells are
still empty even in the "ground" file, so we forward-fill then linearly interpolate any
remaining gaps (reproduction_plan.md Sec. 2 "PM25" pipeline notes).
"""
from __future__ import annotations

import io
import zipfile

import pandas as pd

GROUND_MEMBER = "Code/STMVL/SampleData/pm25_ground.txt"


def load_pm25_raw(zip_path: str) -> pd.DataFrame:
    """Load the hourly [n_hours, n_stations] PM2.5 table, indexed by datetime, sorted."""
    with zipfile.ZipFile(zip_path) as z:
        raw_bytes = z.read(GROUND_MEMBER)
    df = pd.read_csv(io.BytesIO(raw_bytes), parse_dates=["datetime"])
    df = df.set_index("datetime").sort_index()
    return df


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill, then linearly interpolate any still-missing cells (handles leading NaNs)."""
    df = df.ffill().interpolate(method="linear", limit_direction="both")
    assert not df.isna().any().any(), "unfilled NaNs remain after ffill+interpolate"
    return df
