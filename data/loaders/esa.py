"""ESA Anomaly Dataset (Mission1) loader -- the project's standard-protocol test dataset.

Raw layout: datasets/ESA Anomaly Dataset/ESA-Mission1.zip contains
ESA-Mission1/channels.csv (per-channel metadata) and ESA-Mission1/channels/channel_N.zip,
each an inner zip holding a single pickled pandas Series (one column named channel_N,
irregularly-sampled DatetimeIndex).

Channel selection: all 76 channels in Mission1 are Categorical=NO (continuous telemetry).
We use channel_12..channel_27 (16 channels, subsystem_6/physical_unit_3, all Target=YES,
i.e. densely anomaly-labeled): confirmed co-sampled -- identical row count (14,258,506) and
identical [2000-01-01, 2013-12-31] time span, at ~0.375s median native interval. There is no
official train/test split file in the archive, so (matching the PM25 loader's approach) we
resample to a common regular hourly grid and time-split chronologically: the standard
protocol referenced in reproduction_plan.md ("ESA 自身 train/test 划分") is defined here as
the first (1 - test_ratio) fraction of the 14-year span as train, the remainder as test.
"""
from __future__ import annotations

import io
import zipfile

import pandas as pd

MISSION = "ESA-Mission1"
CHANNELS = [f"channel_{i}" for i in range(12, 28)]
RESAMPLE_FREQ = "1h"


def load_esa_channel(zip_path: str, channel: str) -> pd.Series:
    """Load one channel's raw irregularly-sampled series from the outer mission zip."""
    with zipfile.ZipFile(zip_path) as z:
        inner_bytes = z.read(f"{MISSION}/channels/{channel}.zip")
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as z2:
        raw = z2.read(channel)
    series = pd.read_pickle(io.BytesIO(raw))
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    return series.rename(channel).sort_index()


def build_hourly_frame(zip_path: str, channels: list[str] = CHANNELS) -> pd.DataFrame:
    """Load each channel, resample to a common regular hourly grid (mean), and align into
    one wide [n_hours, len(channels)] DataFrame via an outer join on the datetime index."""
    resampled = []
    for ch in channels:
        series = load_esa_channel(zip_path, ch)
        resampled.append(series.resample(RESAMPLE_FREQ).mean())
    df = pd.concat(resampled, axis=1).sort_index()
    return df


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill, then linearly interpolate any still-missing hourly cells."""
    df = df.ffill().interpolate(method="linear", limit_direction="both")
    assert not df.isna().any().any(), "unfilled NaNs remain after ffill+interpolate"
    return df
