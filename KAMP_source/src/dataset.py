"""Original fixed clip split; duplicate timestamps removed before segmentation."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

SENSORS = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]

def prepare(data_dir):
    frames, provenance = [], []
    for filename, source, state in [("press_data_normal.csv", "normal", 0), ("outlier_data.csv", "fault", 1)]:
        path = data_dir / filename
        if not path.exists() and (data_dir / 'raw' / filename).exists():
            path = data_dir / 'raw' / filename
        d = pd.read_csv(path)
        d["timestamp"] = pd.to_datetime(d.TimeStamp)
        if not d.Equipment_state.eq(state).all() or not d.timestamp.is_monotonic_increasing:
            raise ValueError("Unexpected state or timestamp order")
        if d[SENSORS].isna().any().any():
            raise ValueError("Missing sensor values")
        duplicate_times = d[d.timestamp.duplicated(keep=False)]
        for _, g in duplicate_times.groupby("timestamp"):
            if len(g[SENSORS+['Equipment_state']].drop_duplicates()) != 1:
                raise ValueError("Conflicting samples at the same timestamp")
        raw_n = len(d)
        d = d.drop_duplicates("timestamp").reset_index(drop=True)
        numbers = d.timestamp.diff().dt.total_seconds().gt(.5).cumsum()+1
        d["clip_id"] = numbers.map(lambda i: f"{source}_{i:04d}")
        d["sample_index"] = d.groupby("clip_id", sort=False).cumcount()
        d["elapsed_seconds"] = (d.timestamp-d.groupby("clip_id").timestamp.transform("first")).dt.total_seconds()
        d["source"] = source
        d["y_true"] = d.Equipment_state.astype(int)
        d["source_row"] = d.iloc[:, 0]
        frames.append(d[["source", "source_row", "clip_id", "sample_index", "timestamp",
                         "elapsed_seconds", *SENSORS, "y_true"]])
        provenance.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "raw_rows": raw_n, "clean_rows": len(d), "clips": d.clip_id.nunique()})
    samples = pd.concat(frames, ignore_index=True)
    segments = samples.groupby("clip_id", sort=False).agg(
        source=("source", "first"), n_samples=("timestamp", "size"),
        start_time=("timestamp", "first"), end_time=("timestamp", "last")).reset_index()
    normal_ids = segments.loc[segments.source=='normal', 'clip_id'].tolist()
    if len(normal_ids)!=599 or (segments.source=='fault').sum()!=21:
        raise ValueError("Dataset does not match the agreed 599 normal / 21 fault clips")
    split_ids = {"train": normal_ids[:305], "validation": normal_ids[305:359],
                 "calibration": normal_ids[359:479], "normal_test": normal_ids[479:],
                 "fault_test": segments.loc[segments.source=='fault','clip_id'].tolist()}
    mapping = {sid:split for split, ids in split_ids.items() for sid in ids}
    samples["split"] = samples.clip_id.map(mapping)
    segments["split"] = segments.clip_id.map(mapping)
    return samples, segments, provenance
