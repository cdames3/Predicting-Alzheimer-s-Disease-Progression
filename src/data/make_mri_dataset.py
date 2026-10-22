"""Build the OASIS-2 MRI scan inventory and quality metrics from the raw archives.

The two OASIS-2 raw archives hold one folder per MRI session
(``OAS2_0001_MR1/RAW/mpr-1.nifti.img`` + ``.hdr``), with 2 to 4 repeated
T1-weighted MPRAGE scans per session. Extracted, they take about 45 GB, so
this script streams each ``.tar.gz`` once and never writes a full volume to
disk.

For every scan it records the header metadata, intensity statistics, a
signal-to-noise estimate, and an exploratory CSF proxy. For every session it
records the agreement between repeated scans and keeps the central slices of
the first scan for the EDA figures.

Outputs:
    data/processed/oasis2_mri_scans.csv      one row per scan
    data/processed/oasis2_mri_sessions.csv   one row per session (joins on mri_id)
    data/processed/oasis2_mri_metadata.json  archive hashes, counts, linkage
    data/interim/oasis2_mri_slices.npz       central slices for the figures

Usage:
    python -m src.data.make_mri_dataset
"""

import hashlib
import io
import json
import re
import tarfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from src import config

SCAN_PATTERN = re.compile(r"(OAS2_\d{4}_MR\d)/RAW/mpr-(\d)\.nifti\.(img|hdr)$")

# Central box used for the CSF proxy, as a fraction of each image axis. It
# covers the lateral ventricles and deep brain and excludes most of the skull.
CSF_BOX_FRACTION = 0.4
# Voxels darker than this fraction of the bright-tissue reference count as
# CSF-like (ventricles and sulci appear dark on T1-weighted images).
CSF_INTENSITY_RATIO = 0.35
BACKGROUND_CORNER = 20  # voxels per side of each background corner cube
DOWNSAMPLE = 4  # block size for the volumes used to compare repeated scans
# Quality flags mark extreme low outliers only (Tukey's "far out" fence).
QC_IQR_MULTIPLIER = 3.0


class HashingReader(io.RawIOBase):
    """File wrapper that computes the SHA-256 of everything read through it."""

    def __init__(self, path: Path):
        self._file = open(path, "rb")
        self.sha256 = hashlib.sha256()

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        n = self._file.readinto(buffer)
        self.sha256.update(memoryview(buffer)[:n])
        return n

    def close(self) -> None:
        self._file.close()
        super().close()


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Return the Otsu threshold that best separates two intensity classes."""
    counts, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    weight_low = np.cumsum(counts)
    weight_high = weight_low[-1] - weight_low
    mass_low = np.cumsum(counts * centers)
    mean_low = mass_low / np.maximum(weight_low, 1)
    mean_high = (mass_low[-1] - mass_low) / np.maximum(weight_high, 1)
    between_var = weight_low * weight_high * (mean_low - mean_high) ** 2
    return float(centers[np.argmax(between_var)])


def downsample(volume: np.ndarray, block: int = DOWNSAMPLE) -> np.ndarray:
    """Average non-overlapping ``block``-sized cubes of a 3D volume."""
    z, y, x = (s // block for s in volume.shape)
    trimmed = volume[: z * block, : y * block, : x * block]
    return trimmed.reshape(z, block, y, block, x, block).mean(axis=(1, 3, 5))


def central_box(volume: np.ndarray, fraction: float = CSF_BOX_FRACTION) -> np.ndarray:
    """Return the centered sub-volume covering ``fraction`` of each axis."""
    slices = tuple(slice(int(n * (0.5 - fraction / 2)), int(n * (0.5 + fraction / 2)))
                   for n in volume.shape)
    return volume[slices]


def load_volume(img_bytes: bytes, hdr_bytes: bytes) -> tuple[np.ndarray, nib.AnalyzeHeader]:
    """Decode an Analyze 7.5 image into a (sagittal, y, x) float32 array."""
    header = nib.AnalyzeHeader.from_fileobj(io.BytesIO(hdr_bytes))
    nx, ny, nz = (int(d) for d in header.get_data_shape()[:3])
    dtype = header.get_data_dtype()
    # Analyze stores x fastest, so C order gives (z, y, x).
    volume = np.frombuffer(img_bytes, dtype=dtype).reshape(nz, ny, nx)
    return volume.astype(np.float32), header


def scan_metrics(volume: np.ndarray, header: nib.AnalyzeHeader) -> dict:
    """Compute header metadata, intensity statistics, SNR, and the CSF proxy.

    The foreground is separated from air with an Otsu threshold. SNR is the
    mean foreground intensity divided by the standard deviation of the eight
    background corner cubes.
    """
    shape = header.get_data_shape()[:3]
    zooms = header.get_zooms()[:3]
    voxel_mm3 = float(np.prod(zooms))

    threshold = otsu_threshold(volume[::2, ::2, ::2])
    foreground = volume[volume > threshold]
    c = BACKGROUND_CORNER
    corners = [volume[zs, ys, xs]
               for zs in (slice(0, c), slice(-c, None))
               for ys in (slice(0, c), slice(-c, None))
               for xs in (slice(0, c), slice(-c, None))]
    background_std = float(np.std(np.concatenate([b.ravel() for b in corners])))

    tissue_reference = np.percentile(foreground, 75)
    box = central_box(volume) / tissue_reference
    p01, p50, p99 = np.percentile(foreground, [1, 50, 99])
    return {
        "dim_x": int(shape[0]), "dim_y": int(shape[1]), "dim_z": int(shape[2]),
        "voxel_x_mm": float(zooms[0]), "voxel_y_mm": float(zooms[1]),
        "voxel_z_mm": float(zooms[2]),
        "dtype": str(header.get_data_dtype()),
        "intensity_min": float(volume.min()),
        "intensity_max": float(volume.max()),
        "otsu_threshold": threshold,
        "foreground_volume_cm3": foreground.size * voxel_mm3 / 1000,
        "foreground_p01": float(p01),
        "foreground_p50": float(p50),
        "foreground_p99": float(p99),
        "background_std": background_std,
        "snr": float(foreground.mean() / background_std),
        "csf_fraction_proxy": float((box < CSF_INTENSITY_RATIO).mean()),
    }


def central_slices(volume: np.ndarray) -> dict[str, np.ndarray]:
    """Half-resolution sagittal, axial, and coronal slices through the center."""
    z, y, x = (n // 2 for n in volume.shape)
    return {
        "sagittal": volume[z, ::2, ::2].astype(np.float16),
        "axial": volume[::2, y, ::2].astype(np.float16),
        "coronal": volume[::2, ::2, x].astype(np.float16),
    }


def repeat_scan_agreement(small_volumes: list[np.ndarray]) -> float:
    """Mean Pearson r between the first scan and each repeat (downsampled).

    Repeats are not registered, so low values point to head motion or
    repositioning between scans within the session.
    """
    if len(small_volumes) < 2:
        return float("nan")
    first = small_volumes[0].ravel()
    return float(np.mean([np.corrcoef(first, other.ravel())[0, 1]
                          for other in small_volumes[1:]]))


def process_archive(path: Path) -> dict:
    """Stream one archive and return its scan rows, session rows, and slices.

    Inside the archive a scan's ``.img`` may come before its ``.hdr``, so each
    file is held in memory only until its partner arrives.
    """
    reader = HashingReader(path)
    pending: dict[tuple[str, int], dict[str, bytes]] = {}
    scans, small_volumes, slices = [], {}, {}

    with tarfile.open(fileobj=io.BufferedReader(reader, 1 << 20), mode="r|gz") as tar:
        for member in tar:
            match = SCAN_PATTERN.search(member.name)
            if not match or not member.isfile():
                continue
            mri_id, scan_number, ext = match.group(1), int(match.group(2)), match.group(3)
            parts = pending.setdefault((mri_id, scan_number), {})
            parts[ext] = tar.extractfile(member).read()
            if len(parts) < 2:
                continue

            volume, header = load_volume(parts["img"], parts["hdr"])
            del pending[(mri_id, scan_number)]
            scans.append({"mri_id": mri_id, "scan": scan_number,
                          "archive": path.name, **scan_metrics(volume, header)})
            small_volumes.setdefault(mri_id, {})[scan_number] = downsample(volume)
            if scan_number == 1:
                slices[mri_id] = central_slices(volume)

    if pending:
        raise ValueError(f"{path.name}: scans missing their .img or .hdr: {sorted(pending)}")

    sessions = []
    for mri_id, by_scan in small_volumes.items():
        ordered = [by_scan[k] for k in sorted(by_scan)]
        sessions.append({"mri_id": mri_id,
                         "repeat_scan_r": repeat_scan_agreement(ordered)})
    return {"archive": path.name, "sha256": reader.sha256.hexdigest(),
            "size_bytes": path.stat().st_size, "scans": scans,
            "sessions": sessions, "slices": slices}


def summarize_sessions(scans: pd.DataFrame, agreement: pd.DataFrame) -> pd.DataFrame:
    """Average scan metrics per session and add simple quality flags.

    A session is flagged when its SNR or repeat-scan agreement falls more than
    ``QC_IQR_MULTIPLIER`` IQRs below the first quartile across all sessions.
    """
    metric_columns = ["snr", "foreground_p50", "foreground_p99",
                      "foreground_volume_cm3", "csf_fraction_proxy"]
    sessions = (scans.groupby("mri_id")
                .agg(archive=("archive", "first"), n_scans=("scan", "size"),
                     **{c: (c, "mean") for c in metric_columns})
                .reset_index()
                .merge(agreement, on="mri_id", how="left"))
    sessions["subject_id"] = sessions["mri_id"].str[:9]
    sessions["session"] = sessions["mri_id"].str[-1].astype(int)

    for column in ["snr", "repeat_scan_r"]:
        q1, q3 = sessions[column].quantile([0.25, 0.75])
        cutoff = q1 - QC_IQR_MULTIPLIER * (q3 - q1)
        sessions[f"low_{column}"] = (sessions[column] < cutoff).astype(int)
    sessions["qc_flag"] = sessions[["low_snr", "low_repeat_scan_r"]].max(axis=1)
    return sessions


def link_to_clinical(sessions: pd.DataFrame) -> dict:
    """Check that every clinical visit and visit pair has a matching scan.

    Visit pairs are rebuilt from the raw CSV (each visit and the patient's
    next visit), the same way ``make_dataset`` builds them.
    """
    raw = pd.read_csv(config.RAW_CSV).sort_values(["Subject ID", "Visit"])
    raw["next_mri_id"] = raw.groupby("Subject ID")["MRI ID"].shift(-1)
    pairs = (raw.dropna(subset=["next_mri_id"])
             .rename(columns={"MRI ID": "mri_id"}))

    scanned = set(sessions["mri_id"])
    visits = set(raw["MRI ID"])
    return {
        "clinical_visits": len(visits),
        "mri_sessions": len(scanned),
        "visits_with_scan": len(visits & scanned),
        "sessions_without_visit": sorted(scanned - visits),
        "visits_without_scan": sorted(visits - scanned),
        "visit_pairs": len(pairs),
        "pairs_with_scan_at_t": int(pairs["mri_id"].isin(scanned).sum()),
        "pairs_with_scans_at_both_visits": int(
            (pairs["mri_id"].isin(scanned) & pairs["next_mri_id"].isin(scanned)).sum()),
    }


def main() -> None:
    """Stream both archives in parallel and write the MRI tables and metadata."""
    missing = [p.name for p in config.MRI_ARCHIVES if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Place the OASIS-2 archives in data/raw/: {missing}")

    with ProcessPoolExecutor(max_workers=len(config.MRI_ARCHIVES)) as pool:
        results = list(pool.map(process_archive, config.MRI_ARCHIVES))

    scans = (pd.DataFrame([row for r in results for row in r["scans"]])
             .sort_values(["mri_id", "scan"]).reset_index(drop=True))
    agreement = pd.DataFrame([row for r in results for row in r["sessions"]])
    sessions = summarize_sessions(scans, agreement)
    linkage = link_to_clinical(sessions)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    scans.to_csv(config.MRI_SCANS_CSV, index=False)
    sessions.to_csv(config.MRI_SESSIONS_CSV, index=False)

    slices = {f"{mri_id}/{view}": image for r in results
              for mri_id, views in r["slices"].items() for view, image in views.items()}
    np.savez_compressed(config.MRI_SLICES, **slices)

    header_values = scans[["dim_x", "dim_y", "dim_z", "voxel_x_mm", "voxel_y_mm",
                           "voxel_z_mm", "dtype"]].drop_duplicates()
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "archives": [{"file": r["archive"], "sha256": r["sha256"],
                      "size_bytes": r["size_bytes"], "scans": len(r["scans"]),
                      "sessions": len(r["sessions"])} for r in results],
        "scans": len(scans),
        "sessions": len(sessions),
        "subjects": int(sessions["subject_id"].nunique()),
        "header_variants": header_values.to_dict(orient="records"),
        "linkage": linkage,
        "qc_flagged_sessions": sessions.loc[sessions["qc_flag"] == 1, "mri_id"].tolist(),
        "parameters": {"csf_box_fraction": CSF_BOX_FRACTION,
                       "csf_intensity_ratio": CSF_INTENSITY_RATIO,
                       "background_corner_voxels": BACKGROUND_CORNER,
                       "downsample_block": DOWNSAMPLE,
                       "qc_iqr_multiplier": QC_IQR_MULTIPLIER},
        "output_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (config.MRI_SCANS_CSV, config.MRI_SESSIONS_CSV)},
    }
    config.MRI_METADATA.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Wrote {len(scans)} scans / {len(sessions)} sessions to {config.PROCESSED_DIR}")
    print(json.dumps({k: v for k, v in linkage.items() if not isinstance(v, list)}, indent=2))


if __name__ == "__main__":
    main()
