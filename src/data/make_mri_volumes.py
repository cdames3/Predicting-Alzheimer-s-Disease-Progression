"""Build the downsampled MRI volumes used as CNN inputs.

The CNN cannot read the raw archives directly: each scan is a 256 x 256 x 128
Analyze volume and the archives are about 19 GB compressed. This script
streams both archives once (like ``make_mri_dataset``), keeps the first
MPRAGE scan of every session, and stores a small, normalized copy of it.

Preprocessing of each volume:
    1. Average non-overlapping blocks of ``BLOCK`` voxels (128 x 256 x 256 ->
       64 x 64 x 64, about 2.5 x 4 x 4 mm per voxel).
    2. Z-score the intensities using the head voxels only (above the Otsu
       threshold), so scanner gain differences between sessions cancel out.
    3. Clip to ``CLIP_SD`` standard deviations and store as float16.

Only the first scan is kept because repeats within a session are not
registered to each other; averaging them would blur the anatomy.

Output:
    data/interim/oasis2_mri_volumes.npz   one array per mri_id, shape (64, 64, 64)

Usage:
    python -m src.data.make_mri_volumes
"""

import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from src import config
from src.data.make_mri_dataset import SCAN_PATTERN, load_volume, otsu_threshold

# Block size per axis (sagittal, y, x). Sagittal voxels are 1.25 mm and the
# in-plane voxels 1 mm, so (2, 4, 4) gives a cubic 64^3 grid.
BLOCK = (2, 4, 4)
CLIP_SD = 5.0
FIRST_SCAN = 1


def block_mean(volume: np.ndarray, block: tuple[int, int, int] = BLOCK) -> np.ndarray:
    """Average non-overlapping blocks of shape ``block`` in a 3D volume."""
    bz, by, bx = block
    z, y, x = volume.shape[0] // bz, volume.shape[1] // by, volume.shape[2] // bx
    trimmed = volume[: z * bz, : y * by, : x * bx]
    return trimmed.reshape(z, bz, y, by, x, bx).mean(axis=(1, 3, 5))


def preprocess(volume: np.ndarray) -> np.ndarray:
    """Downsample, z-score on head voxels, and clip one raw volume."""
    small = block_mean(volume)
    head = small[small > otsu_threshold(small)]
    normalized = (small - head.mean()) / head.std()
    return np.clip(normalized, -CLIP_SD, CLIP_SD).astype(np.float16)


def process_archive(path: Path) -> dict[str, np.ndarray]:
    """Stream one archive and return the preprocessed first scan per session."""
    pending: dict[str, dict[str, bytes]] = {}
    volumes = {}
    with tarfile.open(path, mode="r|gz") as tar:
        for member in tar:
            match = SCAN_PATTERN.search(member.name)
            if not match or not member.isfile() or int(match.group(2)) != FIRST_SCAN:
                continue
            mri_id, ext = match.group(1), match.group(3)
            parts = pending.setdefault(mri_id, {})
            parts[ext] = tar.extractfile(member).read()
            if len(parts) < 2:
                continue
            volume, _ = load_volume(parts["img"], parts["hdr"])
            volumes[mri_id] = preprocess(volume)
            del pending[mri_id]

    if pending:
        raise ValueError(f"{path.name}: scans missing their .img or .hdr: {sorted(pending)}")
    return volumes


def load_volumes(path: Path = config.MRI_VOLUMES) -> dict[str, np.ndarray]:
    """Load the stored volumes as a ``{mri_id: float32 array}`` dictionary."""
    if not path.exists():
        raise FileNotFoundError("Run `python -m src.data.make_mri_volumes` first.")
    with np.load(path) as data:
        return {mri_id: data[mri_id].astype(np.float32) for mri_id in data.files}


def main() -> None:
    """Stream both archives in parallel and write the volume file."""
    missing = [p.name for p in config.MRI_ARCHIVES if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Place the OASIS-2 archives in data/raw/: {missing}")

    with ProcessPoolExecutor(max_workers=len(config.MRI_ARCHIVES)) as pool:
        results = list(pool.map(process_archive, config.MRI_ARCHIVES))
    volumes = {mri_id: v for r in results for mri_id, v in sorted(r.items())}

    config.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(config.MRI_VOLUMES, **volumes)
    shape = next(iter(volumes.values())).shape
    print(f"Wrote {len(volumes)} volumes of shape {shape} to {config.MRI_VOLUMES}")


if __name__ == "__main__":
    main()
