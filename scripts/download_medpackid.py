#!/usr/bin/env python3
"""Download and extract the public MedPackID Mendeley dataset on the DGX host."""
import hashlib
import os
from pathlib import Path
import zipfile

import requests
from tqdm import tqdm

DATASET_ID = "d23kndp3sv"
VERSION = 1
DOWNLOAD_URL = f"https://data.mendeley.com/public-api/zip/{DATASET_ID}/download/{VERSION}"
EXPECTED_SHA256 = "c47e1b5133765801805e41ce4c3a688126aeaaf257d6e0af7c151ff6ae0df835"
ROOT = Path(os.environ.get("MEDPACK_DATA_ROOT", "data/MedPackID")).resolve()
ARCHIVE = Path(os.environ.get("MEDPACK_ARCHIVE_PATH", ROOT / f"MedPackID_v{VERSION}.zip")).expanduser().resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download():
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if ARCHIVE.exists() and sha256(ARCHIVE).lower() == EXPECTED_SHA256:
        print(f"Using verified archive: {ARCHIVE}")
        return
    if ARCHIVE.exists():
        ARCHIVE.unlink()
    with requests.get(DOWNLOAD_URL, stream=True, timeout=(90, 300)) as response:
        if response.status_code == 401:
            raise RuntimeError("Public Mendeley Download All link returned HTTP 401.")
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with ARCHIVE.open("wb") as out, tqdm(total=total or None, unit="B", unit_scale=True,
                                                desc="MedPackID download") as bar:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    out.write(chunk)
                    bar.update(len(chunk))
    actual = sha256(ARCHIVE)
    if actual.lower() != EXPECTED_SHA256:
        ARCHIVE.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 mismatch; expected {EXPECTED_SHA256}, got {actual}. Archive removed.")
    if not zipfile.is_zipfile(ARCHIVE):
        ARCHIVE.unlink(missing_ok=True)
        raise RuntimeError("Mendeley response is not a valid ZIP; partial file removed.")
    print(f"Verified SHA-256: {actual}")


def extract_nested_zips():
    ROOT.mkdir(parents=True, exist_ok=True)
    extracted_marker = ROOT / ".download_all_extracted"
    if not extracted_marker.exists():
        with zipfile.ZipFile(ARCHIVE) as zf:
            bad = zf.testzip()
            if bad:
                raise RuntimeError(f"Corrupt archive member: {bad}")
            zf.extractall(ROOT)
        extracted_marker.touch()
    # Download All contains images.zip, labels.zip, and annotations_voc.zip.
    while True:
        zips = [p for p in ROOT.rglob("*.zip") if p.is_file()]
        pending = [p for p in zips if not p.with_suffix(p.suffix + ".extracted").exists()]
        if not pending:
            break
        for path in pending:
            with zipfile.ZipFile(path) as zf:
                bad = zf.testzip()
                if bad:
                    raise RuntimeError(f"Corrupt nested archive {path.name}, member {bad}")
                zf.extractall(path.parent)
            path.with_suffix(path.suffix + ".extracted").touch()
            print(f"Extracted {path.name}")
    print(f"MedPackID files ready under {ROOT}")


if __name__ == "__main__":
    download()
    extract_nested_zips()
