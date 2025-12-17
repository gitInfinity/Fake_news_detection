#!/usr/bin/env python3
"""Download model files into the project's `models/` directory.

Usage:
    python scripts/download_models.py [--force]

Configuration:
- Edit the MODELS list below and provide either a 'url' or a 'gdrive_id' for each entry.
- The `filename` is the target name saved under `models/` (e.g., fake_news_classifier.h5).

Notes:
- For Google Drive downloads this script prefers `gdown` (pip install gdown).
  If `gdown` is missing and you supplied a `gdrive_id`, the script will print an instruction.
- For direct URLs the script uses `requests` to stream the file to disk.

This is intentionally simple and does not attempt to authenticate to private stores.
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Configure models to download below.
# Provide either a 'url' (direct HTTP(S) link) or a 'gdrive_id' for Google Drive.
# Example placeholders; replace with your real URLs or Drive IDs.
MODELS: List[Dict[str, str]] = [
    {
        # Recommended filename for the Keras model used by the app
        "filename": "fake_news_classifier.h5",
        # "url": "https://example.com/path/to/fake_news_classifier.h5",
        # or a Google Drive file id:
        # "gdrive_id": "1A2bC3..."
    },
    {
        "filename": "word2vec_model.model",
        # "url": "https://example.com/path/to/word2vec_model.model",
        # "gdrive_id": "1XyZ..."
    },
]

# --- Helper functions ---


def _download_http(url: str, dest: Path, force: bool = False) -> None:
    import requests

    if dest.exists() and not force:
        print(f"Skipping {dest.name} (exists). Use --force to overwrite.")
        return

    print(f"Downloading {url} -> {dest}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = r.headers.get("content-length")
        if total is None:
            # No content length header
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        else:
            total = int(total)
            downloaded = 0
            start = time.time()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        done = int(40 * downloaded / total)
                        elapsed = time.time() - start
                        speed = downloaded / (1024 * elapsed + 1e-9)
                        sys.stdout.write(
                            f"\r[{'=' * done}{' ' * (40-done)}] {downloaded/1024:.1f}KB/{total/1024:.1f}KB {speed:.1f}KB/s"
                        )
                        sys.stdout.flush()
            sys.stdout.write("\n")
    print(f"Saved {dest}.")


def _download_gdrive(gdrive_id: str, dest: Path, force: bool = False) -> None:
    try:
        import gdown
    except Exception:
        print("gdown not installed. To download from Google Drive, install it: pip install gdown")
        raise

    if dest.exists() and not force:
        print(f"Skipping {dest.name} (exists). Use --force to overwrite.)")
        return

    url = f"https://drive.google.com/uc?id={gdrive_id}"
    print(f"Downloading from Google Drive id={gdrive_id} -> {dest}")
    gdown.download(url, str(dest), quiet=False)
    print(f"Saved {dest}.")


# --- Main ---



    def _env_sources_for(filename: str) -> tuple[str | None, str | None]:
        """Look up environment variables for URL or Google Drive id for a filename.

        Supports short keys for the two default filenames and generic derived keys.
        Returns (url, gdrive_id) where either may be None.
        """
        # short convenient environment variable names for common files
        short_map = {
            "fake_news_classifier.h5": ("MODEL_FAKE_URL", "MODEL_FAKE_GDRIVE_ID"),
            "word2vec_model.model": ("MODEL_W2V_URL", "MODEL_W2V_GDRIVE_ID"),
        }
        if filename in short_map:
            key_url, key_gid = short_map[filename]
            return (os.getenv(key_url), os.getenv(key_gid))

        # otherwise derive an env var name from filename
        base = filename.upper().replace('.', '_').replace('-', '_').replace(' ', '_')
        url = os.getenv(f"{base}_URL") or os.getenv(f"{base}_HTTP_URL")
        gdrive = os.getenv(f"{base}_GDRIVE_ID") or os.getenv(f"{base}_GDRIVE")
        return (url, gdrive)
def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download model artifacts into models/")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args(argv)

    if not MODELS:
        print("No models configured in the script. Edit scripts/download_models.py and add URLs or gdrive IDs.")
        return 1

    errors = 0
    for m in MODELS:
        filename = m.get("filename")
        if not filename:
            print("Model entry missing 'filename' key. Skipping.")
            continue
        dest = MODELS_DIR / filename

        # explicit config in script
        url = m.get("url")
        gdrive_id = m.get("gdrive_id")

        # if not provided in the script, try environment variables
        if not url or not gdrive_id:
            env_url, env_gdrive = _env_sources_for(filename)
            url = url or env_url
            gdrive_id = gdrive_id or env_gdrive

        try:
            if url:
                _download_http(url, dest, force=args.force)
            elif gdrive_id:
                _download_gdrive(gdrive_id, dest, force=args.force)
            else:
                print(f"No 'url' or 'gdrive_id' provided for {filename}. Skipping.")
                errors += 1
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
            errors += 1

    if errors:
        print(f"Completed with {errors} errors.")
        return 2
    print("All downloads complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
