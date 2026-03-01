#!/usr/bin/env python3
"""
Download the SciDB Infrared Thermal Image Dataset via FTP.

Connects to ftp.scidb.cn and downloads thermal images
into data/raw/ for processing with prepare_data.py.
"""

import os
import sys
from ftplib import FTP
from pathlib import Path


def list_dir_recursive(ftp, path=".", indent=0):
    """List FTP directory contents recursively (for exploration)."""
    items = []
    try:
        entries = ftp.nlst(path)
    except Exception:
        return items

    for entry in entries:
        name = entry.split("/")[-1] if "/" in entry else entry
        if name in (".", ".."):
            continue
        items.append((entry, indent))
        print(f"{'  ' * indent}{'📁' if '.' not in name else '📄'} {name}")
        # Try to recurse (if it's a directory)
        if "." not in name:  # heuristic: no extension = directory
            try:
                sub = ftp.nlst(entry)
                if sub != [entry]:  # it's a directory
                    items.extend(list_dir_recursive(ftp, entry, indent + 1))
            except Exception:
                pass
    return items


def download_file(ftp, remote_path, local_path):
    """Download a single file from FTP."""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    with open(local_path, "wb") as f:
        ftp.retrbinary(f"RETR {remote_path}", f.write)


def download_directory(ftp, remote_dir, local_dir, extensions=None):
    """
    Recursively download all files from an FTP directory.

    Args:
        ftp: Connected FTP object
        remote_dir: Remote directory path
        local_dir: Local destination directory
        extensions: Set of extensions to filter (e.g. {'.jpg', '.png'})
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        entries = ftp.nlst(remote_dir)
    except Exception as e:
        print(f"  ✗ Cannot list {remote_dir}: {e}")
        return 0

    downloaded = 0
    for entry in entries:
        name = entry.split("/")[-1] if "/" in entry else entry
        if name in (".", ".."):
            continue

        remote_path = f"{remote_dir}/{name}" if not entry.startswith(remote_dir) else entry
        local_path = local_dir / name

        # Check if it's a file (has extension) or directory
        ext = Path(name).suffix.lower()
        if ext:
            # It's a file
            if extensions is None or ext in extensions:
                try:
                    print(f"  ↓ {name}", end="", flush=True)
                    download_file(ftp, remote_path, local_path)
                    size = local_path.stat().st_size
                    print(f"  ({size / 1024:.0f} KB)")
                    downloaded += 1
                except Exception as e:
                    print(f"  ✗ Failed: {e}")
        else:
            # Try as directory
            try:
                sub_entries = ftp.nlst(remote_path)
                if sub_entries and sub_entries != [remote_path]:
                    print(f"\n📁 Entering {name}/")
                    downloaded += download_directory(
                        ftp, remote_path, local_path, extensions
                    )
            except Exception:
                pass

    return downloaded


def main():
    host = "ftp.scidb.cn"
    port = 2121
    user = "Q3lbau"
    passwd = "77jaeq"
    local_raw = Path("data/raw")
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    print(f"{'='*60}")
    print(f"  SciDB Thermal Image Dataset Downloader")
    print(f"{'='*60}")
    print(f"  Host: {host}:{port}")
    print(f"  Destination: {local_raw}/")
    print()

    # Connect
    print("Connecting to FTP server...")
    try:
        ftp = FTP()
        ftp.connect(host, port, timeout=30)
        ftp.login(user, passwd)
        print(f"  ✓ Connected as {user}")
        print(f"  Welcome: {ftp.getwelcome()}")
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        sys.exit(1)

    # Explore the root directory first
    print("\nExploring dataset structure...")
    try:
        root_entries = ftp.nlst(".")
        print(f"  Root contents:")
        for entry in root_entries:
            if entry not in (".", ".."):
                print(f"    📁 {entry}")
    except Exception as e:
        print(f"  Error listing root: {e}")

    # Download everything (the server usually has flat structure)
    print(f"\nDownloading images to {local_raw}/...")
    total = download_directory(ftp, ".", local_raw, image_extensions)

    # Close
    try:
        ftp.quit()
    except Exception:
        ftp.close()

    print(f"\n{'='*60}")
    print(f"  ✅  Download complete!")
    print(f"  Total files: {total}")
    print(f"  Location: {local_raw}/")
    print(f"{'='*60}")
    print(f"\n  Next step:  python prepare_data.py --raw data/raw")


if __name__ == "__main__":
    main()
