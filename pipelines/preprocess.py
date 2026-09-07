"""
preprocess.py — Konversi FITS 2048x2048 → NPY 512x512

Pipeline:
    1. Scan seluruh file .fits di fits_train_dir dan fits_test_dir.
    2. Buka setiap file dengan astropy; tangkap AstropyWarning sebagai sinyal file korup.
    3. Bersihkan NaN/Inf → 0.0.
    4. Downsample 2048×2048 → 512×512 menggunakan cv2.INTER_AREA
       (melestarikan fluks spasial rata-rata, bukan sekadar nearest-neighbor).
    5. Simpan sebagai float32 .npy di data/processed/fits/{train,test}/.

Dipanggil melalui main.py sebagai mode tersendiri (lihat integrasi di bawah).
Dapat juga dijalankan langsung: python pipelines/preprocess.py --config config.yaml
"""

import argparse
import os
import sys
import warnings

import cv2
import numpy as np
from tqdm import tqdm

try:
    from astropy.io import fits
    from astropy.utils.exceptions import AstropyWarning
except ImportError:
    print("[CRITICAL] astropy tidak terinstal. Jalankan: pip install astropy")
    sys.exit(1)

try:
    from omegaconf import OmegaConf
except ImportError:
    print("[CRITICAL] omegaconf tidak terinstal. Jalankan: pip install omegaconf")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------

TARGET_SIZE = (512, 512)   # (width, height) untuk cv2.resize


# ---------------------------------------------------------------------------
# Core: Baca dan Pra-proses Satu File FITS
# ---------------------------------------------------------------------------

def load_and_preprocess_fits(fits_path: str) -> np.ndarray | None:
    """
    Baca file FITS, bersihkan, dan downsample ke TARGET_SIZE.

    Returns:
        np.ndarray float32 shape (512, 512), atau None jika file korup/invalid.

    Strategi penanganan error:
        - AstropyWarning (termasuk pesan "truncated", "corrupt", "incomplete")
          ditangkap sebagai sinyal file bermasalah → return None (skip).
        - Exception keras (OSError, ValueError) juga ditangkap → return None.
        - NaN dan Inf dibersihkan menjadi 0.0 setelah pembacaan berhasil.
    """
    caught_warnings: list[warnings.WarningMessage] = []

    try:
        # Tangkap semua AstropyWarning selama pembacaan
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", AstropyWarning)

            with fits.open(fits_path, memmap=False) as hdul:
                data = None
                for hdu in hdul:
                    if hdu.data is not None and np.ndim(hdu.data) == 2:
                        data = hdu.data
                        break

        # Jika ada AstropyWarning tertangkap, anggap file bermasalah
        if caught_warnings:
            warning_msgs = [str(w.message) for w in caught_warnings]
            print(
                f"  [SKIP] AstropyWarning pada '{os.path.basename(fits_path)}': "
                f"{warning_msgs[0][:80]}..."
            )
            return None

        if data is None:
            print(f"  [SKIP] Tidak ada data 2D valid di: '{os.path.basename(fits_path)}'")
            return None

    except Exception as exc:
        print(f"  [SKIP] Exception saat membaca '{os.path.basename(fits_path)}': {exc}")
        return None

    # Konversi ke float32 (astropy sudah terapkan BZERO/BSCALE)
    data = data.astype(np.float32)

    # Bersihkan anomali NaN dan Inf → 0.0
    # Menggunakan np.isfinite() lebih efisien daripada dua panggilan terpisah
    nan_inf_count = np.sum(~np.isfinite(data))
    if nan_inf_count > 0:
        data = np.where(np.isfinite(data), data, 0.0)

    # Downsample 2048×2048 → 512×512 menggunakan INTER_AREA
    # INTER_AREA menghitung rata-rata piksel dalam area kernel downsampling,
    # melestarikan fluks total per area lebih baik daripada INTER_LINEAR atau
    # INTER_NEAREST untuk faktor downscale besar (4×).
    data_downsampled = cv2.resize(
        data,
        TARGET_SIZE,
        interpolation=cv2.INTER_AREA,
    )

    return data_downsampled.astype(np.float32)


# ---------------------------------------------------------------------------
# Core: Proses Satu Direktori FITS
# ---------------------------------------------------------------------------

def process_fits_directory(
    fits_dir: str,
    output_dir: str,
    partition_name: str,
) -> dict:
    """
    Scan direktori FITS, proses setiap file, simpan sebagai .npy.

    Args:
        fits_dir       : Path direktori sumber berisi *.fits.
        output_dir     : Path direktori tujuan untuk *.npy.
        partition_name : Label partisi untuk logging ("train" atau "test").

    Returns:
        Dict statistik: total, success, skipped, already_exists.
    """
    os.makedirs(output_dir, exist_ok=True)

    fits_files = sorted([
        f for f in os.listdir(fits_dir)
        if f.lower().endswith(".fits")
    ])

    if not fits_files:
        print(f"[WARN] Tidak ada file .fits ditemukan di: '{fits_dir}'")
        return {"total": 0, "success": 0, "skipped": 0, "already_exists": 0}

    print(
        f"\n[INFO] Memproses partisi '{partition_name}': "
        f"{len(fits_files)} file FITS → '{output_dir}'"
    )

    stats = {"total": len(fits_files), "success": 0, "skipped": 0, "already_exists": 0}

    pbar = tqdm(fits_files, desc=f"  [{partition_name.upper()}]", unit="file")

    for fname in pbar:
        fits_path = os.path.join(fits_dir, fname)
        stem = os.path.splitext(fname)[0]
        npy_path = os.path.join(output_dir, f"{stem}.npy")

        # Skip jika sudah ada (idempoten — aman untuk dijalankan ulang)
        if os.path.exists(npy_path):
            stats["already_exists"] += 1
            pbar.set_postfix({"status": "cached"})
            continue

        result = load_and_preprocess_fits(fits_path)

        if result is None:
            stats["skipped"] += 1
            pbar.set_postfix({"status": "SKIP"})
            continue

        np.save(npy_path, result)
        stats["success"] += 1
        pbar.set_postfix({"status": "ok", "shape": str(result.shape)})

    return stats


# ---------------------------------------------------------------------------
# Main Preprocessing Routine (dipanggil dari main.py atau langsung)
# ---------------------------------------------------------------------------

def preprocess_routine(config):
    """
    Entry point utama yang dipanggil dari main.py --mode preprocess.

    Memproses kedua partisi (train dan test) secara berurutan.
    """
    print("[INFO] Memulai rutin pra-pemrosesan FITS → NPY...")

    fits_train_dir = config.system.fits_train_dir
    fits_test_dir = getattr(config.system, "fits_test_dir", None)
    processed_base = os.path.join(config.system.data_dir, "processed", "fits")

    npy_train_dir = os.path.join(processed_base, "train")
    npy_test_dir = os.path.join(processed_base, "test")

    all_stats = {}

    # --- Partisi Train ---
    if os.path.isdir(fits_train_dir):
        stats = process_fits_directory(fits_train_dir, npy_train_dir, "train")
        all_stats["train"] = stats
    else:
        print(f"[WARN] Direktori FITS train tidak ditemukan: '{fits_train_dir}'. Dilewati.")

    # --- Partisi Test ---
    if fits_test_dir and os.path.isdir(fits_test_dir):
        stats = process_fits_directory(fits_test_dir, npy_test_dir, "test")
        all_stats["test"] = stats
    else:
        if fits_test_dir:
            print(f"[WARN] Direktori FITS test tidak ditemukan: '{fits_test_dir}'. Dilewati.")
        else:
            print("[INFO] 'fits_test_dir' tidak dikonfigurasi. Partisi test dilewati.")

    # --- Laporan Akhir ---
    print("\n" + "=" * 60)
    print("[INFO] LAPORAN PRA-PEMROSESAN FITS → NPY")
    print("=" * 60)
    for partition, s in all_stats.items():
        print(
            f"  [{partition.upper()}]  Total: {s['total']}  |  "
            f"Berhasil: {s['success']}  |  "
            f"Skip (korup): {s['skipped']}  |  "
            f"Cache (sudah ada): {s['already_exists']}"
        )
    print("=" * 60)
    print(f"[INFO] Output NPY tersimpan di: '{processed_base}'")


# ---------------------------------------------------------------------------
# Standalone Entry Point (opsional: python pipelines/preprocess.py --config ...)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PADL-FilamentSeg: Pra-pemrosesan FITS → NPY 512×512"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path ke file konfigurasi OmegaConf YAML (default: config.yaml)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"[CRITICAL] File konfigurasi tidak ditemukan: '{args.config}'")
        sys.exit(1)

    cfg = OmegaConf.load(args.config)
    preprocess_routine(cfg)
