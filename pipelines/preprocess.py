"""
preprocess.py — Konversi FITS 2048x2048 → NPY 512x512 (Multi-core)

Pipeline:
    1. Scan seluruh file .fits di fits_train_dir dan fits_test_dir.
    2. Filter file yang sudah ada di output_dir (idempoten).
    3. Distribusikan pekerjaan ke N worker process (ProcessPoolExecutor).
       Setiap worker secara independen:
         a. Membuka file FITS; tangkap AstropyWarning sebagai sinyal korup → skip.
         b. Bersihkan NaN/Inf → 0.0.
         c. Downsample 2048×2048 → 512×512 menggunakan cv2.INTER_AREA.
         d. Simpan sebagai float32 .npy.
    4. Main process mengumpulkan hasil dan menampilkan progress bar tqdm.

Mengapa ProcessPoolExecutor bukan ThreadPoolExecutor?
    Operasi ini adalah CPU-bound (astropy dekompresi + cv2 resize). Python GIL
    memblokir thread dari berjalan paralel di operasi CPU murni. ProcessPoolExecutor
    menghindari GIL dengan melahirkan proses terpisah, sehingga semua core CPU
    dapat digunakan secara simultan.

Dipanggil melalui main.py --mode preprocess, atau langsung:
    python pipelines/preprocess.py --config config.yaml [--workers N]
"""

import argparse
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Literal

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

# Status kembalian worker — lebih ekspresif daripada bool
STATUS_OK           = "ok"
STATUS_SKIP_CORRUPT = "skip_corrupt"
STATUS_SKIP_NO_DATA = "skip_no_data"
STATUS_SKIP_ERROR   = "skip_error"
STATUS_CACHED       = "cached"


# ---------------------------------------------------------------------------
# Worker Function (top-level — wajib picklable untuk ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _worker_process_file(fits_path: str, npy_path: str) -> tuple[str, str]:
    """
    Worker function yang dijalankan di subprocess terpisah.

    Memproses SATU file FITS dan menyimpannya sebagai .npy.
    Harus berada di level modul (top-level) agar dapat di-pickle oleh
    ProcessPoolExecutor. Fungsi nested atau lambda tidak dapat di-pickle.

    Args:
        fits_path : Path absolut file FITS sumber.
        npy_path  : Path absolut file NPY tujuan.

    Returns:
        Tuple (status, pesan_detail) untuk dilaporkan ke main process.
        status adalah salah satu dari konstanta STATUS_*.
    """
    # --- Cek cache (idempoten) ---
    # Pengecekan dilakukan di dalam worker juga untuk menghindari race condition
    # jika dua worker entah bagaimana mendapat file yang sama.
    if os.path.exists(npy_path):
        return STATUS_CACHED, ""

    # --- Baca FITS dengan penangkap AstropyWarning ---
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", AstropyWarning)

            with fits.open(fits_path, memmap=False) as hdul:
                data = None
                for hdu in hdul:
                    if hdu.data is not None and np.ndim(hdu.data) == 2:
                        data = hdu.data
                        break

        if caught_warnings:
            msg = str(caught_warnings[0].message)[:100]
            return STATUS_SKIP_CORRUPT, msg

        if data is None:
            return STATUS_SKIP_NO_DATA, "Tidak ada data 2D valid"

    except Exception as exc:
        return STATUS_SKIP_ERROR, str(exc)[:100]

    # --- Konversi, bersihkan, downsample ---
    data = data.astype(np.float32)

    if not np.all(np.isfinite(data)):
        data = np.where(np.isfinite(data), data, 0.0)

    data_downsampled = cv2.resize(data, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    # --- Simpan ---
    np.save(npy_path, data_downsampled.astype(np.float32))

    return STATUS_OK, ""


# ---------------------------------------------------------------------------
# Core: Proses Satu Direktori FITS secara Paralel
# ---------------------------------------------------------------------------

def process_fits_directory(
    fits_dir: str,
    output_dir: str,
    partition_name: str,
    num_workers: int,
) -> dict:
    """
    Scan direktori FITS dan distribusikan konversi ke NPY ke N worker process.

    Args:
        fits_dir       : Path direktori sumber berisi *.fits.
        output_dir     : Path direktori tujuan untuk *.npy.
        partition_name : Label partisi untuk logging ("train" atau "test").
        num_workers    : Jumlah worker process paralel (dari config.system.workers).

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

    # Pre-filter: pisahkan file yang sudah ada dari yang perlu diproses.
    # Ini mengurangi overhead submit ke executor untuk cache hits.
    pending, cached_count = [], 0
    for fname in fits_files:
        stem = os.path.splitext(fname)[0]
        npy_path = os.path.join(output_dir, f"{stem}.npy")
        if os.path.exists(npy_path):
            cached_count += 1
        else:
            pending.append((os.path.join(fits_dir, fname), npy_path))

    total = len(fits_files)
    print(
        f"\n[INFO] Partisi '{partition_name}': {total} file FITS ditemukan. "
        f"{cached_count} sudah ter-cache, {len(pending)} perlu diproses. "
        f"Menggunakan {num_workers} worker process."
    )

    stats = {
        "total": total,
        "success": 0,
        "skipped": 0,
        "already_exists": cached_count,
    }

    if not pending:
        print(f"[INFO] Semua file '{partition_name}' sudah ter-cache. Tidak ada pekerjaan.")
        return stats

    # --- Submit semua pekerjaan ke ProcessPoolExecutor ---
    # as_completed() memungkinkan tqdm diupdate segera saat setiap future selesai,
    # bukan menunggu seluruh batch selesai (lebih responsif untuk ribuan file).
    skipped_details: list[tuple[str, str]] = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_fname = {
            executor.submit(_worker_process_file, fits_path, npy_path): os.path.basename(fits_path)
            for fits_path, npy_path in pending
        }

        pbar = tqdm(
            as_completed(future_to_fname),
            total=len(pending),
            desc=f"  [{partition_name.upper()}]",
            unit="file",
            dynamic_ncols=True,
        )

        for future in pbar:
            fname = future_to_fname[future]
            try:
                status, detail = future.result()
            except Exception as exc:
                # Tangkap exception tak terduga dari worker (sangat jarang)
                status, detail = STATUS_SKIP_ERROR, str(exc)[:100]

            if status == STATUS_OK:
                stats["success"] += 1
                pbar.set_postfix({"ok": stats["success"], "skip": stats["skipped"]})
            elif status == STATUS_CACHED:
                # Seharusnya tidak terjadi setelah pre-filter, tapi handle anyway
                stats["already_exists"] += 1
            else:
                stats["skipped"] += 1
                skipped_details.append((fname, f"[{status}] {detail}"))
                pbar.set_postfix({"ok": stats["success"], "skip": stats["skipped"]})

    # Cetak detail file yang di-skip setelah progress bar selesai
    if skipped_details:
        print(f"\n[WARN] {len(skipped_details)} file di-skip pada partisi '{partition_name}':")
        for fname, reason in skipped_details:
            print(f"       • {fname}: {reason}")

    return stats


# ---------------------------------------------------------------------------
# Main Preprocessing Routine (dipanggil dari main.py atau langsung)
# ---------------------------------------------------------------------------

def preprocess_routine(config, num_workers: int | None = None):
    """
    Entry point utama yang dipanggil dari main.py --mode preprocess.

    Memproses kedua partisi (train dan test) secara paralel antar file,
    berurutan antar partisi.

    Args:
        config      : OmegaConf config object.
        num_workers : Override jumlah worker. Jika None, gunakan config.system.workers.
    """
    # Resolusi jumlah worker: CLI override > config > default os.cpu_count()
    if num_workers is None:
        num_workers = int(getattr(config.system, "workers", os.cpu_count() or 4))

    # Batasi maksimum worker agar tidak saturate sistem
    num_workers = max(1, min(num_workers, os.cpu_count() or 1))

    print(
        f"[INFO] Memulai rutin pra-pemrosesan FITS → NPY "
        f"dengan {num_workers} worker process (dari {os.cpu_count()} CPU tersedia)..."
    )

    fits_train_dir = config.system.fits_train_dir
    fits_test_dir = getattr(config.system, "fits_test_dir", None)
    processed_base = os.path.join(config.system.data_dir, "processed", "fits")

    npy_train_dir = os.path.join(processed_base, "train")
    npy_test_dir = os.path.join(processed_base, "test")

    all_stats = {}

    # --- Partisi Train ---
    if os.path.isdir(fits_train_dir):
        stats = process_fits_directory(fits_train_dir, npy_train_dir, "train", num_workers)
        all_stats["train"] = stats
    else:
        print(f"[WARN] Direktori FITS train tidak ditemukan: '{fits_train_dir}'. Dilewati.")

    # --- Partisi Test ---
    if fits_test_dir and os.path.isdir(fits_test_dir):
        stats = process_fits_directory(fits_test_dir, npy_test_dir, "test", num_workers)
        all_stats["test"] = stats
    else:
        if fits_test_dir:
            print(f"[WARN] Direktori FITS test tidak ditemukan: '{fits_test_dir}'. Dilewati.")
        else:
            print("[INFO] 'fits_test_dir' tidak dikonfigurasi. Partisi test dilewati.")

    # --- Laporan Akhir ---
    total_success = sum(s["success"] for s in all_stats.values())
    total_skipped = sum(s["skipped"] for s in all_stats.values())
    total_cached  = sum(s["already_exists"] for s in all_stats.values())

    print("\n" + "=" * 60)
    print("[INFO] LAPORAN PRA-PEMROSESAN FITS → NPY")
    print("=" * 60)
    for partition, s in all_stats.items():
        print(
            f"  [{partition.upper():5s}]  Total: {s['total']:5d}  |  "
            f"Berhasil: {s['success']:5d}  |  "
            f"Skip (korup): {s['skipped']:4d}  |  "
            f"Cache: {s['already_exists']:5d}"
        )
    print("-" * 60)
    print(
        f"  [TOTAL]  Berhasil: {total_success}  |  "
        f"Skip: {total_skipped}  |  Cache: {total_cached}"
    )
    print("=" * 60)
    print(f"[INFO] Output NPY tersimpan di: '{processed_base}'")


# ---------------------------------------------------------------------------
# Standalone Entry Point (opsional: python pipelines/preprocess.py --config ...)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Guard wajib untuk ProcessPoolExecutor di Windows dan macOS (spawn context).
    # Di Linux (fork context) ini opsional, tapi tetap best practice.
    import multiprocessing
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(
        description="PADL-FilamentSeg: Pra-pemrosesan FITS → NPY 512×512 (multi-core)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path ke file konfigurasi OmegaConf YAML (default: config.yaml)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Override jumlah worker process paralel. "
            "Default: config.system.workers (atau jumlah CPU jika tidak dikonfigurasi)."
        ),
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"[CRITICAL] File konfigurasi tidak ditemukan: '{args.config}'")
        sys.exit(1)

    cfg = OmegaConf.load(args.config)
    preprocess_routine(cfg, num_workers=args.workers)
