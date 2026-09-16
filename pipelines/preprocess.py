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
from typing import Literal, Optional, Tuple  # Tuple & Optional: kompatibel Python 3.8+

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

def _worker_process_file(fits_path: str, npy_path: str) -> Tuple[str, str]:
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

    # --- Baca FITS: Data di HDU[1], Header Disk di HDU[1] ---
    # DIVERIFIKASI dari inspeksi header aktual (2026-09-15):
    #   HDU[0] = Primary header (NAXIS=0, tidak ada data piksel)
    #   HDU[1] = Image Extension (NAXIS1=2048, NAXIS2=2048, berisi data)
    #
    # Keyword disk yang digunakan (HDU[1]):
    #   CRPIX1 / CRPIX2  → pusat disk dalam piksel (1-indexed, FITS convention)
    #   RADIUS           → radius disk dalam piksel (post-LMBCOR, paling akurat)
    #   Fallback: FNDLMBXC, FNDLMBYC, FNDLMBMI (jika CRPIX tidak tersedia)
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", AstropyWarning)

            with fits.open(fits_path, memmap=False) as hdul:
                data = None
                disk_cx: Optional[float] = None
                disk_cy: Optional[float] = None
                disk_r:  Optional[float] = None

                for i, hdu in enumerate(hdul):
                    if hdu.data is not None and np.ndim(hdu.data) == 2:
                        data = hdu.data
                        hdr  = hdu.header

                        # ── Ekstraksi pusat disk (FITS 1-indexed → Python 0-indexed) ──
                        crpix1 = hdr.get("CRPIX1", None)
                        crpix2 = hdr.get("CRPIX2", None)
                        if crpix1 is not None and crpix2 is not None:
                            # FITS convention: CRPIX adalah 1-indexed
                            disk_cx = float(crpix1) - 1.0
                            disk_cy = float(crpix2) - 1.0
                        else:
                            # Fallback: FNDLMBXC/YC (geometric limb center, juga 1-indexed)
                            fndx = hdr.get("FNDLMBXC", None)
                            fndy = hdr.get("FNDLMBYC", None)
                            if fndx is not None and fndy is not None:
                                disk_cx = float(fndx) - 1.0
                                disk_cy = float(fndy) - 1.0

                        # ── Ekstraksi radius disk (dalam piksel) ──
                        radius_kw = hdr.get("RADIUS", None)
                        if radius_kw is not None and float(radius_kw) > 0:
                            disk_r = float(radius_kw)
                        else:
                            # Fallback: FNDLMBMI (semi-minor axis, lebih konservatif)
                            fndmi = hdr.get("FNDLMBMI", None)
                            if fndmi is not None and float(fndmi) > 0:
                                disk_r = float(fndmi)

                        break  # Ambil HDU pertama yang memiliki data 2D

        if caught_warnings:
            msg = str(caught_warnings[0].message)[:100]
            return STATUS_SKIP_CORRUPT, msg

        if data is None:
            return STATUS_SKIP_NO_DATA, "Tidak ada data 2D valid"

    except Exception as exc:
        return STATUS_SKIP_ERROR, str(exc)[:100]

    # --- Konversi dtype, bersihkan NaN/Inf ---
    data = data.astype(np.float32)
    if not np.all(np.isfinite(data)):
        data = np.where(np.isfinite(data), data, 0.0)

    h_orig, w_orig = data.shape

    # --- Solar Disk Crop (Fisika: buang ruang angkasa kosong) ---
    # Jika header disk berhasil diekstrak, lakukan crop bounding square.
    # Jika tidak (header korup/hilang), fallback ke pipeline lama (resize langsung).
    if disk_cx is not None and disk_cy is not None and disk_r is not None and disk_r > 0:
        # Bounding box bujur sangkar di sekitar disk matahari
        # Margin kecil (+0.05 * r) untuk memastikan seluruh limb tertangkap
        margin = disk_r * 0.05
        r_padded = disk_r + margin

        x0 = disk_cx - r_padded
        y0 = disk_cy - r_padded
        x1 = disk_cx + r_padded
        y1 = disk_cy + r_padded

        # ── Edge-case guard: clipping + zero-padding ──
        # Jika bounding box melewati tepi gambar (matahari tidak di tengah),
        # kita clip ke batas gambar dan pad sisi yang terpotong dengan nol.
        #
        # Contoh: GTCLMBXC=1026.92 → matahari sedikit bergeser ke kanan.
        # Tanpa guard ini, slice negatif atau > dimensi akan melempar IndexError.
        clip_x0 = max(0, int(round(x0)))
        clip_y0 = max(0, int(round(y0)))
        clip_x1 = min(w_orig, int(round(x1)))
        clip_y1 = min(h_orig, int(round(y1)))

        # Lebar/tinggi target (selalu simetris berdasarkan r_padded)
        target_side = int(round(2 * r_padded))
        if target_side < 1:
            target_side = 1

        # Potong area valid
        crop = data[clip_y0:clip_y1, clip_x0:clip_x1]

        # Hitung offset padding jika bounding box terpotong di salah satu sisi
        pad_left = max(0, clip_x0 - int(round(x0)))
        pad_top  = max(0, clip_y0 - int(round(y0)))

        # Alokasikan canvas bujur sangkar dan salin crop ke dalamnya
        canvas = np.zeros((target_side, target_side), dtype=np.float32)
        crop_h, crop_w = crop.shape
        # Guard tambahan: pastikan crop tidak melebihi canvas
        end_row = min(pad_top  + crop_h, target_side)
        end_col = min(pad_left + crop_w, target_side)
        src_h   = end_row - pad_top
        src_w   = end_col - pad_left
        canvas[pad_top:end_row, pad_left:end_col] = crop[:src_h, :src_w]

        data_to_resize = canvas
    else:
        # Fallback: tidak ada informasi disk → resize seluruh gambar
        data_to_resize = data

    # --- Resize ke TARGET_SIZE (flux-conserving: INTER_AREA) ---
    data_final = cv2.resize(data_to_resize, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    # --- Simpan ---
    np.save(npy_path, data_final.astype(np.float32))

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
