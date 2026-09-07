import json
import os
import re

import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Helper: ekstraksi metadata dari file_name
# ---------------------------------------------------------------------------

def _parse_filename_metadata(fname: str) -> dict:
    """
    Ekstraksi timestamp dan kode stasiun dari nama file JPEG/FITS MAGFiLO.

    Format nama file aktual (hasil audit 2026-09-07):
        {YYYYMMDDHHmmss}{KodeStasiun}h.{ext}
        Contoh: 20120506124500Ch.jpeg

    - Timestamp : 14 digit desimal pertama → YYYYMMDDHHmmss
    - Kode stasiun : 1 huruf kapital tepat sebelum 'h.' (L/M/C/B/T/U)

    Bug lama (diperbaiki):
        Pattern lama `_ha_([a-z]{2})\.` tidak pernah cocok dengan format aktual
        karena tidak ada substring '_ha_' dalam nama file MAGFiLO.
    """
    ts_match = re.search(r"(\d{14})", fname)
    st_match = re.search(r"([A-Z])h\.", fname)

    timestamp = ts_match.group(1) if ts_match else "UNKNOWN"
    station_code = st_match.group(1) if st_match else "UNKNOWN"
    year = timestamp[:4] if timestamp != "UNKNOWN" else "UNKNOWN"

    return {
        "file_name": fname,
        "timestamp": timestamp,
        "year": year,
        "station_code": station_code,
    }


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------

def extract_data_routine(config):
    """
    Rutinitas ekstraksi metadata dan pembuatan Anti-Leakage Train/Val split.

    Perbaikan dari versi sebelumnya:
        1. [BUG FIX] Regex stasiun diperbarui dari `_ha_([a-z]{2})\\.` menjadi
           `([A-Z])h\\.` agar sesuai format nama file aktual MAGFiLO.
        2. [ENHANCEMENT] Stratifikasi temporal ditambahkan ke `train_test_split`
           menggunakan kolom 'year' (4 digit) agar representasi siklus matahari
           (Solar Cycle 24–25) terdistribusi seimbang antara Train dan Val.

    Menghasilkan:
        - download_fits_targets.csv : Target FITS untuk SSL (downloader opsional)
        - train_split.json          : Subset COCO JSON untuk training (80%)
        - val_split.json            : Subset COCO JSON untuk validasi (20%)

    Strategi anti-leakage:
        Split dilakukan berdasarkan 'file_name' (gambar fisik unik), bukan 'image_id'.
        Satu file fisik bisa memiliki banyak anotasi. Memisahkan berdasarkan 'image_id'
        bisa membocorkan informasi dari gambar yang sama ke dua set berbeda.
    """
    print("[INFO] Memulai rutin ekstraksi metadata dan Anti-Leakage Train/Val split...")

    # ------------------------------------------------------------------
    # FIX BUG-01: Gunakan config.system.coco_annotation_path
    # ------------------------------------------------------------------
    annotations_path = config.system.coco_annotation_path
    if not os.path.exists(annotations_path):
        print(
            f"[ERROR] File anotasi COCO tidak ditemukan: '{annotations_path}'. "
            f"Periksa nilai 'coco_annotation_path' di config.yaml."
        )
        return

    print(f"[INFO] Memuat anotasi COCO dari: {annotations_path}")
    with open(annotations_path, "r") as f:
        coco_data = json.load(f)

    images_list = coco_data.get("images", [])
    annotations_list = coco_data.get("annotations", [])
    print(
        f"[INFO] Memuat {len(images_list)} gambar dan "
        f"{len(annotations_list)} anotasi dari file COCO."
    )

    # ------------------------------------------------------------------
    # TUGAS A: Ekstraksi target FITS untuk SSL downloader (opsional)
    #
    # PERBAIKAN BUG REGEX:
    #   Lama : re.search(r"_ha_([a-z]{2})\.", fname)  → selalu None
    #   Baru  : re.search(r"([A-Z])h\.", fname)        → menangkap L/M/C/B/T/U
    # ------------------------------------------------------------------
    unique_filenames = list(set([img["file_name"] for img in images_list]))
    unique_filenames.sort()  # Urutan deterministik untuk reproduktibilitas
    print(f"[INFO] Ditemukan {len(unique_filenames)} gambar unik fisik (berdasarkan file_name).")

    fits_targets = [_parse_filename_metadata(fname) for fname in unique_filenames]

    df_targets = pd.DataFrame(fits_targets)
    targets_csv_path = os.path.join(config.system.data_dir, "download_fits_targets.csv")
    os.makedirs(config.system.data_dir, exist_ok=True)
    df_targets.to_csv(targets_csv_path, index=False)

    # Laporan distribusi stasiun dan tahun
    station_counts = df_targets["station_code"].value_counts().to_dict()
    year_counts = df_targets["year"].value_counts().sort_index().to_dict()
    unknown_stations = df_targets[df_targets["station_code"] == "UNKNOWN"].shape[0]

    print(f"[INFO] Target FITS diekspor ke: {targets_csv_path}")
    print(f"       Distribusi Stasiun: {station_counts}")
    print(f"       Distribusi Tahun  : {year_counts}")
    if unknown_stations > 0:
        print(f"[WARN] {unknown_stations} file tidak berhasil diekstrak kode stasiunnya.")

    # ------------------------------------------------------------------
    # TUGAS B: Anti-Leakage Train/Val Split (80/20) dengan STRATIFIKASI TEMPORAL
    #
    # PERBAIKAN STRATIFIKASI:
    #   Sebelumnya: train_test_split tanpa stratify → distribusi tahun acak
    #   Sekarang  : stratify=year_labels → setiap tahun direpresentasikan
    #               secara proporsional di Train dan Val, memastikan cakupan
    #               Solar Cycle 24 (2008–2019) dan Cycle 25 (2019+) yang seimbang.
    #
    # Catatan: scikit-learn membutuhkan minimal 2 sampel per kelas untuk stratifikasi.
    # Tahun dengan hanya 1 file akan menyebabkan error. Guard di bawah menangani ini
    # dengan fallback ke split tanpa stratifikasi jika ada tahun singleton.
    # ------------------------------------------------------------------
    year_labels = df_targets.set_index("file_name")["year"].to_dict()
    stratify_labels = [year_labels.get(fname, "UNKNOWN") for fname in unique_filenames]

    # Hitung kelas dengan sampel terlalu sedikit untuk stratifikasi
    from collections import Counter
    label_counts = Counter(stratify_labels)
    singleton_years = [yr for yr, cnt in label_counts.items() if cnt < 2]

    if singleton_years:
        print(
            f"[WARN] Tahun berikut memiliki < 2 sampel dan tidak dapat distratifikasi: "
            f"{singleton_years}. Fallback ke split tanpa stratifikasi."
        )
        stratify_arg = None
    else:
        stratify_arg = stratify_labels
        print(f"[INFO] Stratifikasi temporal aktif berdasarkan tahun: {sorted(label_counts.keys())}")

    train_files, val_files = train_test_split(
        unique_filenames,
        test_size=config.physics_parameters.test_size_split,
        random_state=config.system.seed,
        stratify=stratify_arg,
    )

    train_files_set = set(train_files)
    val_files_set = set(val_files)

    train_image_ids = set([img["id"] for img in images_list if img["file_name"] in train_files_set])
    val_image_ids = set([img["id"] for img in images_list if img["file_name"] in val_files_set])

    def create_split_json(target_image_ids: set) -> dict:
        """
        Buat subset COCO JSON dengan mempertahankan semua metadata header
        (info, licenses, categories) dan memfilter images & annotations.
        """
        return {
            "info": coco_data.get("info", {}),
            "licenses": coco_data.get("licenses", []),
            "categories": coco_data.get("categories", []),
            "images": [img for img in images_list if img["id"] in target_image_ids],
            "annotations": [
                ann for ann in annotations_list if ann["image_id"] in target_image_ids
            ],
        }

    train_split = create_split_json(train_image_ids)
    val_split = create_split_json(val_image_ids)

    # ------------------------------------------------------------------
    # Tulis output ke disk
    # ------------------------------------------------------------------
    train_out_path = config.system.train_split_json   # → "./data/train_split.json"
    val_out_path = config.system.val_split_json       # → "./data/val_split.json"

    os.makedirs(os.path.dirname(os.path.abspath(train_out_path)), exist_ok=True)

    with open(train_out_path, "w") as f:
        json.dump(train_split, f)
    with open(val_out_path, "w") as f:
        json.dump(val_split, f)

    print(f"[INFO] Anti-Leakage Train/Val Split Selesai.")
    print(
        f"       Train: {len(train_files)} file fisik, "
        f"{len(train_split['annotations'])} anotasi → '{train_out_path}'"
    )
    print(
        f"       Val  : {len(val_files)} file fisik, "
        f"{len(val_split['annotations'])} anotasi → '{val_out_path}'"
    )

    # Laporan distribusi tahun pada hasil split
    if stratify_arg is not None:
        train_year_dist = Counter([year_labels.get(f, "?") for f in train_files])
        val_year_dist = Counter([year_labels.get(f, "?") for f in val_files])
        print(f"       Distribusi Tahun Train: {dict(sorted(train_year_dist.items()))}")
        print(f"       Distribusi Tahun Val  : {dict(sorted(val_year_dist.items()))}")
