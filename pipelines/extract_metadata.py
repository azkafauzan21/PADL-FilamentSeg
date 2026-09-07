import json
import os
import re

import pandas as pd
from sklearn.model_selection import train_test_split


def extract_data_routine(config):
    """
    Rutinitas ekstraksi metadata dan pembuatan Anti-Leakage Train/Val split.

    Menghasilkan:
        - download_fits_targets.csv  : Target FITS untuk SSL (untuk downloader opsional)
        - train_split.json           : Subset COCO JSON untuk training (80%)
        - val_split.json             : Subset COCO JSON untuk validasi (20%)

    Strategi anti-leakage:
        Split dilakukan berdasarkan 'file_name' (gambar fisik unik), bukan 'image_id'.
        Satu file fisik bisa memiliki banyak anotasi. Memisahkan berdasarkan 'image_id'
        bisa membocorkan informasi dari gambar yang sama ke dua set berbeda.
    """
    print("[INFO] Memulai rutin ekstraksi metadata dan Anti-Leakage Train/Val split...")

    # ------------------------------------------------------------------
    # FIX BUG-01: Gunakan config.system.coco_annotation_path
    # (menggantikan os.path.join(data_dir, "annotations.json") yang salah)
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
    # ------------------------------------------------------------------
    unique_filenames = list(set([img["file_name"] for img in images_list]))
    print(f"[INFO] Ditemukan {len(unique_filenames)} gambar unik fisik (berdasarkan file_name).")

    fits_targets = []
    # Asumsi format JPEG Kaggle: misal 20120506124500_ha_ma.jpg
    for fname in unique_filenames:
        ts_match = re.search(r"(\d{14})", fname)
        st_match = re.search(r"_ha_([a-z]{2})\.", fname)

        timestamp = ts_match.group(1) if ts_match else "UNKNOWN"
        station_code = st_match.group(1) if st_match else "UNKNOWN"

        fits_targets.append({
            "file_name": fname,
            "timestamp": timestamp,
            "station_code": station_code,
        })

    df_targets = pd.DataFrame(fits_targets)
    targets_csv_path = os.path.join(config.system.data_dir, "download_fits_targets.csv")
    df_targets.to_csv(targets_csv_path, index=False)
    print(f"[INFO] Target FITS diekspor ke: {targets_csv_path}")

    # ------------------------------------------------------------------
    # TUGAS B: Anti-Leakage Train/Val Split (80/20)
    # ------------------------------------------------------------------
    train_files, val_files = train_test_split(
        unique_filenames,
        test_size=config.physics_parameters.test_size_split,
        random_state=config.system.seed,
    )

    train_image_ids = set([img["id"] for img in images_list if img["file_name"] in train_files])
    val_image_ids = set([img["id"] for img in images_list if img["file_name"] in val_files])

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
    # FIX BUG-01: Gunakan config.system.train_split_json dan val_split_json
    # (menggantikan os.path.join(data_dir, "train_split.json") yang mungkin
    # sudah cocok, tapi sekarang dikontrol secara eksplisit dari config)
    # ------------------------------------------------------------------
    train_out_path = config.system.train_split_json   # → "./data/train_split.json"
    val_out_path = config.system.val_split_json       # → "./data/val_split.json"

    # Pastikan direktori output ada
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
