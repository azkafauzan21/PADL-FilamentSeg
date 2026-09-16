import csv
import os

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_utils
from tqdm import tqdm

try:
    from models.segmenter_mask2former import FilamentMask2Former
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.segmenter_mask2former import FilamentMask2Former

from data_aug.transforms import get_inference_transform


# ---------------------------------------------------------------------------
# RLE Encoding
# ---------------------------------------------------------------------------

def encode_binary_mask_to_rle(binary_mask: np.ndarray) -> str:
    """
    Mengubah mask biner 2D menjadi RLE string sesuai format Kaggle MAGFiLO.

    Gunakan pycocotools (Fortran-order) agar konsisten dengan evaluator resmi
    kompetisi. String yang dikembalikan TIDAK mengandung tanda kutip — pandas
    akan disimpan dengan quoting=QUOTE_NONE untuk menjamin hal ini.
    """
    mask_fortran = np.asfortranarray(binary_mask.astype(np.uint8))
    encoded = mask_utils.encode(mask_fortran)
    rle_str = encoded["counts"]
    if isinstance(rle_str, bytes):
        rle_str = rle_str.decode("utf-8")
    return rle_str


# ---------------------------------------------------------------------------
# Instance Separation
# ---------------------------------------------------------------------------

def separate_instances(mask_logits: torch.Tensor, threshold: float = 0.5) -> list[np.ndarray]:
    """
    Pisahkan setiap query prediction menjadi mask biner individu.

    Args:
        mask_logits: Tensor [num_queries, H, W] — output sigmoid dari model.
        threshold:   Ambang batas untuk binarisasi.

    Returns:
        List of 2D np.ndarray boolean masks — satu elemen per instance aktif.
        List kosong jika tidak ada prediksi yang melampaui ambang batas.
    """
    probs = torch.sigmoid(mask_logits)                   # [Q, H, W]
    binary = (probs > threshold).cpu().numpy()           # [Q, H, W] bool

    instances: list[np.ndarray] = []
    for q_idx in range(binary.shape[0]):
        query_mask = binary[q_idx]                       # [H, W]
        if query_mask.any():
            instances.append(query_mask)

    return instances


# ---------------------------------------------------------------------------
# Main Inference Routine
# ---------------------------------------------------------------------------

def inference_routine(config):
    """
    Rutinitas inferensi dan pembentukan CSV submission Kaggle MAGFiLO.

    Format CSV yang Dihasilkan:
        Header  : filament_id,segmentation_rle
        Baris   : Satu baris per instance filamen yang terdeteksi.
                  filament_id = "{image_stem}_{instance_idx}" (1-indexed).
        Encoding: RLE pycocotools (Fortran-order, tanpa tanda kutip).

    Aturan Kritis:
        HANYA membaca file JPEG dari partisi test.
        File FITS TIDAK BOLEH diproses di sini.
    """
    print("[INFO] Memulai rutin Inferensi dan pembentukan Submission Kaggle...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Muat model terlatih
    # ------------------------------------------------------------------
    model = FilamentMask2Former(num_classes=2)
    # num_classes=2: cocok dengan state_dict dari training (Body + Spine).
    # Perubahan dari num_classes=1 menghilangkan RuntimeError size mismatch
    # saat memuat checkpoint yang dilatih dengan skema 2-kelas.

    # Prioritaskan best checkpoint (smart checkpoint dari train_mask2former.py)
    best_weights_path = os.path.join(config.system.weights_dir, "best_mask2former.pth")
    final_weights_path = os.path.join(config.system.weights_dir, "mask2former_final.pth")

    if os.path.exists(best_weights_path):
        weights_path = best_weights_path
        print(f"[INFO] Menggunakan best checkpoint: {weights_path}")
    elif os.path.exists(final_weights_path):
        weights_path = final_weights_path
        print(f"[INFO] best_mask2former.pth tidak ada; fallback ke: {weights_path}")
    else:
        print(
            f"[ERROR] Bobot model tidak ditemukan di '{config.system.weights_dir}'. "
            "Pastikan training selesai. Inferensi dibatalkan."
        )
        return

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # Persiapkan direktori test
    # ------------------------------------------------------------------
    test_dir = config.system.jpeg_test_dir  # → "./data/raw/MAGFiLO_.../test/test_images/"

    if not os.path.exists(test_dir):
        print(
            f"[ERROR] Direktori test tidak ditemukan: '{test_dir}'. "
            f"Periksa nilai 'jpeg_test_dir' di config.yaml. Inferensi dibatalkan."
        )
        return

    # Hanya baca JPEG — filter ini memastikan file FITS tidak tersedot.
    test_images = [
        f for f in os.listdir(test_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    test_images.sort()
    print(f"[INFO] Ditemukan {len(test_images)} citra JPEG pada partisi test.")

    if len(test_images) == 0:
        print("[INFO] Tidak ada citra ditemukan. Inferensi dihentikan.")
        return

    # ------------------------------------------------------------------
    # Inferensi Loop
    # ------------------------------------------------------------------
    # Setiap elemen: dict dengan key "filament_id" dan "segmentation_rle"
    results: list[dict] = []

    with torch.no_grad():
        for fname in tqdm(test_images, desc="Menghasilkan Inferensi"):
            img_path = os.path.join(test_dir, fname)
            image_stem = os.path.splitext(fname)[0]

            # FIX BUG-06: Guard eksplisit untuk cv2.imread() yang mengembalikan None
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"[WARN] Gagal membaca: '{img_path}'. File mungkin corrupt. Dilewati.")
                # Tambahkan baris kosong agar gambar tetap terwakili dalam CSV
                results.append({"filament_id": f"{image_stem}_0", "segmentation_rle": ""})
                continue

            image = image.astype(np.float32)

            # Normalisasi persentil — parameter dari config (konsisten dengan training)
            p_lo = config.physics_parameters.percentile_clip_lower
            p_hi = config.physics_parameters.percentile_clip_upper
            sat_threshold = float(
                config.data_augmentation.inference.saturation_threshold
            )
            lo = np.percentile(image, p_lo)
            hi = np.percentile(image, p_hi)

            if (hi - lo) < sat_threshold:
                # Gambar hampir konstan — set ke nol
                image = np.zeros_like(image, dtype=np.float32)
            else:
                image = np.clip(image, lo, hi)
                image = (image - lo) / (hi - lo)

            tensor_img = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
            # tensor_img shape: [1, 1, H, W]

            mask_logits, class_logits = model(pixel_values=tensor_img)

            # mask_logits  shape: [1, num_queries, H', W']
            # class_logits shape: [1, num_queries, num_classes+1] (termasuk kelas 'no-object')
            mask_logits  = mask_logits.squeeze(0)   # → [num_queries, H', W']
            class_logits = class_logits.squeeze(0)  # → [num_queries, num_classes+1]

            # Prediksi kelas per query: argmax di kelas valid (abaikan indeks 'no-object' terakhir)
            pred_labels = class_logits[:, :-1].argmax(dim=-1)  # [num_queries], nilai: {0, 1}

            # Pisahkan instance: satu baris CSV per filamen yang terdeteksi
            # mask_threshold dibaca dari config — turunkan untuk recall lebih tinggi
            mask_threshold = float(config.data_augmentation.inference.mask_threshold)
            instances = separate_instances(mask_logits, threshold=mask_threshold)

            if instances:
                inst_csv_idx = 0
                for q_idx, inst_mask in enumerate(instances):
                    # STEALTH FILTER: hanya simpan mask Kelas 0 (Filament Body)
                    # Kelas 1 (Spine) tidak dinilai oleh Kaggle → dibuang secara diam-diam.
                    q_label = int(pred_labels[q_idx].item()) if q_idx < len(pred_labels) else 0
                    if q_label != 0:
                        continue

                    inst_csv_idx += 1
                    rle_str = encode_binary_mask_to_rle(inst_mask)
                    results.append({
                        "filament_id": f"{image_stem}_{inst_csv_idx}",
                        "segmentation_rle": rle_str,
                    })

                if inst_csv_idx == 0:
                    # Semua query diprediksi sebagai Spine → tidak ada body
                    results.append({
                        "filament_id": f"{image_stem}_0",
                        "segmentation_rle": "",
                    })
            else:
                # Tidak ada prediksi aktif — baris kosong wajib tetap ada
                results.append({
                    "filament_id": f"{image_stem}_0",
                    "segmentation_rle": "",
                })

    # ------------------------------------------------------------------
    # Simpan Submission CSV
    #
    # Header  : filament_id,segmentation_rle   (sesuai aturan Kaggle MAGFiLO)
    # Quoting : QUOTE_NONE + escapechar='\\'    → RLE string bebas kutip
    # ------------------------------------------------------------------
    df = pd.DataFrame(results, columns=["filament_id", "segmentation_rle"])
    submission_path = os.path.join(config.system.data_dir, "submission.csv")

    df.to_csv(
        submission_path,
        index=False,
        quoting=csv.QUOTE_NONE,
        escapechar="\\",
    )

    n_predicted = (df["segmentation_rle"] != "").sum()
    n_total = len(df)
    print(
        f"[INFO] Inferensi selesai. {n_predicted}/{n_total} baris memiliki prediksi segmentasi. "
        f"Submission disimpan ke: '{submission_path}'"
    )
