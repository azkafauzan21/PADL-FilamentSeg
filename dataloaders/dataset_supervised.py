import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO


class FilamentSupervisedDataset(Dataset):
    """
    Dataset Hybrid Multi-Task untuk Supervised Fine-Tuning Mask2Former.

    ──────────────────────────────────────────────────────────────────────
    Skema Kelas (2 Kelas):
    ──────────────────────────────────────────────────────────────────────
      Kelas 0 (Filament Body):
          Semua anotasi dari JSON (category_id = 1, 2, 3, atau 4) dilebur
          menjadi SATU kelas tunggal (class_id = 0 dalam format 0-indexed
          yang digunakan PyTorch / Mask2Former).
          Alasan: Perbedaan category_id asli merepresentasikan subkategori
          kiralitas (sinistral/dextral) dan orientasi; untuk evaluasi Kaggle
          yang hanya menilai luasan filamen, semua subkategori disamakan.

      Kelas 1 (Filament Spine):
          Koordinat dari key "spine" (list of [x, y] points) dalam setiap
          anotasi di-render menjadi mask binary menggunakan cv2.polylines
          (thickness=2). Ini merepresentasikan sumbu magnetik utama filamen
          (Polarity Inversion Line / PIL).

    ──────────────────────────────────────────────────────────────────────
    Normalisasi:
    ──────────────────────────────────────────────────────────────────────
      Identik dengan pipeline SSL (percentile clip + normalize ke [0,1]).
      Guard saturasi menggunakan threshold 1e-3 (konsisten dengan SSL).
    """

    def __init__(self, coco_json_path, image_dir, transform=None, config=None):
        """
        Args:
            coco_json_path (str): Path ke file JSON (train_split.json atau val_split.json).
            image_dir (str):      Direktori penyimpanan citra JPEG Kaggle.
            transform (albumentations.Compose): Pipeline augmentasi.
            config (OmegaConf):  Konfigurasi sistem.
        """
        # COCO API digunakan untuk navigasi JSON yang aman.
        # Properti kustom Kaggle (termasuk 'spine') tetap utuh di coco.anns.
        self.coco = COCO(coco_json_path)
        self.image_ids = list(self.coco.imgs.keys())
        self.image_dir = image_dir
        self.transform = transform
        self.config = config

    def __len__(self):
        return len(self.image_ids)

    # ------------------------------------------------------------------
    # Normalisasi Persentil (Konsisten dengan SolarSSLDataset)
    # ------------------------------------------------------------------
    def _percentile_normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Kliping persentil [P_lower, P_upper] → normalisasi [0.0, 1.0].

        Guard saturasi menggunakan threshold dari config.data_augmentation.inference
        (default 1e-3 jika key tidak tersedia).
        """
        p_lower = self.config.physics_parameters.percentile_clip_lower if self.config else 1
        p_upper = self.config.physics_parameters.percentile_clip_upper if self.config else 99

        # Baca threshold saturasi dari config — konsisten dengan inference pipeline
        sat_threshold = 1e-3
        if self.config:
            try:
                sat_threshold = float(self.config.data_augmentation.inference.saturation_threshold)
            except Exception:
                pass  # Fallback ke 1e-3 jika key belum ada di config lama

        lo = np.percentile(image, p_lower)
        hi = np.percentile(image, p_upper)

        # Guard: gambar hampir konstan (saturasi sensor)
        if (hi - lo) < sat_threshold:
            return np.zeros_like(image, dtype=np.float32)

        image = np.clip(image, lo, hi)
        image = (image - lo) / (hi - lo)
        return image.astype(np.float32)

    # ------------------------------------------------------------------
    # Render Spine ke Mask Binary
    # ------------------------------------------------------------------
    @staticmethod
    def _render_spine_mask(spine_coords, h: int, w: int, thickness: int = 2) -> np.ndarray:
        """
        Render koordinat spine (list of [x, y]) menjadi mask binary 2D.

        Args:
            spine_coords: List of [x, y] pairs (koordinat piksel).
            h, w:         Dimensi gambar target (height, width).
            thickness:    Ketebalan garis cv2.polylines() dalam piksel.
                          Dibaca dari config.data_augmentation.spine.polyline_thickness.
                          Default: 2.

        Returns:
            np.ndarray shape (h, w), dtype uint8, nilai {0, 1}.
            Mengembalikan array nol jika spine_coords kurang dari 2 titik.
        """
        mask = np.zeros((h, w), dtype=np.uint8)

        if not spine_coords or len(spine_coords) < 2:
            return mask

        # Konversi ke format yang dibutuhkan cv2.polylines: array int32 [N, 1, 2]
        try:
            pts = np.array(spine_coords, dtype=np.int32).reshape((-1, 1, 2))
        except (ValueError, TypeError):
            return mask

        cv2.polylines(mask, [pts], isClosed=False, color=1, thickness=thickness)
        return mask

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_metadata = self.coco.loadImgs([img_id])[0]
        h_orig = img_metadata.get('height', None)
        w_orig = img_metadata.get('width', None)

        # 1. Pemuatan Citra 8-bit JPEG
        path = os.path.join(self.image_dir, img_metadata['file_name'])
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"[ERROR] Gambar fisik hilang di path: {path}")

        h_img, w_img = image.shape[:2]
        # Gunakan dimensi aktual gambar jika metadata tidak tersedia
        if h_orig is None:
            h_orig, w_orig = h_img, w_img

        image = image.astype(np.float32)

        # Normalisasi Fisika (konsisten dengan pipeline SSL)
        image = self._percentile_normalize(image)

        # 2. Penarikan Anotasi via COCO API
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        annotations = self.coco.loadAnns(ann_ids)

        body_masks = []   # Kelas 0: luasan tubuh filamen (class_id=0)
        spine_masks = []  # Kelas 1: sumbu spine (class_id=1)

        for ann in annotations:
            # --- Kelas 0: Filament Body ---
            # Lebur SEMUA category_id (1,2,3,4) → class_id 0
            body_mask = self.coco.annToMask(ann)  # [H, W], dtype uint8
            body_masks.append((body_mask, 0))     # (mask, class_id)

            # --- Kelas 1: Filament Spine ---
            # Key 'spine' berisi list of [x,y] points (koordinat piksel)
            # Ketebalan garis dibaca dari config; fallback ke 2 jika key belum ada.
            spine_thickness = 2
            if self.config:
                try:
                    spine_thickness = int(self.config.data_augmentation.spine.polyline_thickness)
                except Exception:
                    pass

            spine_coords = ann.get('spine', None)
            if spine_coords:
                spine_mask = self._render_spine_mask(
                    spine_coords, h_orig, w_orig, thickness=spine_thickness
                )
                if spine_mask.sum() > 0:
                    spine_masks.append((spine_mask, 1))  # (mask, class_id)

        # Gabungkan semua mask dan label
        all_masks = body_masks + spine_masks

        # 3. Transformasi Augmentasi
        # LARANGAN FISIKA: Tidak ada RandomHorizontalFlip atau RandomVerticalFlip
        # (melanggar chiralitas magnetik filamen matahari).
        if self.transform is not None and len(all_masks) > 0:
            raw_masks = [m for m, _ in all_masks]
            labels_list = [lbl for _, lbl in all_masks]

            augmented = self.transform(image=image, masks=raw_masks)
            image = augmented['image']
            aug_masks = augmented['masks']

            # Filter ghost mask (mask menjadi kosong setelah transform resize/crop)
            valid_masks = []
            valid_labels = []
            for m, label in zip(aug_masks, labels_list):
                m_bin = (m > 0).astype(np.uint8)
                if m_bin.sum() > 0:
                    valid_masks.append(m_bin)
                    valid_labels.append(label)

            all_masks_filtered = list(zip(valid_masks, valid_labels))
        elif self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented['image']
            all_masks_filtered = []
        else:
            all_masks_filtered = [(m, lbl) for m, lbl in all_masks]

        # 4. Format Output Tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)

        if image.ndim == 2:
            image = image.unsqueeze(0)           # [H, W] → [1, H, W]
        elif image.ndim == 3 and image.shape[-1] == 1:
            image = image.permute(2, 0, 1)       # [H, W, 1] → [1, H, W]

        if len(all_masks_filtered) > 0:
            final_masks = [m for m, _ in all_masks_filtered]
            final_labels = [lbl for _, lbl in all_masks_filtered]

            if not isinstance(final_masks[0], torch.Tensor):
                final_masks = [torch.from_numpy(m) for m in final_masks]

            mask_tensor = torch.stack(final_masks).to(torch.float32)  # [N, H, W]
            class_labels = torch.tensor(final_labels, dtype=torch.long)
        else:
            # Gambar tanpa anotasi (atau semua mask hilang setelah augmentasi)
            mask_tensor = torch.empty(
                (0, image.shape[1], image.shape[2]), dtype=torch.float32
            )
            class_labels = torch.empty((0,), dtype=torch.long)

        return {
            "pixel_values": image,
            "labels": {
                "class_labels": class_labels,
                "masks": mask_tensor,       # key 'masks' digunakan oleh custom_collate_fn
            }
        }
