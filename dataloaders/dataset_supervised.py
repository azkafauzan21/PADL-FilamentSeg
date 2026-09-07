import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from pycocotools.coco import COCO

class FilamentSupervisedDataset(Dataset):
    def __init__(self, coco_json_path, image_dir, transform=None, config=None):
        """
        Dataset Hybrid Khusus COCO JPEG untuk Supervised Fine-Tuning Mask2Former.
        Meresolusi permasalahan anotasi ganda (data leakage/multiple labelers) 
        karena menggunakan hasil split anti-leakage dari 01_extract_metadata.py.
        
        Args:
            coco_json_path (str): Path ke file JSON (train_split.json atau val_split.json).
            image_dir (str): Direktori penyimpanan citra JPEG Kaggle.
            transform (albumentations.Compose): Pipeline augmentasi.
        """
        # COCO API digunakan langsung untuk menavigasi struktur JSON yang aman.
        # Catatan: Properti kustom dari Kaggle (seperti 'spine') tetap utuh 
        # di dalam struktur coco.anns dan bisa ditarik jika diperlukan.
        self.coco = COCO(coco_json_path)
        self.image_ids = list(self.coco.imgs.keys())
        self.image_dir = image_dir
        self.transform = transform
        self.config = config
        
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_metadata = self.coco.loadImgs(img_id)[0]
        
        # 1. Pemuatan Citra 8-bit JPEG
        path = os.path.join(self.image_dir, img_metadata['file_name'])
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise FileNotFoundError(f"[ERROR] Gambar fisik hilang di path: {path}")
            
        image = image.astype(np.float32)
        
        # Normalisasi Fisika (Disamakan persis dengan pipeline SSL FITS)
        p_upper = self.config.physics_parameters.percentile_clip_upper if self.config else 99
        p_lower = self.config.physics_parameters.percentile_clip_lower if self.config else 1
        
        p99 = np.percentile(image, p_upper)
        p1 = np.percentile(image, p_lower)
        image = np.clip(image, p1, p99)
        image = (image - p1) / (p99 - p1 + 1e-8)
        
        # 2. Penarikan Mask (Instances) via COCO
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)
        
        masks = []
        class_labels_list = []
        for ann in annotations:
            # Otomatis decode RLE / Polygon dari JSON menjadi numpy boolean array 2D
            mask = self.coco.annToMask(ann)
            masks.append(mask)
            class_labels_list.append(ann['category_id'])
            
        # 3. Transformasi Augmentasi (Bebas dari Larangan No RandomFlip jika mau)
        # Pada data JPEG H-Alpha Kaggle, orientasi seringkali standar (utara di atas),
        # namun untuk aman, terapkan sinkronisasi image & masks secara berbarengan.
        if self.transform is not None:
            if len(masks) > 0:
                augmented = self.transform(image=image, masks=masks)
                image = augmented['image']
                masks = augmented['masks']
            else:
                augmented = self.transform(image=image)
                image = augmented['image']
                
        # 4. Format Output List of Dictionaries
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)
            
        if image.ndim == 2:
            image = image.unsqueeze(0)
        elif image.ndim == 3 and image.shape[-1] == 1:
            image = image.permute(2, 0, 1)
            
        if len(masks) > 0:
            if not isinstance(masks[0], torch.Tensor):
                masks = [torch.from_numpy(m) for m in masks]
            mask_tensor = torch.stack(masks).to(torch.float32)
            class_labels = torch.tensor(class_labels_list, dtype=torch.long)
        else:
            mask_tensor = torch.empty((0, image.shape[1], image.shape[2]), dtype=torch.float32)
            class_labels = torch.empty((0,), dtype=torch.long)
            
        return {
            "pixel_values": image,
            "labels": {
                "class_labels": class_labels,
                "masks": mask_tensor
            }
        }
