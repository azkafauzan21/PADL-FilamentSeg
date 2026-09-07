import warnings

import numpy as np
import torch
from astropy.io import fits
from torch.utils.data import Dataset


class SolarSSLDataset(Dataset):
    def __init__(self, fits_paths, transform=None, config=None):
        """
        Dataset khusus FITS untuk pra-pelatihan SSL SimCLR.

        Args:
            fits_paths (list): List path file .fits dari partisi TRAIN.
                               JANGAN menyertakan file dari partisi test.
            transform (albumentations.Compose): Transformasi fisika
                                                (wajib bebas dari Flip/Rotation).
            config (OmegaConf): Konfigurasi sistem.
        """
        self.fits_paths = fits_paths
        self.transform = transform
        self.config = config

    def __len__(self):
        return len(self.fits_paths)

    # ------------------------------------------------------------------
    # FIX BUG-02: Pembacaan HDU yang robust untuk format fpack terkompresi
    # ------------------------------------------------------------------
    def _load_fits_image(self, path: str) -> np.ndarray:
        """
        Membaca file FITS dengan dukungan penuh untuk format fpack terkompresi.

        Untuk FITS fpack (GONG): data ada di HDU[1] (CompImageHDU).
        Untuk FITS standar: data ada di HDU[0] (PrimaryHDU).

        Strategi iterasi: cari HDU pertama yang memiliki data array 2D valid,
        tanpa mengandalkan indeks hardcode yang bisa salah.
        """
        with fits.open(path, memmap=False) as hdul:
            data = None
            for hdu in hdul:
                # Cek eksplisit: data ada dan merupakan array 2D (bukan header/tabel)
                if hdu.data is not None and np.ndim(hdu.data) == 2:
                    data = hdu.data
                    break

            if data is None:
                raise ValueError(
                    f"[SolarSSLDataset] Tidak ada data gambar 2D yang valid "
                    f"di file FITS: {path}"
                )

        # Astropy otomatis menerapkan BZERO/BSCALE saat open, sehingga
        # data sudah dalam skala fisik. Konversi ke float32 untuk kompatibilitas PyTorch.
        return data.astype(np.float32)

    # ------------------------------------------------------------------
    # FIX BUG-03: Normalisasi persentil dengan validasi rentang
    # ------------------------------------------------------------------
    def _percentile_normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalisasi kliping persentil [P_lower, P_upper] → [0.0, 1.0].

        Termasuk guard untuk gambar hampir-konstan (misal akibat sensor saturated
        atau file corrupt) yang dapat menyebabkan silent near-zero division.
        """
        p_lower = (
            self.config.physics_parameters.percentile_clip_lower
            if self.config else 1
        )
        p_upper = (
            self.config.physics_parameters.percentile_clip_upper
            if self.config else 99
        )

        lo = np.percentile(image, p_lower)
        hi = np.percentile(image, p_upper)

        # Guard: Jika rentang terlalu kecil, gambar kemungkinan corrupt/saturated.
        # Mengembalikan tensor nol lebih aman daripada membiarkan near-zero division
        # menghasilkan tensor bernilai ~1.0 yang tidak informatif.
        if (hi - lo) < 1e-3:
            warnings.warn(
                f"[SolarSSLDataset] Rentang persentil sangat kecil "
                f"(P{p_upper} - P{p_lower} = {hi - lo:.6f}). "
                f"Gambar mungkin saturated atau corrupt. Mengembalikan tensor nol.",
                stacklevel=2,
            )
            return np.zeros_like(image, dtype=np.float32)

        image = np.clip(image, lo, hi)
        image = (image - lo) / (hi - lo)
        return image.astype(np.float32)

    def __getitem__(self, idx):
        path = self.fits_paths[idx]

        # 1. Baca FITS dengan loader yang robust (FIX BUG-02)
        image = self._load_fits_image(path)

        # 2. Normalisasi fisika dengan validasi rentang (FIX BUG-03)
        #    Membuang cosmic rays & solar flares (noise tajam di FITS)
        image = self._percentile_normalize(image)

        # 3. Augmentasi dua sudut pandang (view) untuk SimCLR
        #    PENTING: transform yang diteruskan HARUS bebas dari Flip/Rotation
        #    agar chirality filamen dan koordinat ekuatorial terjaga.
        if self.transform is not None:
            aug1 = self.transform(image=image)
            view_1 = aug1["image"]

            aug2 = self.transform(image=image)
            view_2 = aug2["image"]
        else:
            view_1 = image.copy()
            view_2 = image.copy()

        # 4. Konversi ke Tensor PyTorch jika belum (albumentations bisa return numpy)
        if not isinstance(view_1, torch.Tensor):
            view_1 = torch.from_numpy(view_1)
            view_2 = torch.from_numpy(view_2)

        # 5. Pastikan dimensi [C, H, W] → [1, H, W] untuk grayscale FITS
        if view_1.ndim == 2:
            view_1 = view_1.unsqueeze(0)
            view_2 = view_2.unsqueeze(0)
        elif view_1.ndim == 3 and view_1.shape[-1] == 1:
            # Albumentations kadang mengembalikan [H, W, 1] — konversi ke [1, H, W]
            view_1 = view_1.permute(2, 0, 1)
            view_2 = view_2.permute(2, 0, 1)

        return view_1, view_2
