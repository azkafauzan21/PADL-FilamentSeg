import warnings

import numpy as np
import torch
from torch.utils.data import Dataset


class SolarSSLDataset(Dataset):
    def __init__(self, fits_paths, transform=None, config=None):
        """
        Dataset untuk pra-pelatihan SSL SimCLR menggunakan file .npy hasil
        pra-pemrosesan FITS (lihat pipelines/preprocess.py).

        Perubahan dari versi sebelumnya:
            - Input yang diterima berubah dari *.fits menjadi *.npy.
            - Pembacaan data menggunakan np.load() (instan, tanpa I/O disk berat)
              menggantikan astropy.io.fits (lambat, overhead dekompresi fpack).
            - Logika normalisasi persentil dan augmentasi dipertahankan sepenuhnya.
            - Format output tensor [C, H, W] = [1, 512, 512] dipertahankan.

        Args:
            fits_paths (list): List path file .npy dari partisi TRAIN.
                               Nama parameter dipertahankan 'fits_paths' untuk
                               kompatibilitas dengan kode pemanggil yang sudah ada
                               di train_simclr.py (cukup ubah glob *.fits → *.npy).
                               JANGAN menyertakan file dari partisi test.
            transform (albumentations.Compose): Transformasi fisika
                                                (wajib bebas dari Flip/Rotation).
            config (OmegaConf): Konfigurasi sistem.
        """
        self.fits_paths = fits_paths   # Sebenarnya list path *.npy — nama dipertahankan
        self.transform = transform
        self.config = config

    def __len__(self):
        return len(self.fits_paths)

    # ------------------------------------------------------------------
    # Pembacaan NPY: O(1) I/O — menggantikan _load_fits_image berbasis astropy
    # ------------------------------------------------------------------
    def _load_npy_image(self, path: str) -> np.ndarray:
        """
        Membaca file .npy yang telah dihasilkan oleh pipelines/preprocess.py.

        File .npy dijamin:
            - dtype: float32
            - shape: (512, 512)
            - bebas NaN/Inf (sudah dibersihkan saat preprocessing)
            - nilai mentah dalam skala intensitas H-alpha fisik (BZERO/BSCALE sudah diterapkan)

        np.load() menggunakan memory-mapped I/O secara internal (mmap_mode='r')
        sehingga akses pertama jauh lebih cepat dari astropy fits.open() yang
        harus melakukan dekompresi fpack dan parsing header FITS secara penuh.
        """
        try:
            data = np.load(path)
        except Exception as exc:
            raise IOError(
                f"[SolarSSLDataset] Gagal membaca file NPY: '{path}'. "
                f"Pastikan pipeline preprocess sudah dijalankan. Error: {exc}"
            ) from exc

        if data.ndim != 2:
            raise ValueError(
                f"[SolarSSLDataset] File NPY '{path}' memiliki shape {data.shape}. "
                f"Diharapkan array 2D (H, W)."
            )

        return data.astype(np.float32)

    # ------------------------------------------------------------------
    # Normalisasi persentil dengan validasi rentang (dipertahankan dari versi FITS)
    # ------------------------------------------------------------------
    def _percentile_normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalisasi kliping persentil [P_lower, P_upper] → [0.0, 1.0].

        Termasuk guard untuk gambar hampir-konstan (misal akibat sensor saturated)
        yang dapat menyebabkan silent near-zero division.

        Catatan: NaN/Inf sudah dibersihkan di preprocessing, namun guard ini
        tetap dipertahankan sebagai lapisan keamanan tambahan.
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

        # Guard: Jika rentang terlalu kecil, gambar kemungkinan saturated.
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

        # 1. Baca NPY — O(1) I/O tanpa overhead dekompresi FITS
        image = self._load_npy_image(path)

        # 2. Normalisasi fisika dengan validasi rentang
        #    Membuang cosmic rays & solar flares (noise tajam)
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

        # 5. Pastikan dimensi [C, H, W] → [1, H, W] untuk grayscale
        #    Output: [1, 512, 512] (setelah downsampling preprocessing)
        if view_1.ndim == 2:
            view_1 = view_1.unsqueeze(0)
            view_2 = view_2.unsqueeze(0)
        elif view_1.ndim == 3 and view_1.shape[-1] == 1:
            # Albumentations kadang mengembalikan [H, W, 1] — konversi ke [1, H, W]
            view_1 = view_1.permute(2, 0, 1)
            view_2 = view_2.permute(2, 0, 1)

        return view_1, view_2
