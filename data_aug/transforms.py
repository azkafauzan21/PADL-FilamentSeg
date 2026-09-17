"""
transforms.py -- Pipeline Augmentasi PADL-FilamentSeg

Perombakan (2026-09-16):
  DIHAPUS:
    - RandomResizedCrop : Crop acak membuang konten piringan matahari yang berguna.
                          Model menerima piringan secara UTUH.
    - ColorJitter       : Manipulasi fluks H-alpha secara artifisial melanggar fisika.
                          Intensitas foton H-alpha adalah besaran fisis terukur.
  DITAMBAHKAN:
    - A.Resize(..., interpolation=cv2.INTER_CUBIC)
      Bicubic mempertahankan kualitas tepi filamen lebih baik dari INTER_LINEAR.
    - A.Affine(translate=0.1, scale=(0.7,1.3), rotate=[-5,5], shear=[-2,2], p=0.5)
      Simulasi tracking error teleskop GONG:
        * translate : drift posisi solar disk dalam frame FOV
        * scale     : variasi magnifikasi akibat refraksi atmosfer elevasi rendah
        * rotate    : rotasi instrumen +-5 deg (tidak merusak kiralitas global filamen)
        * shear     : distorsi optis tangensial

  DIPERTAHANKAN:
    - GaussianBlur : Atmospheric seeing (turbulensi kolom atmosfer darat GONG).
    - GaussNoise   : Read-noise dan dark current CCD sensor GONG.

LARANGAN FISIKA (dikodekan keras):
    x HorizontalFlip / VerticalFlip : Membalik kiralitas magnetik filamen.
    x Rotate bebas (>5 deg)         : Rotasi besar memutar koordinat ekuatorial.

Konsistensi SSL <-> Supervised:
    - Resolusi identik: image_size x image_size (default 512x512)
    - Affine identik: parameter sama di kedua pipeline
    - Interpolasi identik: cv2.INTER_CUBIC
    -> Mencegah "scale shock" antara Stage 1 dan Stage 2.
"""

import cv2
import albumentations as A


# ---------------------------------------------------------------------------
# Helper: Affine presisi rendah (tracking error teleskop)
# Dipanggil oleh SSL dan Supervised -- parameter identik untuk konsistensi.
# ---------------------------------------------------------------------------

def _tracking_error_affine(p: float = 0.5) -> A.Affine:
    """
    Affine transform yang mensimulasikan tracking error teleskop GONG.

      translate_percent=0.1 : Drift +-10% FOV (pointing error GONG antar-exposure).
      scale=(0.7, 1.3)      : Magnifikasi +-30% -- refraksi diferensial atmosfer.
      rotate=[-5, 5]        : Drift rotasi +-5 deg instrumen -- tidak merusak kiralitas
                              global filamen (filamen tidak terbalik, hanya dirotasi).
      shear=[-2, 2]         : Distorsi tangensial optis +-2 deg.
      interpolation=CUBIC   : Bicubic -- konsisten dengan Resize.
      mode=0                : cv2.BORDER_CONSTANT -- padding nol untuk area di luar FOV.
    """
    return A.Affine(
        translate_percent=0.1,
        scale=(0.7, 1.3),
        rotate=[-5, 5],
        shear=[-2, 2],
        interpolation=cv2.INTER_CUBIC,
        mode=0,
        p=p,
    )


# ---------------------------------------------------------------------------
# Stage 1: SSL SimCLR Pre-training
# ---------------------------------------------------------------------------

def get_ssl_transform(config):
    """
    Pipeline augmentasi untuk Stage 1 -- SSL SimCLR pre-training.

    Diterapkan DUA KALI secara independen pada gambar yang sama untuk menghasilkan
    dua view kontrastif (view_1 dan view_2) di SolarSSLDataset.__getitem__().

    Input: NPY 512x512 float32, ternormalisasi [0, 1].

    LARANGAN FISIKA (dikodekan keras):
        x HorizontalFlip / VerticalFlip
        x Rotate bebas (hanya +-5 deg diizinkan via Affine)
    """
    ssl = config.data_augmentation.ssl

    # Prioritas: ssl_training.image_size > data_augmentation.ssl.image_size > 512
    image_size = int(
        getattr(config.ssl_training, "image_size", None)
        or getattr(ssl, "image_size", 512)
    )

    return A.Compose([
        # -- Resize Bicubic --------------------------------------------------
        # Terima piringan matahari secara UTUH -- tidak ada cropping.
        # INTER_CUBIC: presisi lebih tinggi untuk struktur filamen yang tipis.
        A.Resize(
            height=image_size,
            width=image_size,
            interpolation=cv2.INTER_CUBIC,
        ),

        # -- Affine (Tracking Error Teleskop) --------------------------------
        # Parameter identik dengan supervised untuk konsistensi antar-fase.
        _tracking_error_affine(p=0.5),

        # -- GaussianBlur (Atmospheric Seeing) --------------------------------
        # Turbulensi kolom atmosfer darat GONG (Cerro Tololo, Learmonth, Maui).
        A.GaussianBlur(
            blur_limit=(ssl.blur_limit_min, ssl.blur_limit_max),
            p=ssl.blur_prob,
        ),

        # -- GaussNoise (Read-Noise & Dark Current CCD) -----------------------
        # API albumentations >= 2.0: std_range (bukan var_limit).
        # Konversi: std = sqrt(var) agar secara fisika ekuivalen.
        A.GaussNoise(
            std_range=(
                float(ssl.gauss_noise_var_limit[0]) ** 0.5,
                float(ssl.gauss_noise_var_limit[1]) ** 0.5,
            ),
            p=ssl.gauss_noise_prob,
        ),
    ])


# ---------------------------------------------------------------------------
# Stage 2: Supervised Mask2Former Fine-Tuning
# ---------------------------------------------------------------------------

def get_supervised_transform(config):
    """
    Pipeline augmentasi untuk Stage 2 -- Supervised Fine-Tuning Mask2Former.

    Diterapkan secara konsisten ke GAMBAR DAN SEMUA MASK secara bersamaan
    oleh albumentations (pixel-perfect mask alignment terjaga).

    Input: JPEG Kaggle (resolusi bervariasi) float32, ternormalisasi [0, 1].

    Konsistensi dengan SSL:
      - Resolusi: image_size x image_size (identik)
      - Affine: parameter identik (mencegah scale shock Stage 1 -> Stage 2)
      - Interpolasi: INTER_CUBIC (identik)

    LARANGAN FISIKA (dikodekan keras):
        x HorizontalFlip / VerticalFlip
        x Rotate bebas (hanya +-5 deg diizinkan via Affine)
    """
    sup = config.data_augmentation.supervised

    # Prioritas: supervised_training.image_size > data_augmentation.supervised.image_size > 512
    image_size = int(
        getattr(config.supervised_training, "image_size", None)
        or getattr(sup, "image_size", 512)
    )

    return A.Compose(
        [
            # -- Resize Bicubic -----------------------------------------------
            # Standarisasi JPEG Kaggle -> image_size x image_size.
            # INTER_CUBIC identik dengan SSL.
            A.Resize(
                height=image_size,
                width=image_size,
                interpolation=cv2.INTER_CUBIC,
            ),

            # -- Affine (Tracking Error Teleskop) ----------------------------
            # Diterapkan ke gambar DAN mask secara bersamaan -- alignment sempurna.
            _tracking_error_affine(p=0.5),

            # -- GaussNoise (Read-Noise & Dark Current CCD) ------------------
            # Hanya ke gambar (albumentations tidak menoise mask binary).
            A.GaussNoise(
                std_range=(
                    float(sup.gauss_noise_var_limit[0]) ** 0.5,
                    float(sup.gauss_noise_var_limit[1]) ** 0.5,
                ),
                p=sup.gauss_noise_prob,
            ),

            # -- GaussianBlur (Atmospheric Seeing) ---------------------------
            # Lebih konservatif dari SSL (blur_limit_max lebih kecil).
            A.GaussianBlur(
                blur_limit=(sup.blur_limit_min, sup.blur_limit_max),
                p=sup.blur_prob,
            ),
        ],
        # Nonaktifkan cek bentuk ketat untuk mask multi-ukuran.
        is_check_shapes=False,
    )


# ---------------------------------------------------------------------------
# Inferensi: Deterministik (tanpa augmentasi stokastik)
# ---------------------------------------------------------------------------

def get_inference_transform(config):
    """
    Transform deterministik minimal untuk inferensi (generate_submission).

    Hanya Resize Bicubic ke resolusi training -- tidak ada stokastisitas.
    """
    image_size = int(
        getattr(config.supervised_training, "image_size", None)
        or getattr(config.data_augmentation.supervised, "image_size", 512)
    )
    return A.Compose([
        A.Resize(
            height=image_size,
            width=image_size,
            interpolation=cv2.INTER_CUBIC,
        ),
    ])
