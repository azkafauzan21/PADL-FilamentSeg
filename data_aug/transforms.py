import albumentations as A


def get_ssl_transform(config):
    """
    Pipeline augmentasi untuk Stage 1 — SSL SimCLR pre-training.

    Semua parameter dibaca dari config.data_augmentation.ssl sehingga
    dapat dikontrol sepenuhnya melalui config.yaml tanpa menyentuh kode.

    LARANGAN FISIKA (tidak dapat dikonfigurasi):
        - HorizontalFlip / VerticalFlip: melanggar kiralitas filamen
        - Rotate: mengubah koordinat ekuatorial & heliolatitude
    """
    ssl = config.data_augmentation.ssl

    return A.Compose([
        # ── RandomResizedCrop ─────────────────────────────────────────────
        # Mensimulasikan observasi dari jarak & sudut pandang berbeda.
        # crop_size harus ≤ ukuran NPY (512px, output preprocess.py).
        A.RandomResizedCrop(
            size=(ssl.crop_size, ssl.crop_size),
            scale=list(ssl.crop_scale),
            p=1.0,
        ),

        # ── ColorJitter ───────────────────────────────────────────────────
        # Mensimulasikan variasi kecerahan instrumen GONG antar stasiun
        # (Cerro Tololo, Learmonth, Maui, dll.) yang memiliki kalibrasi flux berbeda.
        A.ColorJitter(
            brightness=ssl.color_jitter_brightness,
            contrast=ssl.color_jitter_contrast,
            saturation=0,   # Gambar grayscale — saturation selalu 0
            hue=0,          # Gambar grayscale — hue selalu 0
            p=ssl.jitter_prob,
        ),

        # ── GaussianBlur ──────────────────────────────────────────────────
        # Mensimulasikan smearing atmosfer (atmospheric seeing) — GONG
        # mengobservasi dari darat sehingga turbulensi atmosfer merupakan
        # sumber degradasi yang nyata.
        A.GaussianBlur(
            blur_limit=(ssl.blur_limit_min, ssl.blur_limit_max),
            p=ssl.blur_prob,
        ),

        # ── GaussNoise ────────────────────────────────────────────────────
        # Mensimulasikan read-noise dan dark current detektor CCD GONG.
        # var_limit dalam skala gambar ternormalisasi [0, 1].
        A.GaussNoise(
            var_limit=list(ssl.gauss_noise_var_limit),
            mean=0.0,
            p=ssl.gauss_noise_prob,
        ),
    ])


def get_supervised_transform(config):
    """
    Pipeline augmentasi untuk Stage 2 — Supervised Fine-Tuning Mask2Former.

    Semua parameter dibaca dari config.data_augmentation.supervised.
    Augmentasi ini diterapkan secara konsisten ke GAMBAR DAN MASK secara
    bersamaan oleh albumentations (pixel-perfect mask alignment terjaga).

    Augmentasi lebih konservatif dari SSL karena mask instance harus
    tetap valid setelah transformasi.

    LARANGAN FISIKA (tidak dapat dikonfigurasi):
        - HorizontalFlip / VerticalFlip
        - Rotate (rotate_limit di config harus selalu 0)
    """
    sup = config.data_augmentation.supervised

    # Verifikasi larangan fisika: rotate_limit harus 0
    # Jika seseorang secara tidak sengaja mengubahnya di config.yaml,
    # ini akan memberikan peringatan eksplisit daripada diam-diam melatih
    # model dengan data yang melanggar hukum fisika kiralitas.
    if hasattr(sup, 'rotate_limit') and sup.rotate_limit != 0:
        import warnings
        warnings.warn(
            f"[PHYSICS VIOLATION] supervised.rotate_limit = {sup.rotate_limit} "
            f"terdeteksi di config.yaml! Rotasi melanggar kiralitas magnetik filamen. "
            f"Nilai ini DIPAKSA ke 0. Set ke 0 di config.yaml untuk menghilangkan peringatan ini.",
            stacklevel=2,
        )
        rotate_limit = 0
    else:
        rotate_limit = 0  # Selalu 0, diabaikan jika key tidak ada

    return A.Compose(
        [
            # ── Resize ────────────────────────────────────────────────────
            # Standarisasi ukuran input untuk Mask2Former.
            # resize_dim: 512 px direkomendasikan untuk VRAM < 16 GB.
            A.Resize(
                height=sup.resize_dim,
                width=sup.resize_dim,
            ),

            # ── ShiftScaleRotate ──────────────────────────────────────────
            # Translasi dan skala saja. rotate_limit=0 menonaktifkan rotasi.
            # Mensimulasikan variasi posisi filamen dalam frame observasi GONG.
            A.ShiftScaleRotate(
                shift_limit=sup.shift_limit,
                scale_limit=sup.scale_limit,
                rotate_limit=rotate_limit,   # SELALU 0 — dijamin oleh guard di atas
                border_mode=0,               # cv2.BORDER_CONSTANT: padding nol
                p=sup.shift_scale_prob,
            ),

            # ── ElasticTransform ──────────────────────────────────────────
            # Deformasi elastis ringan mensimulasikan distorsi atmosfer lokal.
            # Diterapkan ke gambar DAN mask secara bersamaan.
            A.ElasticTransform(
                alpha=sup.elastic_alpha,
                sigma=sup.elastic_sigma,
                p=sup.elastic_prob,
            ),

            # ── GaussNoise ────────────────────────────────────────────────
            # Noise pada gambar (tidak diterapkan ke mask).
            # Lebih konservatif dari SSL agar tidak merusak piksel mask.
            A.GaussNoise(
                var_limit=list(sup.gauss_noise_var_limit),
                mean=0.0,
                p=sup.gauss_noise_prob,
            ),

            # ── GaussianBlur ──────────────────────────────────────────────
            # Smearing ringan pada gambar untuk robustness terhadap seeing.
            A.GaussianBlur(
                blur_limit=(sup.blur_limit_min, sup.blur_limit_max),
                p=sup.blur_prob,
            ),

            # ── Sharpen ───────────────────────────────────────────────────
            # Penonjolan tepi filamen. Diterapkan bergantian dengan blur
            # sehingga model belajar pada gambar tajam maupun kabur.
            A.Sharpen(
                alpha=list(sup.sharpen_alpha),
                lightness=(0.9, 1.1),  # Perubahan kecerahan minimal saat sharpening
                p=sup.sharpen_prob,
            ),
        ],
        # Albumentations menjamin transformasi spasial (Resize, ShiftScaleRotate,
        # ElasticTransform) diterapkan identik ke gambar dan semua mask.
        # GaussNoise dan blur HANYA ke gambar (tidak memengaruhi mask).
        is_check_shapes=False,  # Nonaktifkan cek bentuk ketat untuk mask multi-ukuran
    )


def get_inference_transform(config):
    """
    Transform deterministik minimal untuk inferensi (generate_submission).

    Tidak ada augmentasi stokastik — hanya resize deterministik untuk
    memastikan input model memiliki ukuran yang benar.
    """
    resize_dim = config.data_augmentation.supervised.resize_dim
    return A.Compose([
        A.Resize(height=resize_dim, width=resize_dim),
    ])
