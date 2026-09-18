import argparse
import os
import sys

import torch
import numpy as np
import random
from omegaconf import OmegaConf

from pipelines.extract_metadata import extract_data_routine
from pipelines.preprocess import preprocess_routine
from pipelines.train_simclr import train_simclr_routine
from pipelines.train_mask2former import train_mask2former_routine
from pipelines.generate_submission import inference_routine
from utils.reproducibility import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="PADL-FilamentSeg: Physics-Aware Deep Learning for Solar Filaments",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ------------------------------------------------------------------
    # Required: pipeline mode selector
    # ------------------------------------------------------------------
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "extract_metadata",
            "preprocess",
            "train_simclr",
            "train_mask2former",
            "generate_submission",
        ],
        help=(
            "Tentukan fase spesifik pipeline yang akan dieksekusi:\n"
            "  extract_metadata    — Parse COCO JSON, buat anti-leakage train/val split\n"
            "  preprocess          — Konversi FITS 2048×2048 → NPY 512×512 (wajib sebelum train_simclr)\n"
            "  train_simclr        — Tahap 1: SSL pre-training pada data NPY\n"
            "  train_mask2former   — Tahap 2: Supervised fine-tuning pada data JPEG+COCO\n"
            "  generate_submission — Tahap 3: Inferensi dan pembuatan submission CSV"
        ),
    )

    # ------------------------------------------------------------------
    # Optional: path ke file konfigurasi
    # ------------------------------------------------------------------
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Jalur menuju file konfigurasi terpusat (OmegaConf YAML). Default: config.yaml",
    )

    # ------------------------------------------------------------------
    # Optional: flat override arguments
    #
    # Nilai-nilai ini bersifat opsional. Jika tidak diisi, nilai dari
    # config.yaml akan tetap digunakan. Jika diisi, nilai dari argumen
    # ini akan menimpa (override) nilai config yang relevan berdasarkan
    # --mode yang dipilih.
    # ------------------------------------------------------------------
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Jumlah worker DataLoader paralel. "
            "Menimpa config.system.workers. "
            "(Default: nilai dari config.yaml)"
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Ukuran batch per iterasi training.\n"
            "  mode=train_simclr       → menimpa config.ssl_training.batch_size\n"
            "  mode=train_mask2former  → menimpa config.supervised_training.batch_size\n"
            "(Default: nilai dari config.yaml)"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Total epoch training.\n"
            "  mode=train_simclr       → menimpa config.ssl_training.epochs\n"
            "  mode=train_mask2former  → menimpa config.supervised_training.epochs\n"
            "(Default: nilai dari config.yaml)"
        ),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help=(
            "Learning rate optimizer (AdamW).\n"
            "  mode=train_simclr       → menimpa config.ssl_training.learning_rate\n"
            "  mode=train_mask2former  → menimpa config.supervised_training.learning_rate\n"
            "(Default: nilai dari config.yaml)"
        ),
    )

    # ------------------------------------------------------------------
    # Optional: resume from checkpoint
    # ------------------------------------------------------------------
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        nargs="?",
        const="auto",
        metavar="PATH",
        help=(
            "Lanjutkan training dari checkpoint terakhir.\n"
            "  --resume          → cari otomatis 'checkpoint_last_simclr.pt' /\n"
            "                      'checkpoint_last_mask2former.pt' di weights_dir\n"
            "  --resume PATH     → muat checkpoint dari PATH yang ditentukan secara eksplisit\n"
            "(Default: None — mulai training dari awal)"
        ),
    )

    return parser.parse_args()


def apply_overrides(config, args):
    """
    Terapkan flat CLI arguments sebagai override ke objek OmegaConf config.

    Logika routing:
      --workers    → selalu ke config.system.workers (berlaku untuk semua mode)
      --batch_size → config.ssl_training        jika mode=train_simclr
                     config.supervised_training jika mode=train_mask2former
      --epochs     → routing identik dengan --batch_size
      --lr         → routing identik dengan --batch_size
                     (memetakan ke field 'learning_rate' di blok yang relevan)

    Hanya argumen yang secara eksplisit diisi (bukan None) yang akan menimpa
    nilai config. Argumen yang tidak diisi dibiarkan menggunakan nilai YAML.
    """
    overridden: list[str] = []

    # -- workers: global, tidak bergantung mode --
    if args.workers is not None:
        config.system.workers = args.workers
        overridden.append(f"system.workers = {args.workers}")

    # -- Tentukan blok config target berdasarkan mode --
    mode = args.mode
    if mode == "train_simclr":
        training_block = config.ssl_training
        block_name = "ssl_training"
    elif mode == "train_mask2former":
        training_block = config.supervised_training
        block_name = "supervised_training"
    else:
        # extract_metadata dan generate_submission tidak memerlukan routing training
        training_block = None
        block_name = None

    if training_block is not None:
        if args.batch_size is not None:
            training_block.batch_size = args.batch_size
            overridden.append(f"{block_name}.batch_size = {args.batch_size}")

        if args.epochs is not None:
            training_block.epochs = args.epochs
            overridden.append(f"{block_name}.epochs = {args.epochs}")

        if args.lr is not None:
            training_block.learning_rate = args.lr
            overridden.append(f"{block_name}.learning_rate = {args.lr}")

    # Laporan ringkas override yang diterapkan
    if overridden:
        print("[INFO] CLI Overrides diterapkan ke config:")
        for entry in overridden:
            print(f"       ✓ {entry}")
    else:
        print("[INFO] Tidak ada CLI override. Menggunakan seluruh nilai dari config.yaml.")

    # Laporan flag --resume (informatif, tidak mengubah config)
    if hasattr(args, 'resume') and args.resume is not None:
        if args.resume == "auto":
            print("[INFO] --resume: Deteksi otomatis checkpoint terakhir diaktifkan.")
        else:
            print(f"[INFO] --resume: Muat dari path eksplisit → '{args.resume}'")

    return config


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Muat konfigurasi dari YAML
    # ------------------------------------------------------------------
    if not os.path.exists(args.config):
        print(f"[CRITICAL ERROR] File konfigurasi '{args.config}' tidak ditemukan.")
        sys.exit(1)

    config = OmegaConf.load(args.config)

    # ------------------------------------------------------------------
    # Terapkan flat CLI overrides ke config
    # ------------------------------------------------------------------
    config = apply_overrides(config, args)

    # ------------------------------------------------------------------
    # Reproduktibilitas: kunci status stokastik sistem
    # ------------------------------------------------------------------
    if config.system.seed is not None:
        print(f"[INFO] Mengunci status stokastik sistem (Random Seed: {config.system.seed})...")
        set_seed(config.system.seed)

    # Buat direktori weights jika belum ada
    os.makedirs(config.system.weights_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Router eksklusif berbasis mode
    # ------------------------------------------------------------------
    print(f"[INFO] Memulai Eksekusi Pipeline: {args.mode.upper()}")

    if args.mode == "extract_metadata":
        extract_data_routine(config)
    elif args.mode == "preprocess":
        preprocess_routine(config)
    elif args.mode == "train_simclr":
        train_simclr_routine(config, resume_path=args.resume)
    elif args.mode == "train_mask2former":
        train_mask2former_routine(config, resume_path=args.resume)
    elif args.mode == "generate_submission":
        inference_routine(config)
    else:
        print(f"[CRITICAL ERROR] Mode '{args.mode}' tidak dikenali!")


if __name__ == "__main__":
    main()
