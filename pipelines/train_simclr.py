import os
import glob

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_aug.transforms import get_ssl_transform

try:
    from models.backbone_simclr import SolarSimCLR
    from losses.nt_xent import NTXentLoss
    from dataloaders.dataset_ssl import SolarSSLDataset
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.backbone_simclr import SolarSimCLR
    from losses.nt_xent import NTXentLoss
    from dataloaders.dataset_ssl import SolarSSLDataset


def train_simclr_routine(config, resume_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Starting SimCLR routine on device: {device}")

    # ------------------------------------------------------------------
    # TensorBoard SummaryWriter
    # ------------------------------------------------------------------
    log_dir = os.path.join(config.system.weights_dir, "..", "runs", "simclr")
    log_dir = os.path.normpath(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[INFO] TensorBoard logs → '{log_dir}'  (jalankan: tensorboard --logdir runs/)")

    # ------------------------------------------------------------------
    # Model, Loss, Optimizer
    # ------------------------------------------------------------------
    print("[INFO] Initializing SolarSimCLR model...")
    model = SolarSimCLR(
        base_model=config.ssl_training.backbone,
        latent_dim=config.ssl_training.latent_dim,
    )
    model = model.to(device)

    print(f"[INFO] Setting up NT-Xent Contrastive Loss (Temp: {config.ssl_training.temperature})...")
    criterion = NTXentLoss(
        batch_size=config.ssl_training.batch_size,
        temperature=config.ssl_training.temperature,
    )

    print(
        f"[INFO] Setting up AdamW Optimizer "
        f"(LR: {config.ssl_training.learning_rate}, WD: {config.ssl_training.weight_decay})..."
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.ssl_training.learning_rate,
        weight_decay=config.ssl_training.weight_decay,
    )

    scaler = torch.amp.GradScaler('cuda') if config.system.fp16_precision else None
    if scaler:
        print("[INFO] Automatic Mixed Precision (AMP - fp16) ENABLED.")

    # ------------------------------------------------------------------
    # Gunakan direktori NPY yang sudah dihasilkan oleh pipelines/preprocess.py.
    # Fallback ke config.system.npy_train_dir jika ada, atau konstruksi otomatis
    # dari data_dir + processed/fits/train.
    #
    # PENTING: Jalankan `python main.py --mode preprocess` terlebih dahulu
    # sebelum mode ini agar file .npy tersedia.
    # ------------------------------------------------------------------
    npy_train_dir = getattr(
        config.system,
        "npy_train_dir",
        os.path.join(config.system.data_dir, "processed", "fits", "train"),
    )
    if not os.path.isdir(npy_train_dir):
        raise FileNotFoundError(
            f"[CRITICAL] Direktori NPY tidak ditemukan: '{npy_train_dir}'. "
            f"Jalankan terlebih dahulu: python main.py --mode preprocess"
        )

    # Non-rekursif — partisi train sudah terisolasi di satu direktori datar.
    # Glob rekursif dilarang karena ../processed/fits/test/ mengandung 180 file NPY
    # yang tidak boleh tersedot ke pipeline SSL.
    image_paths = glob.glob(os.path.join(npy_train_dir, "*.npy"))

    # FIX: Raise error eksplisit — jangan silent dry run
    if not image_paths:
        raise FileNotFoundError(
            f"[CRITICAL] Tidak ada file .npy ditemukan di '{npy_train_dir}'. "
            f"Pastikan `python main.py --mode preprocess` sudah dijalankan. "
            f"Training DIHENTIKAN untuk mencegah dry run yang menyesatkan."
        )

    # Guard kebocoran data: verifikasi tidak ada path dari partisi 'test' yang bocor
    leaking = [p for p in image_paths if "/test/" in p.replace("\\", "/")]
    if leaking:
        raise RuntimeError(
            f"[CRITICAL] Ditemukan {len(leaking)} file NPY dari partisi 'test' "
            f"dalam dataset SSL! Periksa kembali 'npy_train_dir' di config.yaml."
        )

    print(f"[INFO] Ditemukan {len(image_paths)} file NPY untuk SSL training.")
    print("[INFO] Preparing Dataloader (NPY — fast I/O)...")

    ssl_transform = get_ssl_transform(config)

    dataset = SolarSSLDataset(
        fits_paths=image_paths,
        transform=ssl_transform,
        config=config,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.ssl_training.batch_size,
        shuffle=True,
        num_workers=config.system.workers,
        drop_last=True,
    )

    print(f"[INFO] Starting Epoch Loop ({config.ssl_training.epochs} Epochs)")

    # Guard ERR-08: Jika drop_last=True dan dataset < batch_size, DataLoader memiliki
    # 0 batch. Training akan 'berhasil' tanpa satu pun update gradien dan menyimpan
    # bobot random — hasil yang menyesatkan.
    if len(dataloader) == 0:
        raise RuntimeError(
            f"[CRITICAL] DataLoader SSL menghasilkan 0 batch! "
            f"Jumlah file NPY ({len(image_paths)}) lebih kecil dari batch_size "
            f"({config.ssl_training.batch_size}). "
            f"Solusi: kurangi batch_size atau tambah data NPY."
        )

    os.makedirs(config.system.weights_dir, exist_ok=True)

    # ── LR Scheduler: CosineAnnealingLR ──────────────────────────────────────
    # Identik dengan sthalles SimCLR_01/run.py:
    #   T_max = len(dataloader) → satu siklus cosine selesai dalam T_max epoch
    #   scheduler.step() dipanggil SEKALI PER EPOCH setelah warmup
    # Warmup linear 10 epoch pertama: LR naik dari 0 → base_lr
    WARMUP_EPOCHS = getattr(config.ssl_training, 'warmup_epochs', 10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(dataloader),
        eta_min=0,
        last_epoch=-1,
    )
    print(
        f"[INFO] LR Scheduler: CosineAnnealingLR "
        f"(T_max={len(dataloader)} batch/epoch) "
        f"+ Linear Warmup {WARMUP_EPOCHS} epoch"
    )

    # Counter global untuk log TensorBoard per iterasi
    global_step = 0

    # ── Resume from Checkpoint ────────────────────────────────────────────────
    # Logika deteksi path:
    #   resume_path == None   → mulai dari awal (default)
    #   resume_path == 'auto' → cari 'checkpoint_last_simclr.pt' di weights_dir
    #   resume_path == <path> → muat dari path eksplisit yang diberikan
    start_epoch = 1
    ckpt_last_path = os.path.join(config.system.weights_dir, "checkpoint_last_simclr.pt")

    if resume_path is not None:
        if resume_path == "auto":
            load_path = ckpt_last_path
        else:
            load_path = resume_path

        if os.path.exists(load_path):
            print(f"[INFO] Memuat checkpoint SimCLR dari: '{load_path}'")
            ckpt = torch.load(load_path, map_location=device)

            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if scaler is not None and "scaler_state_dict" in ckpt:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_epoch  = ckpt["epoch"] + 1   # lanjut dari epoch BERIKUTNYA
            global_step  = ckpt.get("global_step", 0)

            print(
                f"[INFO] Resume OK — melanjutkan dari Epoch {start_epoch}/{config.ssl_training.epochs} "
                f"| global_step={global_step}"
            )
        else:
            print(
                f"[WARN] --resume diberikan tetapi checkpoint tidak ditemukan di '{load_path}'. "
                f"Memulai dari awal."
            )

    for epoch in range(start_epoch, config.ssl_training.epochs + 1):
        # ── Warmup LR Linear ─────────────────────────────────────────────────
        # Selama WARMUP_EPOCHS pertama, naikkan LR secara linear: 0 → base_lr.
        # Setelah warmup, CosineAnnealingLR mengambil alih (di-step per epoch).
        if epoch <= WARMUP_EPOCHS:
            warmup_lr = config.ssl_training.learning_rate * (epoch / WARMUP_EPOCHS)
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        # Log LR aktif ke TensorBoard
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("SimCLR/LearningRate_epoch", current_lr, epoch)

        model.train()
        epoch_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.ssl_training.epochs}", leave=False)

        for batch_idx, (view_1, view_2) in enumerate(pbar):
            view_1 = view_1.to(device)
            view_2 = view_2.to(device)

            optimizer.zero_grad()

            if config.system.fp16_precision:
                with torch.amp.autocast('cuda'):   # API baru (PyTorch ≥ 2.0)
                    z_i, _ = model(view_1)
                    z_j, _ = model(view_2)
                    loss = criterion(z_i, z_j)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                z_i, _ = model(view_1)
                z_j, _ = model(view_2)
                loss = criterion(z_i, z_j)
                loss.backward()
                optimizer.step()

            loss_val = loss.item()
            epoch_loss += loss_val

            # Log NT-Xent loss per iterasi ke TensorBoard
            writer.add_scalar("SimCLR/NTXentLoss_iter", loss_val, global_step)
            global_step += 1

            if batch_idx % config.ssl_training.log_every_n_steps == 0:
                pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        if len(dataloader) > 0:
            avg_loss = epoch_loss / len(dataloader)
            current_lr = optimizer.param_groups[0]['lr']
            # Log NT-Xent loss rata-rata per epoch ke TensorBoard
            writer.add_scalar("SimCLR/NTXentLoss_epoch", avg_loss, epoch)
            print(
                f"Epoch [{epoch}/{config.ssl_training.epochs}] "
                f"- Loss: {avg_loss:.4f}"
                f" | LR: {current_lr:.6f}"
                f" {'[WARMUP]' if epoch <= WARMUP_EPOCHS else '[COSINE]'}"
            )

        # ── Step Scheduler (setelah warmup selesai) ───────────────────────────
        # Identik dengan sthalles: scheduler.step() per epoch, mulai epoch ke-11.
        if epoch > WARMUP_EPOCHS:
            scheduler.step()

        # ── Simpan Checkpoint Lengkap ─────────────────────────────────────────
        # Checkpoint per-epoch (bernomor) untuk recovery manual.
        # checkpoint_last_simclr.pt selalu merupakan checkpoint TERBARU
        # dan digunakan oleh --resume auto.
        ckpt_dict = {
            "epoch":                epoch,
            "global_step":          global_step,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict":    scaler.state_dict() if scaler is not None else None,
            "config_ssl_training":  dict(config.ssl_training),
        }
        epoch_ckpt_path = os.path.join(
            config.system.weights_dir, f"simclr_epoch_{epoch}.pt"
        )
        torch.save(ckpt_dict, epoch_ckpt_path)
        torch.save(ckpt_dict, ckpt_last_path)   # overwrite alias terbaru

    # Tutup TensorBoard writer sebelum exit
    writer.close()

    final_weights_path = os.path.join(config.system.weights_dir, "simclr_final.pth")
    torch.save(model.state_dict(), final_weights_path)
    print(f"\n[INFO] SimCLR Pre-training concluded. Saved to: '{final_weights_path}'")
