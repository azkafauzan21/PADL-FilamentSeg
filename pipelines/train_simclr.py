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


def train_simclr_routine(config):
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

    os.makedirs(config.system.weights_dir, exist_ok=True)

    # Counter global untuk log TensorBoard per iterasi
    global_step = 0

    for epoch in range(1, config.ssl_training.epochs + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.ssl_training.epochs}", leave=False)

        for batch_idx, (view_1, view_2) in enumerate(pbar):
            view_1 = view_1.to(device)
            view_2 = view_2.to(device)

            optimizer.zero_grad()

            if config.system.fp16_precision:
                with torch.cuda.amp.autocast():
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
            # Log NT-Xent loss rata-rata per epoch ke TensorBoard
            writer.add_scalar("SimCLR/NTXentLoss_epoch", avg_loss, epoch)
            print(
                f"Epoch [{epoch}/{config.ssl_training.epochs}] "
                f"- Average NT-Xent Loss: {avg_loss:.4f}"
            )

        checkpoint_path = os.path.join(
            config.system.weights_dir, f"simclr_epoch_{epoch}.pth"
        )
        torch.save(model.state_dict(), checkpoint_path)

    # Tutup TensorBoard writer sebelum exit
    writer.close()

    final_weights_path = os.path.join(config.system.weights_dir, "simclr_final.pth")
    torch.save(model.state_dict(), final_weights_path)
    print(f"\n[INFO] SimCLR Pre-training concluded. Saved to: '{final_weights_path}'")
