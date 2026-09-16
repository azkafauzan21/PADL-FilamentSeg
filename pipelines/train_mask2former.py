import os

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_aug.transforms import get_supervised_transform

try:
    from models.segmenter_mask2former import FilamentMask2Former
    from dataloaders.dataset_supervised import FilamentSupervisedDataset
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.segmenter_mask2former import FilamentMask2Former
    from dataloaders.dataset_supervised import FilamentSupervisedDataset


# ---------------------------------------------------------------------------
# Collate Function
# ---------------------------------------------------------------------------

def custom_collate_fn(batch):
    """Collate function kustom untuk menangani label berukuran bervariasi per gambar."""
    if not batch:
        return [], []

    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    targets = []
    for item in batch:
        labels_dict = item["labels"]
        targets.append({
            "class_labels": labels_dict["class_labels"],
            "mask_labels": labels_dict["masks"],
        })
    return pixel_values, targets


# ---------------------------------------------------------------------------
# Metrik: Panoptic Quality (PQ) Sederhana
# ---------------------------------------------------------------------------

def compute_panoptic_quality(
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    iou_threshold: float = 0.5,
) -> float:
    """
    Hitung Panoptic Quality (PQ) sederhana berbasis IoU mask.

    PQ = SUM(IoU(tp)) / (|TP| + 0.5*|FP| + 0.5*|FN|)

    Args:
        pred_masks : Tensor bool [Q_pred, H, W] — prediksi biner dari satu gambar.
        gt_masks   : Tensor bool [Q_gt, H, W]   — ground-truth biner dari satu gambar.
        iou_threshold: Ambang batas IoU untuk menganggap pasangan sebagai True Positive.

    Returns:
        Nilai PQ float dalam rentang [0, 1].
    """
    if pred_masks.numel() == 0 or gt_masks.numel() == 0:
        # Jika salah satu kosong, PQ = 0
        return 0.0

    # Pindahkan ke CPU SEBELUM komputasi matriks — mencegah pembuatan tensor
    # sementara berukuran [Q_pred, Q_gt, H, W] di VRAM GPU (bisa > 4 GB).
    pred_masks = pred_masks.bool().cpu()
    gt_masks = gt_masks.bool().cpu()

    Q_pred = pred_masks.shape[0]
    Q_gt = gt_masks.shape[0]

    # Hitung IoU semua pasangan (Q_pred × Q_gt) secara vektorisasi di CPU
    # Reshape: [Q_pred, 1, H, W] & [1, Q_gt, H, W]
    p = pred_masks.unsqueeze(1).float()  # [Q_pred, 1, H, W]
    g = gt_masks.unsqueeze(0).float()    # [1, Q_gt, H, W]

    intersection = (p * g).sum(dim=(-2, -1))            # [Q_pred, Q_gt]
    union = ((p + g) > 0).float().sum(dim=(-2, -1))     # [Q_pred, Q_gt]
    iou_matrix = intersection / (union + 1e-6)           # [Q_pred, Q_gt]

    # Greedy matching: pasangkan prediksi dengan GT terbaik (IoU tertinggi)
    matched_gt = set()
    tp_iou_sum = 0.0
    tp_count = 0

    # Urutkan pasangan berdasarkan IoU menurun
    iou_flat = iou_matrix.cpu().numpy().ravel()
    sorted_indices = iou_flat.argsort()[::-1]

    matched_pred = set()
    for flat_idx in sorted_indices:
        p_idx = int(flat_idx) // Q_gt
        g_idx = int(flat_idx) % Q_gt
        iou_val = float(iou_matrix[p_idx, g_idx].item())

        if iou_val < iou_threshold:
            break  # IoU sisa sudah pasti lebih kecil

        if p_idx not in matched_pred and g_idx not in matched_gt:
            tp_iou_sum += iou_val
            tp_count += 1
            matched_pred.add(p_idx)
            matched_gt.add(g_idx)

    fp = Q_pred - tp_count
    fn = Q_gt - tp_count

    denominator = tp_count + 0.5 * fp + 0.5 * fn
    if denominator < 1e-6:
        return 1.0 if tp_count == 0 and Q_gt == 0 else 0.0

    pq = tp_iou_sum / denominator
    return float(pq)


# ---------------------------------------------------------------------------
# Validation Loop
# ---------------------------------------------------------------------------

def run_validation(
    model: torch.nn.Module,
    val_dataloader: DataLoader,
    device: torch.device,
    dice_metric,          # torchmetrics.segmentation.DiceScore instance
    iou_threshold: float = 0.5,
    threshold: float = 0.5,
) -> tuple[float, float]:
    """
    Jalankan validation loop dan kembalikan (avg_dice, avg_pq).

    Args:
        model           : Model dalam mode eval.
        val_dataloader  : DataLoader subset validasi.
        device          : Device target.
        dice_metric     : Instance DiceScore dari torchmetrics.
        iou_threshold   : Ambang IoU untuk PQ matching.
        threshold       : Ambang sigmoid untuk binarisasi prediksi.

    Returns:
        Tuple (avg_dice_score, avg_pq_score) — rata-rata di seluruh batch.
    """
    model.eval()
    dice_metric.reset()
    pq_scores: list[float] = []

    with torch.no_grad():
        for images, targets in tqdm(val_dataloader, desc="  [Val]", leave=False):
            if not len(images):
                continue

            images = images.to(device)

            mask_logits, _class_logits = model(pixel_values=images)
            # mask_logits: [B, Q, H, W]
            
            target_h, target_w = targets[0]["mask_labels"].shape[-2:]
        
            # Upsample prediksi kembali ke ukuran asli (misal: 256 -> 1024)
            mask_logits = F.interpolate(
                mask_logits, 
                size=(target_h, target_w), 
                mode="bilinear", 
                align_corners=False
            )

            batch_size = images.shape[0]
            for b in range(batch_size):
                logits_b = mask_logits[b]           # [Q, H, W]
                probs_b = torch.sigmoid(logits_b)
                pred_bin = (probs_b > threshold)    # [Q, H, W] bool

                # Collapse ke single-channel mask untuk DiceScore (binary segmentation)
                # Ambil union semua query sebagai satu mask prediksi
                pred_union = pred_bin.any(dim=0, keepdim=True).float()   # [1, H, W]

                gt_masks_b = targets[b]["mask_labels"].to(device)        # [Q_gt, H, W]
                gt_union = gt_masks_b.bool().any(dim=0, keepdim=True).float()  # [1, H, W]

                # torchmetrics DiceScore expects [B, C, H, W] — bungkus dimensi batch
                dice_metric.update(
                    pred_union.unsqueeze(0).long(),   # [1, 1, H, W]
                    gt_union.unsqueeze(0).long(),      # [1, 1, H, W]
                )

                # PQ per gambar (instance-level)
                # Hilangkan query kosong sebelum passing ke compute_pq
                active_preds = pred_bin[pred_bin.any(dim=(-2, -1))]  # [Q_active, H, W]
                active_gts = gt_masks_b.bool()
                active_gts = active_gts[active_gts.any(dim=(-2, -1))]

                pq_val = compute_panoptic_quality(active_preds, active_gts, iou_threshold)
                pq_scores.append(pq_val)

    # dice_metric dengan average='none' seharusnya menghasilkan tensor [num_classes].
    # Guard diperlukan: jika epoch berjalan tanpa sampel valid (edge case),
    # torchmetrics bisa mengembalikan tensor 0-dim (scalar) alih-alih [num_classes].
    # Indexing [0] pada 0-dim tensor → IndexError. Tangani keduanya dengan aman.
    dice_per_class = dice_metric.compute()
    if dice_per_class.ndim == 0:
        # Scalar fallback — tidak ada sampel valid yang diproses
        avg_dice = 0.0
    else:
        avg_dice = float(dice_per_class[0].item())  # Kelas 0 = Filament Body
    avg_pq = float(sum(pq_scores) / len(pq_scores)) if pq_scores else 0.0
    return avg_dice, avg_pq


# ---------------------------------------------------------------------------
# Main Training Routine
# ---------------------------------------------------------------------------

def train_mask2former_routine(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Starting Mask2Former supervised routine on device: {device}")

    # ------------------------------------------------------------------
    # TensorBoard SummaryWriter
    # ------------------------------------------------------------------
    log_dir = os.path.join(config.system.weights_dir, "..", "runs", "mask2former")
    log_dir = os.path.normpath(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[INFO] TensorBoard logs → '{log_dir}'  (jalankan: tensorboard --logdir runs/)")

    # ------------------------------------------------------------------
    # Inisialisasi Model
    # ------------------------------------------------------------------
    print(f"[INFO] Initializing FilamentMask2Former model (Backbone: {config.ssl_training.backbone})...")
    model = FilamentMask2Former(
        num_classes=2,
        base_model=config.ssl_training.backbone,
        # Teruskan latent_dim agar Stage 2 sinkron dengan Stage 1 SSL (FIX ERR-10)
        # FilamentMask2Former meneruskan ini ke SolarSimCLR.__init__
    )
    # num_classes=2: Kelas 0 = Filament Body, Kelas 1 = Filament Spine

    # ------------------------------------------------------------------
    # Muat bobot backbone SimCLR dari Tahap 1
    # ------------------------------------------------------------------
    simclr_path = os.path.join(config.system.weights_dir, "simclr_final.pth")
    if os.path.exists(simclr_path):
        print(f"[INFO] Loading backbone weights from: {simclr_path}")
        try:
            state_dict = torch.load(simclr_path, map_location="cpu")
            # strict=False mengizinkan key yang tidak cocok (misal projection head
            # yang tidak ada di FilamentMask2Former) tanpa error fatal.
            missing, unexpected = model.backbone.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[WARN] Key backbone tidak ditemukan (normal jika projection head berbeda): "
                      f"{missing[:3]}{'...' if len(missing) > 3 else ''}")
            if unexpected:
                print(f"[WARN] Key tidak dikenali dari checkpoint: "
                      f"{unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")
        except Exception as e:
            print(f"[ERROR] Gagal memuat bobot backbone: {e}. Melanjutkan dengan bobot acak.")
    else:
        print("[WARN] Bobot SimCLR tidak ditemukan. Melatih dari awal (scratch).")

    model = model.to(device)

    # ------------------------------------------------------------------
    # FIX BUG-05: Backbone Freezing Strategy
    #
    # Fase 1 (epoch 1 – freeze_backbone_epochs):
    #   Bekukan semua parameter backbone. Hanya PixelDecoder, TransformerDecoder,
    #   dan feature_projections yang dilatih.
    #   Ini mencegah catastrophic forgetting representasi SSL.
    #
    # Fase 2 (setelah freeze_backbone_epochs):
    #   Cairkan backbone dengan learning rate 10× lebih kecil (differential LR).
    # ------------------------------------------------------------------
    FREEZE_EPOCHS = getattr(config.supervised_training, "freeze_backbone_epochs", 5)

    def set_backbone_grad(requires_grad: bool):
        """Toggle gradient backbone dan laporkan statusnya."""
        for param in model.backbone.parameters():
            param.requires_grad = requires_grad
        state_label = "AKTIF (full fine-tuning)" if requires_grad else "BEKU (frozen)"
        print(f"[INFO] Backbone gradient: {state_label}")

    def make_optimizer(phase: str) -> optim.Optimizer:
        """
        Buat optimizer yang sesuai untuk setiap fase training.

        phase='frozen':    hanya latih parameter non-backbone
        phase='unfrozen':  differential LR — backbone 10× lebih kecil dari decoder
        """
        if phase == "frozen":
            params = [p for p in model.parameters() if p.requires_grad]
            print(f"[INFO] Optimizer (Fase frozen): {len(params)} parameter group(s)")
            return optim.AdamW(
                params,
                lr=config.supervised_training.learning_rate,
                weight_decay=config.supervised_training.weight_decay,
            )
        else:  # unfrozen
            backbone_params = list(model.backbone.parameters())
            other_params = [p for n, p in model.named_parameters() if "backbone" not in n]
            lr_base = config.supervised_training.learning_rate
            print(
                f"[INFO] Optimizer (Fase unfrozen): "
                f"backbone LR={lr_base * 0.1:.6f}, decoder LR={lr_base:.6f}"
            )
            return optim.AdamW(
                [
                    {"params": backbone_params, "lr": lr_base * 0.1},
                    {"params": other_params, "lr": lr_base},
                ],
                weight_decay=config.supervised_training.weight_decay,
            )

    # Mulai dengan backbone beku (Fase 1)
    set_backbone_grad(requires_grad=False)
    optimizer = make_optimizer(phase="frozen")

    scaler = torch.amp.GradScaler('cuda') if config.system.fp16_precision else None
    if scaler:
        print("[INFO] Automatic Mixed Precision (AMP - fp16) ENABLED.")

    # ------------------------------------------------------------------
    # Torchmetrics: DiceScore
    # ------------------------------------------------------------------
    try:
        import torchmetrics
        tm_version = tuple(int(x) for x in torchmetrics.__version__.split(".")[:2])
        if tm_version < (1, 0):
            print(
                f"[WARN] torchmetrics versi {torchmetrics.__version__} terdeteksi. "
                f"DiceScore dari 'torchmetrics.segmentation' membutuhkan >= 1.0.0. "
                f"Upgrade: pip install 'torchmetrics>=1.0.0'"
            )
            use_metrics = False
            dice_metric = None
        else:
            from torchmetrics.segmentation import DiceScore
            # average='none' → kembalikan skor per-kelas; kita ambil hanya kelas 0 (Filament Body)
            # Kelas 1 (Spine) diabaikan karena Kaggle hanya menilai luasan tubuh filamen.
            dice_metric = DiceScore(num_classes=2, average="none").to(device)
            use_metrics = True
            print("[INFO] torchmetrics DiceScore: AKTIF (num_classes=2, eval: Kelas 0 = Body only)")
    except ImportError:
        print("[WARN] torchmetrics tidak terinstal. Dice/PQ tidak akan dihitung. "
              "Install dengan: pip install 'torchmetrics>=1.0.0'")
        use_metrics = False
        dice_metric = None

    # ------------------------------------------------------------------
    # FIX BUG-01: Gunakan config.system.train_split_json dan jpeg_train_dir
    # (menggantikan konstruksi path salah yang tidak cocok struktur direktori)
    # ------------------------------------------------------------------
    print("[INFO] Preparing Supervised Dataloader...")

    train_split_path = config.system.train_split_json  # → "./data/train_split.json"
    val_split_path   = config.system.val_split_json    # → "./data/val_split.json"
    jpeg_train_dir   = config.system.jpeg_train_dir    # → "./data/raw/MAGFiLO_.../train/train_images"

    if not os.path.exists(train_split_path):
        raise FileNotFoundError(
            f"[CRITICAL] File split training tidak ditemukan: '{train_split_path}'. "
            f"Jalankan pipeline 'extract_metadata' terlebih dahulu untuk membuat split ini."
        )

    if not os.path.isdir(jpeg_train_dir):
        raise FileNotFoundError(
            f"[CRITICAL] Direktori JPEG training tidak ditemukan: '{jpeg_train_dir}'. "
            f"Periksa nilai 'jpeg_train_dir' di config.yaml."
        )

    supervised_transform = get_supervised_transform(config)

    train_dataset = FilamentSupervisedDataset(
        coco_json_path=train_split_path,
        image_dir=jpeg_train_dir,
        transform=supervised_transform,
        config=config,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.supervised_training.batch_size,
        shuffle=True,
        num_workers=config.system.workers,
        collate_fn=custom_collate_fn,
    )

    # ------------------------------------------------------------------
    # Validation DataLoader (opsional — lewati jika val_split_json tidak ada)
    # ------------------------------------------------------------------
    val_dataloader = None
    if os.path.exists(val_split_path) and os.path.isdir(jpeg_train_dir):
        val_dataset = FilamentSupervisedDataset(
            coco_json_path=val_split_path,
            image_dir=jpeg_train_dir,
            transform=supervised_transform,
            config=config,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.supervised_training.batch_size,
            shuffle=False,
            num_workers=config.system.workers,
            collate_fn=custom_collate_fn,
        )
        print(f"[INFO] Validation dataset: {len(val_dataset)} sampel.")
    else:
        print(f"[WARN] val_split.json tidak ditemukan di '{val_split_path}'. "
              "Validation loop dinonaktifkan.")

    print(
        f"[INFO] Train dataset: {len(train_dataset)} sampel, "
        f"{len(train_dataloader)} batch/epoch. "
        f"Backbone akan dicairkan setelah epoch {FREEZE_EPOCHS}."
    )
    print(f"[INFO] Starting Epoch Loop ({config.supervised_training.epochs} Epochs)")

    os.makedirs(config.system.weights_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Smart Checkpointing: simpan best_mask2former.pth hanya jika PQ naik
    # ------------------------------------------------------------------
    best_pq: float = -1.0
    best_checkpoint_path = os.path.join(config.system.weights_dir, "best_mask2former.pth")

    # Counter global untuk log TensorBoard per iterasi
    global_step = 0

    for epoch in range(1, config.supervised_training.epochs + 1):
        # ------------------------------------------------------------------
        # Transisi Fase 1 → Fase 2: Cairkan backbone setelah FREEZE_EPOCHS
        # ------------------------------------------------------------------
        if epoch == FREEZE_EPOCHS + 1:
            print(f"\n[INFO] Epoch {epoch}: Transisi ke Fase 2 — Mencairkan backbone.")
            set_backbone_grad(requires_grad=True)
            optimizer = make_optimizer(phase="unfrozen")

        # ==================================================================
        # Training Loop
        # ==================================================================
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch}/{config.supervised_training.epochs}",
            leave=False,
        )

        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(device)
            # Pindahkan hanya nilai Tensor ke device; abaikan non-Tensor (FIX ERR-12)
            targets = [
                {k: v.to(device) for k, v in target.items() if isinstance(v, torch.Tensor)}
                for target in targets
            ]

            optimizer.zero_grad()

            if config.system.fp16_precision:
                with torch.amp.autocast('cuda'):   # API baru (PyTorch ≥ 2.0)
                    outputs = model(pixel_values=images, labels=targets)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(pixel_values=images, labels=targets)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

            loss_val = loss.item()
            epoch_loss += loss_val

            # Log loss per iterasi ke TensorBoard
            writer.add_scalar("Train/BipartiteLoss_iter", loss_val, global_step)
            global_step += 1

            if batch_idx % config.supervised_training.log_every_n_steps == 0:
                pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        # ------------------------------------------------------------------
        # Log rata-rata loss per epoch
        # ------------------------------------------------------------------
        if len(train_dataloader) > 0:
            avg_loss = epoch_loss / len(train_dataloader)
            writer.add_scalar("Train/BipartiteLoss_epoch", avg_loss, epoch)
            print(
                f"Epoch [{epoch}/{config.supervised_training.epochs}] "
                f"- Average Mask2Former Loss: {avg_loss:.4f}"
            )

        # ==================================================================
        # Validation Loop
        # ==================================================================
        avg_dice, avg_pq = 0.0, 0.0

        if val_dataloader is not None and use_metrics:
            avg_dice, avg_pq = run_validation(
                model=model,
                val_dataloader=val_dataloader,
                device=device,
                dice_metric=dice_metric,
                iou_threshold=0.5,
                threshold=0.5,
            )

            # Log metrik validasi ke TensorBoard
            writer.add_scalar("Val/DiceScore", avg_dice, epoch)
            writer.add_scalar("Val/PanopticQuality", avg_pq, epoch)

            print(
                f"  [Val] Epoch {epoch}: Dice={avg_dice:.4f} | PQ={avg_pq:.4f}"
            )

            # ------------------------------------------------------------------
            # Smart Checkpointing: simpan jika PQ memecahkan rekor
            # ------------------------------------------------------------------
            if avg_pq > best_pq:
                best_pq = avg_pq
                torch.save(model.state_dict(), best_checkpoint_path)
                print(
                    f"  [CKPT] ✅ Rekor PQ baru: {avg_pq:.4f}. "
                    f"Disimpan ke '{best_checkpoint_path}'"
                )
            else:
                print(
                    f"  [CKPT] PQ {avg_pq:.4f} tidak melampaui rekor {best_pq:.4f}. "
                    "Checkpoint tidak diperbarui."
                )

        # Checkpoint epoch biasa
        checkpoint_path = os.path.join(
            config.system.weights_dir, f"mask2former_epoch_{epoch}.pth"
        )
        torch.save(model.state_dict(), checkpoint_path)

    # Tutup TensorBoard writer
    writer.close()

    final_weights_path = os.path.join(config.system.weights_dir, "mask2former_final.pth")
    torch.save(model.state_dict(), final_weights_path)
    print(f"\n[INFO] Mask2Former fine-tuning concluded. Saved to: '{final_weights_path}'")
    if best_pq >= 0:
        print(f"[INFO] Best PQ achieved: {best_pq:.4f} → '{best_checkpoint_path}'")
