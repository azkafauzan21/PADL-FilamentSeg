import torch
import random
import traceback
from models.backbone_simclr import SolarSimCLR
from losses.nt_xent import NTXentLoss
from models.segmenter_mask2former import FilamentMask2Former
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# Sesuaikan import class dataset ini jika agen AI memberikan nama yang berbeda
from dataloaders.dataset_ssl import SolarSSLDataset 

def main_sanity_check():
    print("="*50)
    print("MEMULAI SANITY CHECK ALIRAN KOMPUTASI (TENSORS)")
    print("="*50)
    
    # ---------------------------------------------------------
    # Skenario 1: Forward Pass SolarSimCLR (Dummy Data)
    # ---------------------------------------------------------
    try:
        print("\nSkenario 1: Forward Pass SolarSimCLR")
        model_simclr = SolarSimCLR(base_model='resnet50')
        model_simclr.eval()
        
        x = torch.randn(2, 1, 512, 512)
        with torch.no_grad():
            z, features = model_simclr(x)
            
        assert z.shape == (2, 128), f"Error dimensi z: {z.shape}"
        assert len(features) == 5, f"Error jumlah fitur spasial: {len(features)}"
        assert features[-1].shape[1] == 2048, f"Error jumlah channel c4: {features[-1].shape[1]}"
        print("✅ [PASS] Skenario 1: Forward Pass SolarSimCLR berhasil. Dimensi sesuai.")
    except Exception as e:
        print(f"❌ [FAIL] Skenario 1: {e}")
        
    # ---------------------------------------------------------
    # Skenario 2: Komputasi NT-Xent Loss
    # ---------------------------------------------------------
    try:
        print("\nSkenario 2: Komputasi NT-Xent Loss")
        bs = 4
        dim = 128
        criterion = NTXentLoss(batch_size=bs, temperature=0.5)
        
        z_i = torch.randn(bs, dim)
        z_j = torch.randn(bs, dim)
        
        loss = criterion(z_i, z_j)
        val = loss.item()
        
        print(f"✅ [PASS] Skenario 2: NT-Xent Loss berhasil dihitung. Nilai = {val:.4f}")
    except Exception as e:
        print(f"❌ [FAIL] Skenario 2: {e}")
        
    # ---------------------------------------------------------
    # Skenario 3: Forward Pass & Bipartite Loss Mask2Former
    # ---------------------------------------------------------
    try:
        print("\nSkenario 3: Forward Pass & Bipartite Loss Mask2Former")
        model_mask2former = FilamentMask2Former(num_classes=1)
        model_mask2former.eval()
        
        pixel_values = torch.randn(2, 1, 512, 512)
        
        labels = []
        for _ in range(2):
            num_instances = random.randint(1, 3) 
            class_labels = torch.zeros(num_instances, dtype=torch.long)
            mask_labels = torch.randint(0, 2, (num_instances, 512, 512)).float()
            labels.append({
                "class_labels": class_labels,
                "mask_labels": mask_labels
            })
            
        with torch.no_grad():
            outputs = model_mask2former(pixel_values, labels=labels)
            
        assert "loss" in outputs, "Error: Output tidak mengandung atribut loss."
        print(f"✅ [PASS] Skenario 3: Bipartite Loss berhasil dihitung. Nilai Loss = {outputs.loss.item():.4f}")
    except Exception as e:
        print(f"❌ [FAIL] Skenario 3: {e}")
        traceback.print_exc()

    # ---------------------------------------------------------
    # Skenario 4: Integrasi Data Asli (FITS) & Config YAML
    # ---------------------------------------------------------
    try:
        print("\nSkenario 4: Integrasi Dataloader FITS Asli & Config YAML")
        
        # 1. Load Config YAML hasil refaktor
        config = OmegaConf.load("config.yaml")
        
        # 2. Inisialisasi Dataset Asli
        import glob
        import os
        
        # Gunakan path dari config atau fallback ke direktori download sebelumnya
        data_dir = config.system.data_dir
        if not os.path.exists(data_dir) or len(glob.glob(os.path.join(data_dir, "*.fits"))) == 0:
            data_dir = "/mnt/ntfs/project/kaggle/data/gong_fits"
            
        fits_paths = glob.glob(os.path.join(data_dir, "**", "*.fits"), recursive=True)
        if not fits_paths:
            print("   -> ⚠️ Data FITS asli tidak ditemukan. Melewati Skenario 4.")
            return
            
        # Parameter disesuaikan dengan SolarSSLDataset (fits_paths, config, transform)
        dataset = SolarSSLDataset(
            fits_paths=fits_paths[:10],  # Ambil maksimal 10 file untuk tes cepat
            config=config,
            transform=None
        )
        
        # Ambil batch sangat kecil untuk sekadar tes I/O
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        # 3. Tarik 1 Batch Aktual
        batch = next(iter(dataloader))
        
        # Dataloader SSL biasanya mengembalikan dua view/crop augmentasi dari citra yang sama
        x_i, x_j = batch 
        
        print(f"   -> Tipe Data Tensor : {x_i.dtype} (Harus torch.float32)")
        print(f"   -> Dimensi View 1   : {x_i.shape}")
        print(f"   -> Rentang Piksel   : Min {x_i.min().item():.4f}, Max {x_i.max().item():.4f}")
        
        # 4. Forward Pass dengan Data FITS Asli
        model_simclr = SolarSimCLR(base_model=config.ssl_training.backbone)
        model_simclr.eval()
        with torch.no_grad():
            z_real, _ = model_simclr(x_i)
            
        assert z_real.shape == (2, config.ssl_training.latent_dim), f"Dimensi laten salah: {z_real.shape}"
        assert x_i.dtype == torch.float32, f"Tipe data salah, terdeteksi: {x_i.dtype}"
        
        print("✅ [PASS] Skenario 4: Dataloader FITS dan Integrasi Config berhasil berjalan mulus.")
        
    except Exception as e:
        print(f"❌ [FAIL] Skenario 4: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main_sanity_check()
