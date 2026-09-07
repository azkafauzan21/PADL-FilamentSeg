import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet18

class SolarSimCLR(nn.Module):
    def __init__(self, base_model='resnet50', latent_dim=128):
        super(SolarSimCLR, self).__init__()
        
        # Muat backbone (tanpa bobot pre-trained karena kita latih dari awal/domain berbeda)
        if base_model == 'resnet18':
            base_network = resnet18(weights=None)
            proj_in_dim = 512
        else:
            base_network = resnet50(weights=None)
            proj_in_dim = 2048
        
        # Modifikasi conv1 untuk menerima input 1-channel (grayscale FITS)
        # Mempertahankan argumen lain (kernel_size=7, stride=2, padding=3, bias=False) sama seperti ResNet asli
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = base_network.bn1
        self.relu = base_network.relu
        self.maxpool = base_network.maxpool
        
        # Ekstrak layer spasial
        self.layer1 = base_network.layer1
        self.layer2 = base_network.layer2
        self.layer3 = base_network.layer3
        self.layer4 = base_network.layer4
        
        self.avgpool = base_network.avgpool
        
        # Buat projector MLP (Linear -> ReLU -> Linear) untuk menghasilkan output laten
        # Dimensi fitur keluaran backbone berbeda-beda. Dimensi target (z) adalah 128.
        self.projector = nn.Sequential(
            nn.Linear(proj_in_dim, proj_in_dim),
            nn.ReLU(),
            nn.Linear(proj_in_dim, latent_dim)
        )
        
    def forward(self, x):
        # Melewati layer konvolusi awal
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        
        # Maxpooling sebelum masuk ke layer1
        p0 = self.maxpool(x0)
        
        # Ekstrak fitur spasial dari masing-masing block (untuk skip-connections di decoder)
        c1 = self.layer1(p0)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        
        # Hitung vektor representasi global untuk SimCLR
        pooled = self.avgpool(c4)
        pooled = torch.flatten(pooled, 1)
        
        # Proyeksikan ke ruang laten (untuk contrastive loss)
        z = self.projector(pooled)
        
        # Kembalikan laten z dan list fitur spasial
        return z, [x0, c1, c2, c3, c4]

if __name__ == "__main__":
    # Sanity check sederhana
    model = SolarSimCLR()
    
    # Batch berisi 8 citra FITS 1-channel dengan ukuran 512x512
    x = torch.randn(8, 1, 512, 512)
    
    # Forward pass
    z, features = model(x)
    
    print(f"Shape input (x): {x.shape}")
    print(f"Shape vektor laten (z): {z.shape}")
    print("-" * 50)
    print("Shape list spatial feature maps (untuk decoder):")
    for i, f in enumerate(features):
        print(f"  Feature {i} (c{i}): {f.shape}")
