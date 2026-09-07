import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super(NTXentLoss, self).__init__()
        # batch_size dapat disimpan sebagai referensi, 
        # namun kita mengambil batch_size dinamis di forward pass (menghindari error di batch terakhir)
        self.batch_size = batch_size
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, z_i, z_j):
        """
        z_i: Tensor representasi laten dari view 1 (augmented 1), shape [batch_size, dim]
        z_j: Tensor representasi laten dari view 2 (augmented 2), shape [batch_size, dim]
        """
        device = z_i.device
        current_batch_size = z_i.shape[0] 
        
        # 1. Normalisasi L2 (agar dot product otomatis menjadi cosine similarity)
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # 2. Gabungkan z_i dan z_j
        # z akan berdimensi [2*batch_size, dim]
        # Urutan: [z_i_1, ..., z_i_N, z_j_1, ..., z_j_N]
        z = torch.cat([z_i, z_j], dim=0)
        
        # 3. Hitung scaled cosine similarity matrix
        # Shape: [2*batch_size, 2*batch_size]
        similarity_matrix = torch.matmul(z, z.T) / self.temperature
        
        # 4. Masking diagonal utama (menghilangkan similarity dengan diri sendiri)
        mask = torch.eye(2 * current_batch_size, dtype=torch.bool, device=device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)
        
        # 5. Buat target labels
        # Sampel ke-i (dari z_i) punya pasangan positif di indeks (i + batch_size) (yaitu z_j)
        # Sampel ke-i (dari z_j) punya pasangan positif di indeks (i - batch_size) (yaitu z_i)
        labels = torch.cat([
            torch.arange(current_batch_size) + current_batch_size, 
            torch.arange(current_batch_size)
        ], dim=0).to(device)
        
        # 6. Hitung CrossEntropyLoss 
        # (tiap baris di similarity_matrix dianggap sebagai logits klasifikasi terhadap 2*batch_size kelas)
        loss = self.criterion(similarity_matrix, labels)
        
        return loss

if __name__ == "__main__":
    # Sanity check sederhana
    bs = 8
    dim = 128
    
    criterion = NTXentLoss(batch_size=bs, temperature=0.5)
    
    # Dummy laten vectors
    z1 = torch.randn(bs, dim)
    z2 = torch.randn(bs, dim)
    
    loss_val = criterion(z1, z2)
    print(f"NT-Xent Loss val: {loss_val.item():.4f}")
