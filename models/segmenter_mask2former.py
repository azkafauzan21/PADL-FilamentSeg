import torch
import torch.nn as nn
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

try:
    from .backbone_simclr import SolarSimCLR
except ImportError:
    from backbone_simclr import SolarSimCLR


class DummyEncoderOutput:
    """Objek output minimal yang kompatibel dengan interface HuggingFace Mask2Former."""

    def __init__(self, feature_maps):
        self.feature_maps = feature_maps


class DummyEncoder(nn.Module):
    """
    Modul bypass encoder bawaan Mask2Former.

    feature_maps diset dari FilamentMask2Former.forward() sebelum memanggil
    self.mask2former(), kemudian dikembalikan saat PixelLevelModule memanggil
    encoder ini.
    """

    def __init__(self):
        super().__init__()
        self.feature_maps = None

    def forward(self, pixel_values, output_hidden_states=False, return_dict=True):
        return DummyEncoderOutput(self.feature_maps)


class FilamentMask2Former(nn.Module):
    """
    Model segmentasi filamen matahari — SolarSimCLR backbone + Mask2Former decoder.

    ──────────────────────────────────────────────────────────────
    Alur Data & Patch Arsitektur (Dynamic Shape Inference)
    ──────────────────────────────────────────────────────────────
    Model ini menggunakan arsitektur agnostik yang secara otomatis mengukur 
    dan beradaptasi dengan dimensi matriks dari backbone apa pun (misal: ResNet34, ResNet50).
    
    SolarSimCLR menghasilkan 5 feature maps:
        x0: [B, C0, H/2,  W/2]   (post-conv1, sebelum layer1)
        c1: [B, C1, H/4,  W/4]   (layer1)
        c2: [B, C2, H/8,  W/8]   (layer2)
        c3: [B, C3, H/16, W/16]  (layer3)
        c4: [B, C4, H/32, W/32]  (layer4)
        
        *Catatan Dimensi (C): 
         - ResNet34: C1=64, C2=128, C3=256, C4=512
         - ResNet50: C1=256, C2=512, C3=1024, C4=2048

    Kita kirim 4 fitur ke DummyEncoder dalam urutan fine→coarse:
        feature_maps = (c1, c2, c3, c4)

    Mask2Former PixelDecoder (dari kode HF line 1336 & 1395):

        1) input_projections → `features[::-1][:num_feature_levels]`
           Dengan num_feature_levels=3, fitur yang diproses (setelah pembalikan):
               level=0 ← c4 (terdalam) → patch Conv2d(C4 → 256)
               level=1 ← c3            → patch Conv2d(C3 → 256)
               level=2 ← c2            → patch Conv2d(C2 → 256)

        2) lateral_convolutions → `features[:num_fpn_levels][::-1]`
           Dengan num_fpn_levels=1, fitur yang diproses:
               idx=0 ← c1 (terdangkal) → patch Conv2d(C1 → 256)

    Dengan kata lain, pengiriman (c1, c2, c3, c4) fine→coarse memastikan:
        - c4 menjadi features[-1] → diambil pertama setelah [::-1]
        - c1 menjadi features[0]  → diambil untuk FPN lateral
        
    Kapasitas saluran (channels) diukur secara otomatis saat inisialisasi awal 
    menggunakan tensor dummy, menghilangkan kebutuhan hardcode matriks dimensi.
    """

    def __init__(self, num_classes=4, base_model='resnet34'):
        super().__init__()

        # 1. Backbone SolarSimCLR (terima parameter nama model)
        self.backbone = SolarSimCLR(base_model=base_model)
        
        # --- [ADAPTASI OTOMATIS: Dynamic Shape Inference] ---
        dummy_x = torch.randn(1, 1, 256, 256)
        self.backbone.eval()
        with torch.no_grad():
            _, features = self.backbone(dummy_x)
        
        # features = [x0, c1, c2, c3, c4]. Kita ukur channel (dimensi ke-1) dari c1-c4.
        self.BACKBONE_CHANNELS = [f.shape[1] for f in features[1:]]
        self.backbone.train()
        # ----------------------------------------------------

        # 2. Konfigurasi Mask2Former tanpa backbone bawaan
        config = Mask2FormerConfig(
            num_queries=100,
            num_labels=num_classes,
            use_pretrained_backbone=False,
        )
        self.mask2former = Mask2FormerForUniversalSegmentation(config)

        # 3. Patch konvolusi PixelDecoder agar menerima channel dari backbone kita.
        #    Patch dilakukan post-init karena Mask2FormerConfig tidak meng-expose
        #    hidden_sizes backbone secara langsung.
        self._patch_pixel_decoder_projections()

        # 4. Pasang DummyEncoder — bypass encoder bawaan HuggingFace
        self.dummy_encoder = DummyEncoder()
        self.mask2former.model.pixel_level_module.encoder = self.dummy_encoder

    def _patch_pixel_decoder_projections(self):
        """
        Patch semua Conv2d proyeksi di PixelDecoder agar menerima channel ResNet50.

        PixelDecoder memiliki dua set projection yang perlu dicocokan:
          - input_projections[i]: memproses features[::-1][i]
                                  → (c4, c3, c2) = (2048, 1024, 512)
          - lateral_convolutions[i]: memproses features[:i+1][::-1][i]
                                      → (c1,) = (256,)
        """
        pixel_decoder = self.mask2former.model.pixel_level_module.decoder
        n_feat = pixel_decoder.num_feature_levels   # 3
        n_fpn  = pixel_decoder.num_fpn_levels       # 1

        # Verifikasi jumlah level sesuai ekspektasi arsitektur
        total_expected = n_feat + n_fpn  # 4 = jumlah stage ResNet50 yang kita pakai
        assert total_expected == len(self.BACKBONE_CHANNELS), (
            f"PixelDecoder total levels ({n_feat}+{n_fpn}={total_expected}) "
            f"!= jumlah backbone channels ({len(self.BACKBONE_CHANNELS)}). "
            f"Verifikasi konfigurasi Mask2Former atau BACKBONE_CHANNELS."
        )

        # -- Patch input_projections [i] ← features[::-1][i] = (c4, c3, c2) --
        # Channel yang diharapkan setelah pembalikan: BACKBONE_CHANNELS dibalik,
        # dimulai dari yang terdalam (c4=2048) ke yang lebih dangkal (c2=512).
        reversed_channels = list(reversed(self.BACKBONE_CHANNELS))  # [2048, 1024, 512, 256]
        for i, proj_seq in enumerate(pixel_decoder.input_projections):
            assert isinstance(proj_seq[0], nn.Conv2d), (
                f"input_projections[{i}][0] bukan Conv2d: {type(proj_seq[0])}"
            )
            out_ch = proj_seq[0].out_channels  # feature_size, biasanya 256
            in_ch  = reversed_channels[i]       # channel ResNet50 yang sesuai
            proj_seq[0] = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

        # -- Patch lateral_convolutions [i] ← features[:n_fpn][::-1][i] = (c1,) --
        # features[:1] = (c1,); features[:1][::-1] = (c1,)
        # lateral_convolutions[0] harus menerima c1 = 256 channels
        fpn_channels = self.BACKBONE_CHANNELS[:n_fpn]  # [256] (hanya c1)
        fpn_channels_reversed = list(reversed(fpn_channels))  # [256]
        for i, lateral_conv in enumerate(pixel_decoder.lateral_convolutions):
            # lateral_conv bisa Sequential atau Conv2d langsung
            if isinstance(lateral_conv, nn.Sequential):
                target = lateral_conv[0]
                is_seq  = True
            else:
                target  = lateral_conv
                is_seq  = False

            assert isinstance(target, nn.Conv2d), (
                f"lateral_convolutions[{i}] Conv2d tidak ditemukan: {type(target)}"
            )
            out_ch = target.out_channels
            in_ch  = fpn_channels_reversed[i]  # channel ResNet50 yang sesuai

            new_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
            if is_seq:
                lateral_conv[0] = new_conv
            else:
                pixel_decoder.lateral_convolutions[i] = new_conv

    def forward(self, pixel_values, labels=None):
        # A. Ekstrak feature maps dari backbone SolarSimCLR
        z, features = self.backbone(pixel_values)
        # features = [x0(64), c1(256), c2(512), c3(1024), c4(2048)]

        # B. Kirim 4 fitur dalam urutan fine→coarse: (c1, c2, c3, c4)
        #    Urutan ini KRITIS karena PixelDecoder membaliknya secara internal:
        #      features[::-1][:3] = (c4, c3, c2) → masuk ke input_projections
        #      features[:1][::-1] = (c1,)         → masuk ke lateral_convolutions
        c1, c2, c3, c4 = features[1], features[2], features[3], features[4]
        self.dummy_encoder.feature_maps = (c1, c2, c3, c4)

        # C. Forward Mask2Former
        if labels is not None:
            # Mode training: HuggingFace menghitung loss secara internal
            mask_labels  = [t["mask_labels"]  for t in labels]
            class_labels = [t["class_labels"] for t in labels]
            return self.mask2former(
                pixel_values=pixel_values,
                mask_labels=mask_labels,
                class_labels=class_labels,
            )
        else:
            # Mode inferensi
            outputs = self.mask2former(pixel_values=pixel_values)
            return outputs.masks_queries_logits, outputs.class_queries_logits


if __name__ == "__main__":
    print("Inisialisasi FilamentMask2Former...")
    model = FilamentMask2Former(num_classes=4)

    x = torch.randn(1, 1, 256, 256)
    print(f"Forward pass dengan input shape: {x.shape}...")
    with torch.no_grad():
        mask_logits, class_logits = model(x)

    print("Berhasil!")
    print(f"  mask_logits  shape: {mask_logits.shape}")
    print(f"  class_logits shape: {class_logits.shape}")
    print("Tidak ada RuntimeError — forward pass sukses.")
