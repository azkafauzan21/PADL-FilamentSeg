# 🏛️ Arsitektur PADL-FilamentSeg

Dokumen ini membedah rancang bangun (arsitektur) di balik repositori `PADL-FilamentSeg`. Sistem ini didesain khusus untuk segmentasi filamen matahari (H-alpha) menggunakan penggabungan algoritma *Self-Supervised Learning* (SSL) dan *Supervised Fine-Tuning* berbasis Mask2Former, sambil tetap mematuhi hukum fisika astronomi (seperti pelestarian chiralitas).

---

## 1. Arsitektur Level Tinggi (Dual-Pipeline)

Sistem beroperasi melalui dua jalur aliran data diskrit yang bekerja secara berurutan:

1. **Jalur Pra-pelatihan (FITS - SSL):**
   Model dilatih terlebih dahulu tanpa label menggunakan citra mentah astronomi format FITS. Tujuannya adalah mengajarkan *backbone* konvolusional untuk mengenali secara intrinsik "apa itu fitur matahari" (tekstur, bintik matahari, filamen) murni dari intensitas piksel H-alpha, tanpa bias dari anotator manusia.
2. **Jalur Supervised Fine-Tuning (JPEG COCO - Segmentasi):**
   Setelah representasi fisika dikuasai, *backbone* dipindahkan ke dalam arsitektur segmentasi (Mask2Former). Pada tahap ini, model diajarkan untuk menarik garis batas (masking) yang akurat menggunakan dataset kompetisi Kaggle (berupa JPEG 8-bit tersupervisi) yang diformat dalam struktur COCO JSON.

**Jembatan Utama:** Bobot yang diekstraksi dari jalur SSL di-hand-off ke jalur Supervised, menghasilkan model yang tidak hanya mengerti instruksi anotasi manusia, tetapi juga memahami fisika citra matahari di level fundamental.

---

## 2. Anatomi Model (Aliran Tensor)

### A. SolarSimCLR (Ekstraksi Fitur Fisika)
`SolarSimCLR` adalah backbone (ResNet50) yang dimodifikasi. 
* **Input (1, H, W):** Tensor 1-channel dimasukkan ke model (FITS Grayscale).
* **Modifikasi Awal:** Layer `conv1` standar ResNet (berharap 3-channel RGB) telah dioverride untuk langsung memproses matriks 1-channel secara *native*.
* **Output Ganda:**
  1. **Z-Vector:** Melewati proyektor MLP (Linear -> ReLU -> Linear), fitur dipadatkan menjadi vektor $z$ (dimensi 128) khusus untuk fungsi kerugian (Contrastive Loss) di fase SSL.
  2. **Spatial Feature Maps:** Sebelum *global pooling*, model mengekstrak peta fitur perantara secara berurutan `[x0, c1, c2, c3, c4]`. Kumpulan resolusi hirarkis ini (256, 512, 1024, 2048 channel) dirancang untuk dikonsumsi oleh decoder di tahap supervised sebagai *skip-connections*.

### B. FilamentMask2Former (Integrasi Pixel Decoder)
Hugging Face `Mask2FormerForUniversalSegmentation` aslinya membutuhkan *backbone* internal (seperti Swin atau ResNet bawaan).
* Di dalam `FilamentMask2Former`, kita menggunakan objek *DummyEncoder* untuk melewati pembuatan *backbone* bawaan Hugging Face.
* Peta spasial `[x0, c1, c2, c3, c4]` hasil dari `SolarSimCLR` disuntikkan secara manual langsung ke dalam komponen *Pixel Decoder* (Transformer-based) Mask2Former. Ini memungkinkan Mask2Former untuk menghasilkan *mask queries* dari fitur berbekal fisika yang kita latih sendiri.

---

## 3. Dataloaders & Transformasi Fisika

### Normalisasi H-alpha (Kliping 99th Percentile)
Citra astronomi rentan terhadap derau sekunder (seperti *cosmic rays* atau instrumen *flares*) yang menyebabkan lonjakan nilai intensitas piksel tak beraturan. 
Oleh sebab itu, `dataset_ssl.py` & `dataset_supervised.py` mengadopsi algoritma **Percentile Clipping**: 
1. Menghitung persentil ke-1 dan ke-99 dari gambar.
2. Melakukan *clip* intensitas agar nilai *outlier* dipaksa masuk ke batas rentang wajar.
3. Melakukan normalisasi ke skala tensor `[0, 1]`.

### Larangan Keras: RandomFlip (Chirality Preservation)
Filamen matahari memiliki atribut fisis yang disebut *chirality* (dextral/sinistral) yang mendefinisikan arah rotasi magnetik mereka. Membalik gambar secara horizontal atau vertikal (Flip) **akan mengubah hukum fisika yang terekam pada filamen tersebut**. 
*Oleh karenanya, augmentasi dua-pandangan SimCLR kita dibatasi pada translasi, pergeseran kontras, dan distorsi, dengan pelarangan absolut terhadap permutasi rotasi balik (No RandomFlip).*

### Dekoding Mask Supervised (COCO API)
Dataset Kaggle memuat label berformat RLE (Run-Length Encoding) di dalam file JSON. 
Di dalam `dataset_supervised.py`, kita mengutus algoritma `pycocotools` untuk:
1. Membaca anotasi instance per *image_id*.
2. Memutar struktur RLE/Polygon asli menggunakan fungsi `.annToMask()` menjadi *array* spasial (H, W).
3. Mengumpulkan semuanya menjadi satu tumpukan target tensor biner (N, H, W) untuk diteruskan ke perhitungan loss.

---

## 4. Algoritma Training & Loss

### A. NT-Xent Loss (Self-Supervised Phase)
`NTXentLoss` membedah matriks probabilitas *Contrastive Learning*.
* Menerima vektor $z_i$ (augmentasi 1) dan $z_j$ (augmentasi 2).
* Melakukan dot product untuk membangun *similarity matrix* terhadap seluruh sampel dalam satu batch komputasi (memakai operasi matmul PyTorch murni demi akselerasi GPU).
* *Diagonal Masking* diterapkan untuk mencegah model membandingkan representasi gambar dengan dirinya sendiri yang identik.
* Target dirumuskan: model dihukum *(cross-entropy)* jika gagal mendekatkan vektor dari dua augmentasi gambar yang berasal dari sumber astronomi fisik yang sama.

### B. Bipartite Matching Loss (Supervised Phase)
Berkat modifikasi *parameter pass*, argumen `labels` (berisi *list of dicts* mask biner COCO) dioper ke rutin *forward pass* Hugging Face. 
* Mask2Former mengeksekusi **Hungarian Algorithm (Bipartite Matching)** di internal modul komputasi HF-nya untuk menetapkan 1-ke-1 dari mask prediksi ($N_{queries} = 100$) dengan mask kebenaran target (*ground truth*).
* Model kemudian meminimalkan komposit *Cross Entropy* (untuk klasifikasi filamen) dan perpaduan *Dice/Focal Loss* (untuk bentuk mask piksel yang akurat).

---

## 5. Siklus Hidup Eksekusi (CLI Orchestration)

Skrip `main.py` menggunakan `argparse` bertindak sebagai komandan tertinggi komputasi *pipeline*.

```text
                        [ CLI: main.py --mode ... ]
                                     |
    +--------------------------------+--------------------------------+
    |                                |                                |
[extract_data]                 [pretrain_simclr]             [train_mask2former]
    |                                |                                |
(1) Parse JSON Kaggle           (1) Load FITS Dataset          (1) Load JPEG Dataset
(2) Export target CSV           (2) Setup SolarSimCLR          (2) Load 'simclr_final.pth'
(3) Train/Val Anti-Leakage      (3) Contrastive Epoch Loop     (3) Setup FilamentMask2Former
(4) Create split JSONs          (4) Save 'simclr_final.pth'    (4) Inject Weights
    |                                |                         (5) Bipartite Matching Epochs
[ DONE ]                        [ HANDOFF TO 03 ]              (6) Save 'mask2former_final.pth'
```

### Mekanisme Estafet (Weight Handoff)
1. Setelah `pretrain_simclr` rampung, ia mengekspor matriks bobot tensor ke *disk* (`simclr_final.pth`).
2. Ketika mode `train_mask2former` dipanggil, skrip memeriksa ekstensi bobot ini. 
3. *State dict* diinjeksi secara spesifik ke dalam `model.backbone` milik Mask2Former. Pemuatan memilah (*strict mapping*) hanya pada modul konvolusi ResNet yang bersesuaian, memberikan "pemahaman astronomi" yang telah matang untuk menuntun modul dekoder (transformer).
