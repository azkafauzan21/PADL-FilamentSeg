# 🌌 Physics-Aware Deep Learning: Rasionalitas Heliofisika dalam PADL-FilamentSeg

Dokumen ini menguraikan dasar pemikiran heliofisika di balik rancangan sistem *PADL-FilamentSeg*. Kami menjelaskan secara ilmiah *mengapa* integrasi konstrain fisika ke dalam arsitektur kecerdasan buatan (Physics-Aware Deep Learning) jauh lebih superior dibandingkan model *Computer Vision* (CV) konvensional untuk tugas analisis citra matahari.

---

## 1. Mengapa Computer Vision Standar Gagal di Domain Heliofisika?

Model segmentasi semantik komersial (seperti YOLO, Mask R-CNN, atau U-Net standar) umumnya dilatih menggunakan dataset gambar kehidupan sehari-hari (COCO, ImageNet) yang berupa citra RGB 3-channel. Ketika model-model ini diaplikasikan langsung pada citra matahari H-alpha, mereka sering mengalami kegagalan fundamental:

* **Absennya Batas Tepi yang Tegas (Fuzzy Boundaries):** Objek sehari-hari memiliki gradien piksel yang tajam (seperti tepi mobil atau garis meja). Sebaliknya, filamen matahari adalah awan plasma dingin dan padat yang mengambang di atas kromosfer, menyebabkan batas strukturnya menjadi sangat kabur (*fuzzy*) dan sangat bergantung pada distribusi suhu serta densitas lokal.
* **Interferensi Fitur Aktif:** Permukaan matahari (*solar disk*) tidaklah statis. Varian intensitas akibat bintik matahari (*sunspots*), suar (*solar flares*), dan penggelapan tepi (*limb darkening*) dapat mengecoh model CV standar, menyebabkan alarm palsu (*false positives*) karena model tersebut tidak dilatih untuk memahami konteks radiatif dan penyerapan plasma.

---

## 2. Filosofi Dual-Pipeline: Keunggulan Termodinamika Format FITS

Sebagai solusi, PADL-FilamentSeg mengadopsi struktur *Dual-Pipeline* dengan menolak penggunaan murni gambar JPEG berukuran 8-bit dari dataset Kaggle untuk tahapan *pre-training* (fase pengenalan dasar). 

**Mengapa penggunaan FITS sangat vital?**
Citra *Flexible Image Transport System* (FITS) dari instrumen astronomi seperti jaringan GONG tidak sekadar mempresentasikan gambar visual (layaknya JPEG), melainkan **merekam rentang dinamis (dynamic range) termodinamika plasma yang sebenarnya (dalam skala 16-bit atau 32-bit float)**. Kompresi gambar dari teleskop menjadi matriks 8-bit JPEG menghilangkan gradasi piksel halus yang sangat penting untuk membedakan kedalaman optis dan kerapatan filamen.

Melalui pra-pelatihan Self-Supervised Learning (SSL) menggunakan arsitektur SimCLR berinput 1-channel secara langsung pada file FITS, kita mengajarkan model untuk mendeteksi fluktuasi intensitas piksel yang merepresentasikan kuantitas fisik energi yang nyata, bukan sekadar nilai RGB layar yang telah dikompresi berlebihan.

---

## 3. Augmentasi Data yang "Sadar Fisika" (Physics-Aware Augmentation)

Pendekatan *Physics-Aware* paling krusial dan menonjol terlihat pada fase penyusunan transformasi augmentasi data.

### Larangan Absolut Terhadap Permutasi *RandomFlip*
Dalam visi komputer klasik, membalik gambar secara horizontal atau vertikal (*Flip*) lazim digunakan untuk memperbanyak sampel data tanpa merusak konteks objek dasar (sebuah mobil tetaplah mobil meskipun gambar dibalik 180 derajat). Namun, dalam fisika matahari, struktur filamen memiliki sifat **Chirality (Kiralitas)**.

Filamen matahari didasari oleh untaian pita magnetik dengan arah putaran heliks tertentu (*Left-handed* / sinistral atau *Right-handed* / dextral). Memutarbalik matriks gambar berarti **memalsukan hukum fisika orientasi magnetik** yang direkam oleh instrumen. Tindakan augmentasi sembrono ini akan merusak korelasi spasial magnetik sejati yang justru sedang dipelajari oleh model *Deep Learning* kita.

### Astronomi Ekstrem: Normalisasi Kliping Persentil 99%
Observasi satelit maupun teleskop dari permukaan bumi (*ground-based*) senantiasa terpapar radiasi acak dari partikel berenergi tinggi yang memicu artefak (*cosmic rays*) atau rekaman emisi suar (*solar flares*) sesaat yang bersifat impulsif, mendominasi nilai intensitas piksel maksimum.
Kami menggunakan algoritma kliping pada limit persentil ke-1 dan ke-99 sebagai teknik kurasi astronomis standar. Pendekatan ini mengisolasi varian *noise* frekuensi tinggi sekunder, memastikan matriks laten dari jaringan saraf internal hanya mengalokasikan bobot kognisinya pada rentang struktur termodinamika filamen yang stabil secara optis.

---

## 4. NT-Xent Loss sebagai "Penangkap Distribusi Plasma"

Pada fase SSL, kita menerapkan fungsi kerugian *Normalized Temperature-scaled Cross Entropy Loss* (NT-Xent) untuk model SimCLR. Dalam kacamata konteks fisika matahari:

Fungsi perhitungan *Loss* ini secara agresif memaksa representasi vektor laten ($z$) dari model untuk mengkalkulasi aglomerasi (mendekatkan jarak vektor secara kosinus) terhadap potongan-potongan awan filamen dengan topologi yang identik, meskipun terdapat distorsi kondisi pencahayaan observasional yang ekstrem (sebagai contoh, saat struktur filamen tersorot penuh di ekuator tengah piringan matahari vs ketika filamen tersamarkan di zona penggelapan tepi atmosferik atau *limb darkening*). 
Model "dipaksa" mereduksi sensitivitasnya terhadap variasi fluks cahaya sesaat instrumen optik, guna berfokus sepenuhnya mencari dan mengukuhkan invarian pola-pola topologi dari awan plasma gas hidrogen terionisasi.

---

## 5. Nilai Tambah Kaggle: Preservasi Integritas "Spine"

Pada fase evolusi arsitektur selanjutnya (*Supervised Fine-Tuning*), algoritma kita transisi menggunakan ketersediaan anotasi ahli dari kompetisi Kaggle. Salah satu penemuan dan inovasi penanganan data penting dalam repositori ini adalah rekayasa perangkat lunak untuk menjaga dan melestarikan *key* kustom bernama `"spine"` pada restrukturisasi file JSON COCO kita di `01_extract_metadata.py`.

*Spine* (tulang punggung) merepresentasikan kelengkungan kerangka utama dari struktur kurva pita filamen (biasanya berkaitan dengan penanda Garis Netral Polaritas Magnetik - *Polarity Inversion Line*). 
Melestarikan struktur ini (menghindarkannya dari pembersihan pemangkasan standar COCO Parser) menjadikan model PADL-FilamentSeg melampaui peran sekadar mesin lokalisasi batas tepi piksel. Aksesibelnya data geometri *spine* mentah akan memampukan adaptasi masa depan model ini (seperti analisis metrik cuaca antariksa dan predisposisi kelengkungan awal letusan badai *Coronal Mass Ejection*) karena ia akan bertumpu langsung pada rasionalitas tulang kerangka kelengkungan fisika sejati sang filamen.
