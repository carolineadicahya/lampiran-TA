# Penerapan YOLOv11 dengan Instance Segmentation untuk Segmentasi Genangan Air pada Permukaan Ruas Jalan

Proyek ini menggunakan model **YOLOv11-Seg** dari Ultralytics untuk melakukan pelatihan model segmentasi instance terhadap citra genangan air dengan variasi YOLOv11-Nano dan YOLOv11-Small. Dataset yang digunakan telah dianotasi dalam format YOLO dan dikonfigurasi melalui file `data.yaml`.

---

### Cara Penggunaan

Untuk menjalankan pelatihan model ini, pastikan semua dependensi sudah terpasang dan dataset tersedia di direktori yang sesuai. Berikut langkah-langkahnya:

1. Pastikan file `data.yaml` terletak di `dataset puddle/`
2. Jalankan notebook `training.....ipynb`
3. Model akan memulai pelatihan selama 300 epoch
4. Hasil pelatihan disimpan otomatis di folder `runs/segment/train/`

---

### Contoh Sederhana

```python
from ultralytics import YOLO

model = YOLO("yolo11s-seg.pt")  # segmentasi YOLOv8m, kemudian variasi bisa diganti dengan YOLOv8n YOLOv8s
model.train(
    data="dataset puddle/data.yaml",  # Path ke konfigurasi dataset
    epochs=300,
    optimizer="SGD",
    lr0=0.01,
    momentum=0.9,
)
```

---

## Konfigurasi Dataset (`data.yaml`)

```yaml
path: ../dataset puddle
train: images/train
val: images/val

nc: 1
names: ["puddle"]
```

> Catatan: Format folder harus mengikuti struktur YOLO dengan folder `images/train`, `images/val`, dan folder `labels` untuk anotasi.

---

### Metodologi

- Dataset diperoleh dari hasil dokumentasi genangan air di jalan raya dan dilakukan augmentasi (brightness, noise, blur, exposure).
- Model dilatih menggunakan YOLOv11-Nano dan YOLOv11-Small.
- Optimizer yang digunakan: **AdamW**, **SGD**, dan **Auto**.

---

## Contoh Training

| Parameter   | Nilai          | Keterangan                                |
| ----------- | -------------- | ----------------------------------------- |
| `model`     | yolo11s-seg.pt | Model YOLO11 versi small untuk segmentasi |
| `epochs`    | 300            | Jumlah epoch pelatihan                    |
| `optimizer` | SGD            | Algoritma optimasi                        |
| `lr0`       | 0.01           | Learning rate awal                        |

---

## Skenario Hyperparameter

| Skenario   | Optimizer | Learning Rate Awal | Momentum |
| ---------- | --------- | ------------------ | -------- |
| Skenario 1 | Auto      | 0.01               | 0.937    |
| Skenario 2 | AdamW     | 0.002              | 0.9      |
| Skenario 3 | SGD       | 0.01               | 0.9      |

---

## Arsitektur Model

Model `yolov11-seg.pt` memiliki arsitektur yang terdiri dari tiga komponen utama, yaitu backbone, neck, dan head, yang masing-masing memiliki fungsi spesifik.

- **Backbone**: Mengekstraksi fitur dari citra menggunakan kombinasi konvolusi 3×3, _C3k2 Blocks_ (versi efisien dari C2F pada YOLOv8), _SPPF_ untuk skala multi-level, dan _C2PSA_ untuk fokus spasial.
- **Neck**: Menghubungkan backbone dan head, meningkatkan resolusi fitur melalui upsampling dan concat untuk mendeteksi objek kecil, serta mengintegrasikan informasi dari berbagai level.
- **Head**: Menghasilkan prediksi akhir berupa _bounding box_, kelas, dan segmentasi objek dengan memanfaatkan fitur dari neck.

---

## Hasil Pelatihan

Setelah training selesai, hasil akan tersimpan di folder:

```
runs/segment/train5/
├── results.png         # Grafik hasil loss dan mAP
├── weights/
│   ├── best.pt         # Model dengan performa terbaik
│   └── last.pt         # Model hasil epoch terakhir
```

---

## Evaluasi Model

Untuk melakukan evaluasi:

```python
model.val()
```

---

## Catatan Tambahan

- Gunakan GPU untuk performa maksimal
- Untuk prediksi atau inference, gunakan model `best.pt` dari hasil pelatihan
- Pastikan format anotasi label sesuai standar YOLO11 (segmentation polygon format)

---

## Lingkungan Pengujian

- **Ultralytics**: 8.3.95
- **Python**: 3.11.9
- **Torch**: 2.5.1 + cu124
- **GPU**: NVIDIA GeForce RTX 3090 (24GB)

---

## Penulis

Caroline Adi Cahya
Institut Teknologi Kalimantan  
2025
