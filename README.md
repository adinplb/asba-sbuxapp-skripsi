# Klasifikasi Fine-Grained Aspek Usability Ulasan Aplikasi Starbucks (Google Play Store)

> Klasifikasi ulasan pengguna ke dalam **4 aspek usability** (bukan sentimen positif/negatif) menggunakan pendekatan rule-based silver-labelling + SVM & Conv1D (deep learning).

---

## Daftar Isi

1. [Gambaran Proyek](#gambaran-proyek)
2. [Struktur Direktori](#struktur-direktori)
3. [Dataset](#dataset)
4. [Pipeline Pelabelan](#pipeline-pelabelan)
5. [3 Skema Pelatihan](#3-skema-pelatihan)
6. [Inference](#inference)
7. [Cara Menjalankan](#cara-menjalankan)
8. [Requirements](#requirements)

---

## Gambaran Proyek

Proyek ini mengklasifikasikan ulasan aplikasi Starbucks Indonesia (`com.starbucks.id`) dan Starbucks Global (`com.starbucks.mobilecard`) di Google Play Store ke dalam **4 kelas aspek usability**:

| Aspek | Deskripsi |
|-------|-----------|
| **Errors** | Kegagalan fungsional konkret: gagal login/sign in/sign up, OTP gagal, force close/crash, saldo hilang, "internal server error", app stuck/hang |
| **Learnability** | Kebingungan cara memakai aplikasi: tidak paham kegunaan field/kode referral/PIN, navigasi/menu membingungkan |
| **Efficiency** | Kecepatan & kepraktisan: lambat/berat/loading lama, baterai boros, hemat/buang waktu |
| **Satisfaction** | Opini/verdict umum: pujian atau kekecewaan generik, reward/promo/poin, customer service, permintaan fitur |

**Metodologi:**
- **Scraping**: `google-play-scraper`, multi-source (ID + EN), 11.588 ulasan unik
- **Translasi**: ulasan berbahasa Inggris diterjemahkan ke Bahasa Indonesia (MarianMT) agar satu bahasa
- **Pelabelan**: rule-based (kata kunci/regex per aspek) yang didefinisikan dari 1.500 anotasi manual, lalu difilter dengan *confidence score* per kelas untuk membuang label yang bising
- **Preprocessing**: cleaning → normalisasi slang → stopword removal → stemming (PySastrawi)
- **3 Skema Pelatihan**: SVM+TF-IDF+80/20 | Conv1D (deep learning)+Word Embedding+80/20 | SVM+TF-IDF+70/30

Catatan penting soal alur data: dataset anotasi 1.500 sampel (`labelling_aspek_starbucks_app - scrapped_data1500_sbux.csv.csv`) **tidak** dipakai langsung untuk melatih model — dataset itu hanya dipakai untuk mempelajari definisi & kata kunci tiap aspek (lihat notebook, bagian awal). Model dilatih di atas seluruh 11.588 baris raw yang dilabeli rule-based, kemudian difilter berdasarkan confidence kecocokan kata kunci, menyisakan **5.021 baris** yang jauh lebih bersih (`dataset/aspect_labeled_dataset.csv`) — inilah yang dipakai untuk training 3 skema.

Detail lengkap alasan desain (kenapa Conv1D dan bukan BiLSTM, kenapa threshold confidence berbeda per kelas, dsb.) ada langsung sebagai markdown cells di `sentiment_analysis_training.ipynb`.

---

## Struktur Direktori

```
sentiment-analysis/
|-- scraping.py                              # Script scraping Google Play Store
|-- sentiment_analysis_training.ipynb        # Notebook utama (pelabelan + preprocessing + 3 skema)
|-- requirements.txt                         # Dependensi Python
|-- kriteria.txt                             # Kriteria submission tugas
|-- scraping.log                             # Log hasil scraping
|-- labelling_aspek_starbucks_app - scrapped_data1500_sbux.csv.csv
|                                             # 1.500 anotasi manual (dipakai untuk mempelajari kata kunci aspek)
|-- starbucks_reviews_raw.csv                 # Salinan dataset scraping mentah (root)
|
`-- dataset/
    |-- starbucks_reviews_raw.csv            # 11.588 ulasan hasil scraping (ID + EN)
    |-- raw_translated.csv                   # Setelah translasi EN -> ID
    |-- aspect_labeled_dataset.csv           # 5.021 baris setelah rule-based labelling + filter confidence
    `-- kamus_normalisasi.csv                # Kamus normalisasi kata slang -> baku
```

> Folder `reference/` (jika ada secara lokal) adalah clone repo referensi terpisah (proyek sentimen 3-kelas sebelumnya) dan sengaja tidak disertakan dalam repo ini — lihat `.gitignore`.

---

## Dataset

### Proses Scraping (`scraping.py`)

| Sumber | App ID | Bahasa | Negara |
|--------|--------|--------|--------|
| Starbucks Indonesia (ID) | `com.starbucks.id` | id | id |
| Starbucks Indonesia (EN) | `com.starbucks.id` | en | id |
| Starbucks Global (US) | `com.starbucks.mobilecard` | en | us |

Total **11.588** ulasan unik (deduplikasi berdasarkan `reviewId`). Kolom output: `reviewId`, `userName`, `content`, `score`, `thumbsUpCount`, `at`, `app_source`.

### Distribusi Label Aspek (setelah filter confidence, dasar training)

| Aspek | Jumlah |
|-------|--------|
| Satisfaction | 2.830 |
| Errors | 1.247 |
| Efficiency | 516 |
| Learnability | 428 |
| **Total** | **5.021** |

---

## Pipeline Pelabelan

```
Ulasan Mentah (ID + EN)
  |
  |-- 1. Translasi EN -> ID (MarianMT, jika perlu)
  |-- 2. Cleaning + normalisasi slang (kamus_normalisasi.csv)
  |-- 3. Label rule-based per aspek (regex/kata kunci, prioritas Errors > Learnability > Efficiency > Satisfaction)
  |-- 4. Filter confidence per kelas (buang baris paling bising/ambigu)
  `-- 5. Stopword removal + stemming (PySastrawi) -> kolom 'stemming' (dipakai TF-IDF & tokenizer)
         |
         v
  dataset/aspect_labeled_dataset.csv (5.021 baris, kolom: content, content_id, stemming, aspek_label, app_source)
```

Validasi fungsi rule-based terhadap 1.500 anotasi manual menghasilkan akurasi ~78% (F1-macro ~60%) — cukup baik sebagai *silver-label* mengingat ulasan pendek, banyak typo/slang, dan seringkali ambigu bahkan bagi anotator manusia.

---

## 3 Skema Pelatihan

| Skema | Algoritma | Ekstraksi Fitur | Split | Catatan |
|-------|-----------|-----------------|-------|---------|
| **1** | SVM + GridSearchCV | TF-IDF (unigram-trigram) | **80/20** | Baseline |
| **2** | **Conv1D** (Deep Learning) | Word Embedding (100-dim) | 80/20 | Algoritma + fitur berbeda dari Skema 1 |
| **3** | SVM + GridSearchCV | TF-IDF (unigram-trigram) | **70/30** | Split berbeda dari Skema 1 |

Skema 2 menggunakan Conv1D + GlobalMaxPooling1D, bukan BiLSTM — BiLSTM sempat dicoba namun mentok di 79–83,5% test accuracy karena ukuran dataset training (~5.000 baris) terlalu kecil untuk BiLSTM belajar representasi optimal dari nol; Conv1D terbukti lebih stabil pada skala data ini.

Hasil akurasi aktual (train/test) untuk ketiga skema dihasilkan di sel "PERBANDINGAN 3 SKEMA PELATIHAN" pada notebook — jalankan notebook untuk melihat nilai terkini.

---

## Inference

Notebook menyediakan fungsi `predict_aspect_s1/s2/s3(text)` yang mengeluarkan salah satu dari 4 kelas kategorikal: `Errors`, `Satisfaction`, `Learnability`, `Efficiency`. Contoh (lihat sel "Inference Gabungan" di notebook):

```python
predict_aspect_s1("Aplikasinya sering error dan tidak bisa dibuka sama sekali")  # -> 'Errors'
predict_aspect_s2("Loading lama banget, aplikasinya berat padahal sinyal kuat")  # -> 'Efficiency'
predict_aspect_s3("Gimana caranya masukin kode referral ya? Bingung banget")     # -> 'Learnability'
```

---

## Cara Menjalankan

```bash
# 1. Install dependensi
pip install -r requirements.txt

# 2. Scraping (opsional -- dataset sudah tersedia di dataset/)
python scraping.py

# 3. Buka notebook (bisa lokal atau Google Colab)
jupyter notebook sentiment_analysis_training.ipynb
```

Notebook didesain CPU-friendly; GPU (mis. T4 di Colab) mempercepat translasi & training Conv1D tapi bukan keharusan.

---

## Requirements

```
google-play-scraper>=1.2.7
pandas>=1.5.3
numpy>=1.23.5
PySastrawi>=1.0.1
scikit-learn>=1.2.2
joblib>=1.2.0
tensorflow>=2.12.0
transformers>=4.30.0
sentencepiece>=0.1.99
sacremoses>=0.0.53
torch>=2.0.0
matplotlib>=3.7.1
seaborn>=0.12.2
notebook>=6.5.4
ipykernel>=6.22.0
```
