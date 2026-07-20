"""
Scraping ulasan Starbucks dari Google Play Store
Target: >= 10.000 ulasan (gabungan beberapa app & negara)
Output: dataset/starbucks_reviews_raw.csv

App yang di-scrape (urutan prioritas):
1. com.starbucks.id  — Starbucks Indonesia (id)
2. com.starbucks.id  — Starbucks Indonesia versi bahasa Inggris
3. com.starbucks.mobilecard — Starbucks Global/US (en, us)
4. com.starbucks.mobilecard — Starbucks Global, region MY (Malaysia)
5. com.starbucks.mobilecard — Starbucks Global, region SG (Singapore)
6. com.starbucks.mobilecard — Starbucks Global, region PH (Philippines)

Kolom output:
- reviewId      : ID unik review
- userName      : Nama pengguna
- content       : Teks ulasan
- score         : Rating bintang (1-5)
- thumbsUpCount : Jumlah thumbs up
- at            : Tanggal ulasan
- app_source    : Asal scraping (untuk tracking)
"""

import os
import csv
import time
import logging
from datetime import datetime

# ==============================================================================
# KONFIGURASI
# ==============================================================================
TARGET_COUNT    = 11000                # Ambil lebih dari 10.000 untuk margin
OUTPUT_DIR      = "dataset"
OUTPUT_FILE     = os.path.join(OUTPUT_DIR, "starbucks_reviews_raw.csv")
BATCH_SIZE      = 200                  # Jumlah review per request
SLEEP_BETWEEN   = 1.5                  # Detik jeda antar request (hindari rate-limit)

# Daftar sumber scraping (app_id, lang, country, label)
SCRAPE_SOURCES = [
    ("com.starbucks.id",          "id", "id",  "starbucks_id_indonesian"),
    ("com.starbucks.id",          "en", "id",  "starbucks_id_english"),
    ("com.starbucks.mobilecard",  "en", "us",  "starbucks_global_us"),
    ("com.starbucks.mobilecard",  "en", "my",  "starbucks_global_my"),
    ("com.starbucks.mobilecard",  "en", "sg",  "starbucks_global_sg"),
    ("com.starbucks.mobilecard",  "en", "ph",  "starbucks_global_ph"),
    ("com.starbucks.mobilecard",  "en", "au",  "starbucks_global_au"),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("scraping.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)


# ==============================================================================
# FUNGSI UTAMA
# ==============================================================================

def scrape_single_source(
    app_id: str,
    lang: str,
    country: str,
    source_label: str,
    max_count: int,
) -> list[dict]:
    """
    Scrape ulasan dari satu app/lang/country combination.
    Menggunakan continuation_token untuk paginasi hingga max_count terpenuhi.
    """
    from google_play_scraper import reviews, Sort

    collected: list[dict] = []
    continuation_token = None
    batch_num = 0

    logger.info(f"  [{source_label}] Mulai scraping (target: {max_count})...")

    while len(collected) < max_count:
        batch_num += 1
        try:
            result, continuation_token = reviews(
                app_id,
                lang=lang,
                country=country,
                sort=Sort.NEWEST,
                count=BATCH_SIZE,
                continuation_token=continuation_token,
            )
        except Exception as exc:
            logger.warning(f"  [{source_label}] Batch {batch_num} gagal: {exc}. Jeda 5s...")
            time.sleep(5)
            continue

        if not result:
            logger.info(f"  [{source_label}] Tidak ada data lagi.")
            break

        # Tandai sumber scraping
        for r in result:
            r["app_source"] = source_label

        collected.extend(result)
        logger.info(
            f"  [{source_label}] Batch {batch_num}: +{len(result)} | "
            f"Subtotal: {len(collected)}"
        )

        if continuation_token is None:
            logger.info(f"  [{source_label}] Semua ulasan sudah habis.")
            break

        time.sleep(SLEEP_BETWEEN)

    logger.info(f"  [{source_label}] Selesai: {len(collected)} ulasan.")
    return collected


def scrape_reviews() -> list[dict]:
    """
    Scrape ulasan dari semua sumber sampai target terpenuhi.
    Deduplikasi berdasarkan reviewId.
    """
    try:
        from google_play_scraper import reviews, Sort  # noqa: F401
    except ImportError:
        logger.error(
            "Package 'google-play-scraper' belum terinstall. "
            "Jalankan: pip install google-play-scraper"
        )
        raise

    all_reviews: list[dict] = []
    seen_ids: set = set()

    logger.info(f"Mulai scraping multi-source (target: {TARGET_COUNT} ulasan)...")

    for app_id, lang, country, label in SCRAPE_SOURCES:
        remaining = TARGET_COUNT - len(all_reviews)
        if remaining <= 0:
            logger.info("Target tercapai, berhenti.")
            break

        batch = scrape_single_source(app_id, lang, country, label, max_count=remaining + 500)

        # Deduplikasi
        new_added = 0
        for r in batch:
            rid = r.get("reviewId", "")
            if rid and rid not in seen_ids:
                seen_ids.add(rid)
                all_reviews.append(r)
                new_added += 1

        logger.info(f"[{label}] +{new_added} unik | Grand total: {len(all_reviews)}")

    logger.info(f"Scraping selesai. Total unik: {len(all_reviews)} ulasan.")
    return all_reviews


def save_to_csv(review_list: list[dict], filepath: str) -> None:
    """Simpan list review ke file CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    fieldnames = ["reviewId", "userName", "content", "score", "thumbsUpCount", "at", "app_source"]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in review_list:
            writer.writerow({
                "reviewId":      r.get("reviewId", ""),
                "userName":      r.get("userName", ""),
                "content":       r.get("content", ""),
                "score":         r.get("score", ""),
                "thumbsUpCount": r.get("thumbsUpCount", 0),
                "at":            r.get("at", ""),
                "app_source":    r.get("app_source", ""),
            })

    logger.info(f"Data disimpan ke: {filepath}  ({len(review_list)} baris)")


def show_summary(filepath: str) -> None:
    """Tampilkan ringkasan distribusi rating dari file CSV yang sudah disimpan."""
    try:
        import pandas as pd

        df = pd.read_csv(filepath)
        total = len(df)
        logger.info(f"\n{'='*50}")
        logger.info(f"RINGKASAN DATASET: {total} ulasan")
        logger.info(f"{'='*50}")
        logger.info(f"Distribusi Rating (score):\n{df['score'].value_counts().sort_index()}")

        # Hitung label sentimen berdasarkan rating
        df["label"] = df["score"].apply(
            lambda s: "Negatif" if s <= 2 else ("Netral" if s == 3 else "Positif")
        )
        logger.info(f"\nDistribusi Label Sentimen:\n{df['label'].value_counts()}")
        logger.info(f"{'='*50}\n")

        # Cek apakah ulasan kosong
        empty = df["content"].isna().sum() + (df["content"] == "").sum()
        logger.info(f"Ulasan kosong: {empty}")

    except ImportError:
        logger.warning("pandas tidak tersedia, skip summary.")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    start = datetime.now()
    logger.info(f"Waktu mulai: {start.strftime('%Y-%m-%d %H:%M:%S')}")

    reviews_data = scrape_reviews()

    if reviews_data:
        save_to_csv(reviews_data, OUTPUT_FILE)
        show_summary(OUTPUT_FILE)
    else:
        logger.error("Tidak ada data yang berhasil di-scrape!")

    elapsed = datetime.now() - start
    logger.info(f"Total waktu: {elapsed}")
    logger.info("Scraping selesai. Jalankan notebook untuk preprocessing & training.")
