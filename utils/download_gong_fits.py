import os
import argparse
import glob
import re
import time
import random
import requests
from datetime import datetime, timedelta
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def parse_args():
    parser = argparse.ArgumentParser(description="Pemuat Turun FITS GONG H-alpha (Kaggle Direct URL + Extra Data)")
    parser.add_argument('--input-dir', type=str, required=True, help="Folder berisi gambar JPEG Kaggle (misal: data/train/images)")
    parser.add_argument('--output-dir', type=str, required=True, help="Folder tempat menyimpan unduhan FITS")
    parser.add_argument('--workers', type=int, default=4, help="Jumlah thread paralel untuk unduhan (dikurangi untuk mencegah IP Banned/429)")
    
    # Argumen untuk data ekstra
    parser.add_argument('--extra-years', type=str, default="", help="Rentang tahun (misal: '2012-2014') atau tahun spesifik ('2013') untuk data tambahan")
    parser.add_argument('--extra-count', type=int, default=0, help="Jumlah data ekstra acak yang ingin didownload (misal: 100)")
    return parser.parse_args()

def robust_get(url, stream=False, timeout=15):
    """Fungsi pembungkus HTTP GET dengan Exponential Backoff untuk melawan 429 Too Many Requests"""
    max_retries = 8
    for attempt in range(max_retries):
        try:
            r = requests.get(url, stream=stream, timeout=timeout)
            if r.status_code == 200:
                return r
            elif r.status_code == 429: # Rate Limiting
                sleep_time = (2 ** attempt) + random.uniform(1, 3)
                time.sleep(sleep_time)
            elif r.status_code == 404:
                return r
        except Exception:
            time.sleep(2 + attempt)
    return None

def process_download(item):
    """Task worker untuk mengunduh FITS secara langsung dari HTTP Archive NSO"""
    basename, output_filepath = item
    
    year = basename[0:4]
    month = basename[4:6]
    day = basename[6:8]
    
    yyyymm = f"{year}{month}"
    yyyymmdd = f"{year}{month}{day}"
    
    url = f"https://gong2.nso.edu/HA/haf/{yyyymm}/{yyyymmdd}/{basename}.fits.fz"
    
    r = robust_get(url, stream=True)
    if r and r.status_code == 200:
        with open(output_filepath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    return False

def get_extra_gong_files(start_year, end_year, count, existing_basenames):
    """Mencari file FITS acak dari direktori arsip GONG NSO pada rentang tahun tertentu"""
    print(f"\n[INFO] Mencari {count} data tambahan dari arsip GONG untuk tahun {start_year}-{end_year}...")
    
    extra_items = []
    attempts = 0
    max_attempts = count * 5 # Batas pencarian agar tidak infinite loop
    
    pbar = tqdm(total=count, desc="Scraping Katalog NSO")
    
    while len(extra_items) < count and attempts < max_attempts:
        attempts += 1
        
        # 1. Pilih tanggal acak di dalam rentang tahun
        start_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year, 12, 31)
        delta_days = (end_date - start_date).days
        random_days = random.randint(0, delta_days)
        target_date = start_date + timedelta(days=random_days)
        
        yyyymm = target_date.strftime("%Y%m")
        yyyymmdd = target_date.strftime("%Y%m%d")
        
        # 2. Cek direktori tanggal tersebut di server NSO
        url = f"https://gong2.nso.edu/HA/haf/{yyyymm}/{yyyymmdd}/"
        r = robust_get(url)
        
        if r and r.status_code == 200:
            # 3. Parse HTML untuk mencari semua file .fits.fz
            # Format contoh href: href="20120101001414Lh.fits.fz"
            matches = re.findall(r'href="(20\d{12}[a-zA-Z]{2})\.fits\.fz"', r.text)
            if matches:
                # Acak susunan file pada hari tersebut
                random.shuffle(matches)
                for basename in matches:
                    if basename not in existing_basenames:
                        extra_items.append(basename)
                        existing_basenames.add(basename)
                        pbar.update(1)
                        if len(extra_items) >= count:
                            break
                            
    pbar.close()
    return extra_items

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. IDENTIFIKASI DATA KAGGLE REFERENSI
    jpeg_files = glob.glob(os.path.join(args.input_dir, "*.jp*g"))
    if not jpeg_files and args.extra_count == 0:
        print(f"[WARN] Tidak ada file JPEG ditemukan dan tidak ada permintaan data ekstra.")
        return
        
    missing_files = []
    existing_basenames = set()
    
    for jpeg_path in jpeg_files:
        filename = os.path.basename(jpeg_path)
        match = re.search(r'^([A-Za-z0-9_]+)\.jp', filename)
        if not match:
            continue
            
        basename = match.group(1)
        existing_basenames.add(basename)
        
        custom_filename = f"{basename}.fits"
        output_filepath = os.path.join(args.output_dir, custom_filename)
        
        if not os.path.exists(output_filepath):
            missing_files.append((basename, output_filepath))
            
    # 2. IDENTIFIKASI DATA EKSTRA (JIKA DIMINTA)
    if args.extra_count > 0 and args.extra_years:
        years = args.extra_years.split('-')
        start_year = int(years[0])
        end_year = int(years[1]) if len(years) > 1 else start_year
        
        extra_basenames = get_extra_gong_files(start_year, end_year, args.extra_count, existing_basenames)
        
        for basename in extra_basenames:
            output_filepath = os.path.join(args.output_dir, f"{basename}.fits")
            if not os.path.exists(output_filepath):
                missing_files.append((basename, output_filepath))
                
    if not missing_files:
        print("[INFO] Seluruh target FITS sudah eksis di penyimpanan. Tidak ada data yang perlu diunduh.")
        return
        
    estimated_size_mb = len(missing_files) * 4.5
    estimated_size_gb = estimated_size_mb / 1024
    
    print("\n" + "="*50)
    print("📊 RINGKASAN METRIK UNDUHAN NSO GONG DIRECT")
    print("="*50)
    print(f"Data Referensi Kaggle (Belum Diunduh) : {len(missing_files) - (args.extra_count if args.extra_count > 0 else 0)} file")
    print(f"Data Ekstra Acak (Tambahan)           : {args.extra_count if args.extra_count > 0 else 0} file")
    print(f"Total citra yang akan diunduh         : {len(missing_files)} file")
    print(f"Estimasi total ukuran                 : {estimated_size_gb:.2f} GB (~{estimated_size_mb:.1f} MB)")
    print("="*50)
    
    confirm = input("Apakah Anda ingin memulai proses unduhan sekarang? [Y/n]: ")
    if confirm.lower() not in ['', 'y', 'yes']:
        print("[INFO] Operasi pengunduhan dibatalkan oleh pengguna.")
        return
        
    print(f"\n[INFO] Memulai unduhan paralel ({args.workers} threads) via NSO HTTP Archive...")
    
    success_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_download, item): item for item in missing_files}
        
        for future in tqdm(as_completed(futures), total=len(missing_files), desc="Mengunduh FITS"):
            if future.result():
                success_count += 1
                
    print(f"\n[INFO] Eksekusi selesai! Berhasil mengunduh {success_count} dari {len(missing_files)} file FITS.")

if __name__ == '__main__':
    main()
