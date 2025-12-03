import os
import shutil
import random

# =========================
# KONFIGURASI PENTING
# =========================

# 1. Folder sumber (tempat SEMUA 4500 gambar scabies berada)
# Anda harus membuat folder ini dan memindahkan gambar ke sini terlebih dahulu.
SOURCE_RAW_DIR = 'data/raw_scabies' 

# 2. Direktori tujuan (harus sesuai dengan struktur folder training Anda)
TRAIN_DIR = 'data/train/scabies'
VAL_DIR = 'data/val/scabies'

# 3. Rasio Pembagian
# 80% untuk Train, 20% untuk Validation
TRAIN_RATIO = 0.8 

# =========================
# FUNGSI UTAMA
# =========================

def split_scabies_data():
    print("Memulai proses pemisahan data Scabies...")
    
    # 1. Buat folder tujuan jika belum ada
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    
    # 2. Dapatkan semua nama file gambar dari folder mentah
    # Filter hanya untuk file gambar umum (case-insensitive)
    all_files = [f for f in os.listdir(SOURCE_RAW_DIR) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]
    
    if not all_files:
        print(f"❌ Error: Tidak ada file gambar ditemukan di {SOURCE_RAW_DIR}. Mohon periksa path.")
        return

    # 3. Acak urutan file untuk memastikan pembagian yang merata
    random.shuffle(all_files)
    
    # 4. Hitung jumlah file untuk training dan validation
    total_images = len(all_files)
    num_train = int(total_images * TRAIN_RATIO)
    
    # 5. Pisahkan daftar file
    train_files = all_files[:num_train]
    val_files = all_files[num_train:]

    print("-" * 40)
    print(f"Total gambar Scabies ditemukan: {total_images}")
    print(f"  Akan dikirim ke Train (80%): {len(train_files)} files")
    print(f"  Akan dikirim ke Val (20%): {len(val_files)} files")
    print("-" * 40)

    # 6. Salin file ke folder Train
    print("Menyalin ke folder Train...")
    for i, filename in enumerate(train_files):
        source = os.path.join(SOURCE_RAW_DIR, filename)
        destination = os.path.join(TRAIN_DIR, filename)
        shutil.copyfile(source, destination) # Menyalin file
        if (i + 1) % 500 == 0:
            print(f"  {i + 1} file Train selesai disalin.")

    # 7. Salin file ke folder Validation
    print("Menyalin ke folder Validation...")
    for i, filename in enumerate(val_files):
        source = os.path.join(SOURCE_RAW_DIR, filename)
        destination = os.path.join(VAL_DIR, filename)
        shutil.copyfile(source, destination) # Menyalin file
        if (i + 1) % 100 == 0:
            print(f"  {i + 1} file Val selesai disalin.")

    print("-" * 40)
    print(f"✅ Pemisahan data Scabies selesai! Total {total_images} file telah didistribusikan.")

# Panggil fungsi
if __name__ == "__main__":
    split_scabies_data()