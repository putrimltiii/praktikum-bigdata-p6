# =====================================================================
# PRAKTIKUM BIG DATA - PERTEMUAN 6
# Pra-pemrosesan Data dan Rekayasa Fitur Skala Besar
# Bagian 11.1 Latihan Terpadu (Latihan 1 - 4)
# Dataset: data_pelanggan.csv  (jalankan buat_dataset.py lebih dulu)
# =====================================================================
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

# ---------------------------------------------------------------------
# PERSIAPAN: baca data, buang duplikat, gabungkan provinsi
# ---------------------------------------------------------------------
df = pd.read_csv("data_pelanggan.csv", parse_dates=["tgl_daftar"])
print("Ukuran awal:", df.shape)

# Alasan: baris ganda membuat 1 pelanggan terhitung 2 kali, sehingga
# jumlah dan rata-rata menjadi menyesatkan (lihat bagian 4.4).
df = df.drop_duplicates().reset_index(drop=True)
print("Setelah buang duplikat:", df.shape)

# Alasan: tabel utama tidak punya kolom provinsi, jadi di-merge dari tabel
# wilayah. how="left" supaya semua baris pelanggan tetap dipertahankan.
wilayah = pd.DataFrame({
    "kota": ["Jakarta", "Bandung", "Surabaya", "Karawang", "Medan"],
    "provinsi": ["DKI Jakarta", "Jawa Barat", "Jawa Timur",
                 "Jawa Barat", "Sumatera Utara"],
})
df = df.merge(wilayah, on="kota", how="left")

# Alasan: replace(0, np.nan) mencegah pembagian dengan nol.
df["belanja_per_transaksi"] = (
    df["total_belanja"] / df["jumlah_transaksi"].replace(0, np.nan)
)

# =====================================================================
# LATIHAN 1 - MANIPULASI DATA
# =====================================================================
print("\n" + "=" * 70)
print("LATIHAN 1 - MANIPULASI DATA")
print("=" * 70)

# Alasan dropna=False: kota/pendidikan ada yang kosong. Kalau default,
# pelanggan dengan nilai kosong diam-diam hilang dari ringkasan.
# Dengan dropna=False mereka tampil sebagai baris "NaN".
ringkasan = (
    df.groupby(["provinsi", "pendidikan"], dropna=False)["total_belanja"]
      .agg(jumlah_pelanggan="count", rata_total_belanja="mean")
      .round(0)
)
print("\nJumlah pelanggan & rata-rata total_belanja per provinsi dan pendidikan:")
print(ringkasan)

# Alasan: nlargest lebih ringkas & efisien daripada sort_values().head().
print("\n5 pelanggan dengan belanja_per_transaksi tertinggi:")
print(
    df.nlargest(5, "belanja_per_transaksi")
      [["id_pelanggan", "kota", "jumlah_transaksi",
        "total_belanja", "belanja_per_transaksi"]]
      .round(0)
      .to_string(index=False)
)

# =====================================================================
# LATIHAN 2 - NILAI KOSONG (pendapatan, 3 cara)
# =====================================================================
print("\n" + "=" * 70)
print("LATIHAN 2 - NILAI KOSONG PADA PENDAPATAN")
print("=" * 70)

print("Pendapatan kosong:", df["pendapatan"].isna().sum(),
      f"({df['pendapatan'].isna().mean() * 100:.1f}%)")

# Alasan: kota juga ada yang kosong. Isi dulu dengan "Tidak diketahui"
# supaya baris itu punya kelompok pada cara ke-3 (median per kota).
# Kalau tidak, pendapatannya tetap kosong (lihat catatan bagian 5.4).
kota_terisi = df["kota"].fillna("Tidak diketahui")

# Cara 1: rata-rata global
pend_mean = df["pendapatan"].fillna(df["pendapatan"].mean())

# Cara 2: median global
pend_median = df["pendapatan"].fillna(df["pendapatan"].median())

# Cara 3: median per kota
median_kota = df.groupby(kota_terisi)["pendapatan"].transform("median")
pend_median_kota = df["pendapatan"].fillna(median_kota)

perbandingan = pd.DataFrame({
    "asli (ada kosong)": df["pendapatan"],
    "isi rata-rata": pend_mean,
    "isi median": pend_median,
    "isi median per kota": pend_median_kota,
}).agg(["count", "mean", "median", "std", "skew"]).round(2)
print("\nPerbandingan hasil (satuan rupiah):")
print(perbandingan.to_string())

print("\nMedian pendapatan per kota (sebagai dasar cara ke-3):")
print(df.groupby(kota_terisi)["pendapatan"].median().round(0))

# Alasan memilih cara mengisi pendapatan:
# Pendapatan yang kosong ada 581 baris (11,6%), lumayan banyak. Sebelum
# mengisi, saya lihat dulu datanya. Rata-rata aslinya sekitar 10,6 juta,
# sedangkan mediannya 8,8 juta. Selisih sebesar ini biasanya tanda ada
# outlier di sisi atas (maksimumnya sampai 313 juta), jadi datanya miring
# ke kanan.
#
# Kalau diisi pakai rata-rata, 581 baris itu semuanya dapat angka sekitar
# 10,6 juta, padahal kebanyakan pelanggan pendapatannya di bawah itu.
# Efeknya kelihatan di hasil: median naik dari 8,83 juta jadi 9,60 juta.
# Jadi rata-rata ikut "terseret" outlier dan datanya bergeser ke atas.
#
# Isi median tidak kena masalah itu, mediannya tetap 8,83 juta. Median per
# kota hasilnya hampir sama (8,84 juta). Bedanya tipis, mungkin karena
# data ini dibuat acak sehingga median tiap kota tidak jauh berbeda.
#
# Menurut saya yang paling masuk akal median per kota. Selain tahan
# outlier, cara ini mengisi tiap pelanggan sesuai kelompok kotanya. Di data
# nyata, pendapatan Jakarta dan Medan bisa beda jauh, jadi hasilnya lebih
# adil. Untuk data yang sangat besar median biasa pun sudah cukup dan lebih
# simpel. Skewness-nya masih 14,98 karena outlier belum ditangani, itu
# nanti diatasi pakai log di Latihan 3.

# =====================================================================
# LATIHAN 3 - TRANSFORMASI LOG DI DALAM ColumnTransformer
# =====================================================================
print("\n" + "=" * 70)
print("LATIHAN 3 - TRANSFORMASI LOG DALAM ColumnTransformer")
print("=" * 70)

raw = pd.read_csv("data_pelanggan.csv").drop_duplicates()

num_cols = ["umur", "jumlah_transaksi", "total_belanja"]  # numerik biasa
log_cols = ["pendapatan"]                                  # miring -> di-log
cat_cols = ["kota", "pendidikan"]

# Alasan urutan: impute DULU, baru log, baru scaling.
# log1p tidak boleh menerima NaN, dan scaling sebaiknya setelah bentuk
# sebaran diperbaiki oleh log.
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
log_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ("scaler", StandardScaler()),
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

praproses = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("log", log_pipe, log_cols),
    ("cat", cat_pipe, cat_cols),
])

# Alasan: bagi data DULU, fit hanya pada data latih -> cegah data leakage.
from sklearn.model_selection import train_test_split
X_tr, X_te = train_test_split(raw, test_size=0.2, random_state=42)
hasil_tr = praproses.fit_transform(X_tr)
hasil_te = praproses.transform(X_te)
print("Bentuk hasil (latih, uji):", hasil_tr.shape, hasil_te.shape)
print("Nama kolom hasil:")
print(list(praproses.get_feature_names_out()))

# Bukti: skewness pendapatan sebelum vs sesudah log
pend_asli = X_tr["pendapatan"].fillna(X_tr["pendapatan"].median())
pend_log = np.log1p(pend_asli)
print(f"\nSkewness pendapatan: {pend_asli.skew():.2f} (asli) -> "
      f"{pend_log.skew():.2f} (setelah log1p)")

# =====================================================================
# LATIHAN 4 - DIMENSI TINGGI: PCA 0,90 vs 0,99
# =====================================================================
print("\n" + "=" * 70)
print("LATIHAN 4 - PCA n_components 0,90 dan 0,99")
print("=" * 70)

fitur_cols = [c for c in raw.columns if c.startswith("fitur_")]
X = raw[fitur_cols]

# Alasan: buang fitur hampir konstan SEBELUM scaling (bagian 9.1).
# Setelah di-scaling, fitur_31/32 akan "digelembungkan" jadi varians 1.
X_vt = VarianceThreshold(threshold=0.01).fit_transform(X)

# Alasan: PCA wajib diskalakan dulu agar fitur berskala besar tidak mendominasi.
X_std = StandardScaler().fit_transform(X_vt)

baris = []
for target in (0.90, 0.99):
    pca = PCA(n_components=target).fit(X_std)
    baris.append({
        "target varians": f"{target:.0%}",
        "jumlah komponen": pca.n_components_,
        "varians tercapai": f"{pca.explained_variance_ratio_.sum():.2%}",
        "fitur: asli -> sesudah": f"{X_std.shape[1]} -> {pca.n_components_}",
        "penghematan kolom": f"{(1 - pca.n_components_ / X_std.shape[1]):.0%}",
    })
print()
print(pd.DataFrame(baris).to_string(index=False))

# Hasil PCA: untuk mempertahankan 90% varians cukup 8 komponen dari 30
# fitur, tapi untuk 99% dibutuhkan 25 komponen.
#
# Di sinilah trade-off-nya. Dengan 90%, jumlah kolom turun sekitar 73%,
# jadi data lebih ringan, model lebih cepat, dan risiko overfitting lebih
# kecil. Konsekuensinya, sekitar 10% informasi hilang. Dengan 99%,
# informasinya hampir utuh, tapi kolom hanya berkurang 17%, jadi
# penghematannya kecil.
#
# Lonjakan dari 8 ke 25 komponen kemungkinan karena data ini dibuat dari
# 5 faktor tersembunyi ditambah derau (noise), seperti dijelaskan di
# modul. Faktor utamanya tertangkap di komponen-komponen awal, sedangkan
# komponen sisanya kecil dan kebanyakan hanya derau. Jadi menurut saya
# mengejar 99% kurang sepadan, kecuali kalau memang tidak boleh kehilangan
# informasi sama sekali.