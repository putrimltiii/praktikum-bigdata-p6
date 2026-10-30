import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 5000
df = pd.DataFrame({
    "id_pelanggan": np.arange(1, n + 1),
    "umur": rng.normal(35, 10, n).round().clip(17, 70),
    "pendapatan": rng.lognormal(mean=16, sigma=0.5, size=n).round(),
    "kota": rng.choice(["Jakarta", "Bandung", "Surabaya", "Karawang", "Medan"],
                       n, p=[0.3, 0.2, 0.2, 0.2, 0.1]),
    "pendidikan": rng.choice(["SMA", "D3", "S1", "S2"], n,
                             p=[0.3, 0.15, 0.45, 0.1]),
    "tgl_daftar": pd.to_datetime("2022-01-01")
                  + pd.to_timedelta(rng.integers(0, 1000, n), unit="D"),
    "jumlah_transaksi": rng.poisson(12, n),
})
df["total_belanja"] = (df["jumlah_transaksi"]
                       * rng.normal(150_000, 40_000, n)).round()
laten = rng.normal(size=(n, 5))
bobot = rng.normal(size=(5, 30))
fitur = laten @ bobot + rng.normal(scale=0.7, size=(n, 30))
for i in range(30):
    df[f"fitur_{i + 1:02d}"] = fitur[:, i]
df["fitur_31"] = rng.normal(5, 0.01, n)
df["fitur_32"] = rng.normal(0, 0.01, n)
skor = 0.9 * laten[:, 0] - 0.7 * laten[:, 1] + rng.normal(scale=0.5, size=n)
df["beli_lagi"] = (skor > 0).astype(int)
idx = rng.choice(n, 15, replace=False)
df.loc[idx, "pendapatan"] = df.loc[idx, "pendapatan"] * 15
for kol, persen in [("umur", 0.08), ("pendapatan", 0.12),
                    ("kota", 0.05), ("pendidikan", 0.06)]:
    df.loc[rng.random(n) < persen, kol] = np.nan
df = pd.concat([df, df.sample(50, random_state=1)], ignore_index=True)
df.to_csv("data_pelanggan.csv", index=False)
print("Selesai:", df.shape)
