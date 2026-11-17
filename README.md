# Backtest Lab (pribadi, non-komersial)

Aplikasi Streamlit yang meniru **alur kerja** backtest seperti "Quant" di Vela: tulis target → atur pengaturan → komputer mencoba ribuan variasi strategi → hanya yang lolos ujian yang ditampilkan → lihat di chart → export ke TradingView.

Ini bukan produk LuxAlgo dan tidak memakai merek atau kode mereka. Dipakai untuk keperluan pribadi, tidak untuk dikomersilkan.

## Datanya dari mana? (otomatis, tidak perlu download manual)

| Pasar | Sumber | Catatan |
|---|---|---|
| Crypto (BTCUSDT, ETHUSDT, …) | Binance (gratis, realtime) | Riwayat sejak koin listing (BTC sejak Agustus 2017). |
| Forex & emas (EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD, XAUUSD, …) | **Dukascopy** (default) | Riwayat sejak 2003, dipakai untuk backtest dan data unseen. |
| Candle yang sedang berjalan untuk forex & emas | biquote.io | Harga live dari broker MT5, jadi bisa beda sedikit dengan Dukascopy. Itu normal. |
| Cadangan | Yahoo Finance | 1h hanya ±2 tahun, 1m hanya 7 hari. Emas memakai GC=F (futures). |
| Opsional (pakai API key gratis) | OANDA practice, Twelve Data | Nonaktif sampai kamu mengisi key. Belum diuji karena belum ada key. |

Data pertama kali diunduh lalu disimpan di cache (`/tmp/vela_cache` secara default, atau folder di `VELA_CACHE`). Setelah itu hanya **candle baru** yang diambil, dan chart diperbarui otomatis setiap beberapa detik. Backtest hanya memakai candle yang **sudah tutup**.

Rincian sumber ada di [DATA_SOURCES.md](DATA_SOURCES.md). Panduan pemakaian lebih lengkap ada di [README_ID.md](README_ID.md).

## Cara mencegah "overfitting"

- **Data unseen.** Bagian data terbaru (misalnya 30%) tidak dipakai saat mencari. Strategi harus tetap bagus di bagian ini.
- **Walk-forward.** Data pencarian dibagi beberapa periode, dan strategi harus untung di sebagian besar periode.
- **Stabilitas parameter.** Angka di sekitar parameternya juga harus untung.
- **Jumlah trade minimal**, serta fee, slippage, dan spread selalu dihitung.
- **Monte Carlo.** Urutan trade diacak 1000 kali.
- Urutan hasil hanya berdasarkan data pencarian (in-sample). Status **⚠️ Overfit** berarti hasil di data unseen jauh lebih buruk.

## Menjalankan di komputer sendiri

Perlu Python 3.11 atau lebih baru.

```
pip install -r requirements.txt
streamlit run app.py
```

Lalu buka http://localhost:8501 di browser.

## Deploy gratis di Streamlit Community Cloud

1. Repo ini sudah berisi file yang dibutuhkan: `app.py` di root, `requirements.txt`, dan `packages.txt`.
2. Buka [https://share.streamlit.io](https://share.streamlit.io) dan login dengan akun GitHub yang sama (Iceberg0004).
3. Klik **Create app** / **Deploy an app**.
4. Isi:
   - **Repository:** `Iceberg0004/backtest-lab`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Buka **Advanced settings**:
   - **Python version:** biarkan **3.12** (default). Boleh 3.11 atau 3.13. Jangan 3.10.
   - Secrets tidak wajib. Kalau nanti memakai OANDA atau Twelve Data, tambahkan `OANDA_API_TOKEN` atau `TWELVEDATA_API_KEY` di Secrets. Jangan ditulis di kode.
6. Klik **Deploy**. Tunggu beberapa menit sampai build selesai.

Catatan Streamlit Cloud gratis:

- Aplikasi akan "tidur" kalau tidak dibuka beberapa waktu. Saat dibuka lagi, dia bangun sendiri (bisa 1 menit atau lebih).
- Penyimpanannya tidak permanen, jadi data akan diunduh ulang setelah restart. Ini otomatis; pertama kali saja agak lama.
- Server Streamlit Cloud ada di AS. `api.binance.com` sering diblokir dari sana, jadi aplikasi ini otomatis memakai `data-api.binance.vision` dan host cadangan lain.

## Batasan (penting)

- **Backtest bukan jaminan untung.** Hasil masa lalu tidak menjamin masa depan.
- Belum termasuk swap/bunga menginap forex, funding rate futures, dan likuidasi saat memakai leverage.
- Harga Dukascopy adalah BID. Spread ditambahkan lewat pengaturan spread (pip).
- Data gratis bisa sesekali error atau terlambat. Kalau itu terjadi, klik ulang atau tunggu sebentar.
- Data Dukascopy hanya untuk pemakaian pribadi non-komersial (sesuai ketentuan mereka). Jangan dibagikan ulang.
