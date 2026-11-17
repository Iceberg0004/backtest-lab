# Backtest Lab (pribadi, non-komersial)

Aplikasi ini meniru **alur kerja** backtest seperti "Quant" di Vela: tulis target → atur pengaturan → komputer mencoba ribuan variasi strategi → hanya yang lolos ujian yang ditampilkan → lihat di chart → export ke TradingView.
Ini bukan produk LuxAlgo dan tidak memakai merek atau kode mereka.

## Datanya dari mana? (otomatis, tidak perlu download manual)
| Pasar | Sumber | Catatan |
|---|---|---|
| Crypto (BTCUSDT, ETHUSDT, …) | Binance (gratis, realtime) | Riwayat sejak koin listing (BTC sejak Agustus 2017). |
| Forex & emas (EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD, XAUUSD, …) | **Dukascopy** (default) | Riwayat sejak 2003, dipakai untuk backtest dan data unseen. |
| Candle yang sedang berjalan untuk forex & emas | biquote.io | Harga live dari broker MT5, jadi bisa beda sedikit dengan Dukascopy. Itu normal. |
| Cadangan | Yahoo Finance | 1h hanya ±2 tahun, 1m hanya 7 hari. Emas memakai GC=F (futures). |
| Opsional (pakai API key gratis) | OANDA practice, Twelve Data | Nonaktif sampai kamu mengisi key. Belum diuji karena belum ada key. |

Data pertama kali diunduh lalu disimpan di cache (`/tmp/vela_cache` secara default, atau folder di `VELA_CACHE`). Setelah itu hanya **candle baru** yang diambil, dan chart diperbarui otomatis setiap beberapa detik.
Backtest hanya memakai candle yang **sudah tutup**.

## Cara pakai singkat
1. **Sidebar:** pilih sumber, simbol, timeframe, dan tanggal mulai.
2. **Tab "Cari Strategi":**
   - Tulis target, misalnya "BTC 1h trend, profit factor 1.5+, drawdown di bawah 20%", lalu klik **Terapkan**. Atau atur sendiri.
   - Atur persen data unseen, walk-forward, biaya (fee/slippage untuk crypto, **spread dalam pip** untuk forex), dan jenis strategi.
   - Klik **Mulai cari strategi**.
3. **Tab "Hasil & Detail":** klik satu baris. Di situ ada:
   - Chart dengan garis indikator dan label Long/Short/Exit beserta P&L.
   - Performance: equity curve dan P&L harian/mingguan/bulanan.
   - Trades: daftar transaksi.
   - Monte Carlo.
   - Export Pine Script.
4. **Tab "Chart Live":** chart realtime. Kalau strategi sudah dipilih, garis dan sinyalnya ikut tampil, dan kamu bisa lihat apakah strategi sedang dalam posisi.

## Cara mencegah "overfitting" (strategi yang cuma cocok dengan data lama)
- **Data unseen.** Bagian data terbaru (misalnya 30%) tidak dipakai saat mencari. Strategi harus tetap bagus di bagian ini.
- **Walk-forward.** Data pencarian dibagi beberapa periode, dan strategi harus untung di sebagian besar periode.
- **Stabilitas parameter.** Angka di sekitar parameternya (misalnya EMA 18/36 di sekitar EMA 20/40) juga harus untung.
- **Jumlah trade minimal**, serta fee, slippage, dan spread selalu dihitung.
- **Monte Carlo.** Urutan trade diacak 1000 kali untuk melihat drawdown terburuk yang mungkin terjadi.
- Urutan hasil hanya berdasarkan data pencarian (in-sample). Data unseen hanya dipakai sebagai ujian, bukan untuk memilih.
- Status **⚠️ Overfit** berarti hasil di data unseen jauh lebih buruk, jadi strategi itu jangan dipakai.

## Export ke TradingView
Ada dua file `.pine` (Pine Script v5):
- **strategy**: menampilkan hasil backtest di TradingView.
- **indicator**: hanya label Long/Short/Exit, dan bisa dipasang alert.

Caranya: buka TradingView → Pine Editor → tempel kodenya → Add to chart, di simbol dan timeframe yang sama.
Hasil di TradingView bisa sedikit berbeda karena:
- data broker-nya beda (misalnya OANDA vs Dukascopy);
- urutan kena SL/TP di dalam satu candle (aplikasi ini selalu menganggap SL kena lebih dulu);
- biaya spread di Pine dihitung kira-kira sebagai komisi %.

## Menjalankan di komputer sendiri
Perlu Python 3.10 atau lebih baru.
```
pip install -r requirements.txt
streamlit run app.py
```
Lalu buka http://localhost:8501 di browser.

## Online gratis di Streamlit Community Cloud
Langkah lengkap ada di [README.md](README.md). Ringkasnya:
1. Buka https://share.streamlit.io dan login dengan akun GitHub.
2. Klik **Create app**, pilih repo `Iceberg0004/backtest-lab`, branch `main`, file `app.py`.
3. Di **Advanced settings**, biarkan Python **3.12**. Klik **Deploy**.

Aplikasi gratis akan "tidur" kalau tidak dibuka beberapa waktu, lalu bangun sendiri saat dibuka lagi. Cache data tidak permanen, jadi unduhan pertama setelah restart bisa agak lama.

API key OANDA atau Twelve Data (opsional) diisi di Secrets Streamlit Cloud dengan nama `OANDA_API_TOKEN` / `TWELVEDATA_API_KEY`. Jangan ditulis langsung di kode.

## Batasan (penting)
- **Backtest bukan jaminan untung.** Hasil masa lalu tidak menjamin masa depan.
- Belum termasuk swap/bunga menginap forex, funding rate futures, dan likuidasi saat memakai leverage.
- Harga Dukascopy adalah BID. Spread ditambahkan lewat pengaturan spread (pip).
- Data gratis bisa sesekali error atau terlambat. Kalau itu terjadi, klik ulang atau tunggu sebentar.
- Data Dukascopy hanya untuk pemakaian pribadi non-komersial (sesuai ketentuan mereka). Jangan dibagikan ulang.
