# Backtest Lab (pribadi, non-komersial)

Aplikasi Streamlit yang meniru **alur kerja** backtest seperti "Quant" di Vela: tulis target → atur pengaturan → komputer mencoba ribuan variasi strategi → hanya yang lolos ujian yang ditampilkan → lihat di chart → export ke TradingView.

Ini bukan produk LuxAlgo dan tidak memakai merek atau kode mereka. Dipakai untuk keperluan pribadi, tidak untuk dikomersilkan.

## Datanya dari mana? (otomatis, tidak perlu download manual)

| Pasar | Sumber | Catatan |
|---|---|---|
| Crypto (±80 pair USDT paling ramai) | Binance (gratis, realtime) | Riwayat sejak koin listing (BTC sejak Agustus 2017). |
| Forex mayor & cross, emas, perak | **Dukascopy** (default) | Riwayat sejak 2003, dipakai untuk backtest dan data unseen. |
| Indeks (US30, NAS100, SPX500, GER40, UK100, JPN225), WTI, Brent | Dukascopy | Indeks sejak ±2012, minyak sejak ±2011. |
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
- **Tambahan (v2):** Deflated Sharpe Ratio, PBO (CSCV), tes harga acak (bar permutation) untuk strategi dan untuk seluruh pencarian,
  geser parameter, biaya 2x/3x, walk-forward optimisation sungguhan (anchored/rolling, kurva OOS disambung + WFE),
  multi-aset, portofolio strategi berkorelasi rendah, cek urutan SL/TP pakai candle 1 menit, funding Binance dan swap forex,
  filter ADX, rapor A–E, simpan/bandingkan strategi lewat JSON. Detail di [README_ID.md](README_ID.md).
- **v3:** 28 jenis strategi; exit yang ikut dicari (SL ATR/%, TP1 50%, breakeven, trailing ATR/chandelier, keluar setelah N candle,
  keluar saat sinyal kebalikan); filter konfirmasi (sesi, hari, tren timeframe besar tanpa lookahead, volatilitas, RSI, volume);
  pencarian Acak atau Evolusi; mode Ketat (default) atau mode Vela (longgar, ada peringatan); tab Scan banyak pasar;
  ±80 pair Binance USDT, forex cross, indeks (US30, NAS100, SPX500, GER40, UK100, JPN225), WTI/Brent.

## Advanced (v4, all opt-in)
- **ML tab:** indicator features, triple-barrier labels, purged/embargoed walk-forward CV, OOS test with the same grade, permutation feature importance (scikit-learn HistGradientBoosting or logistic regression, imported lazily). No Pine export.
- **Meta-labeling** filter for any rule strategy; **regime detection** (k-means on causal vol/trend features, fitted in-sample) with per-regime strategy switcher.
- **Position sizing** (fixed, risk % from stop distance, vol target, capped fractional Kelly, compounding on/off) and a **prop-firm challenge simulator** (example rules only; block-bootstrap pass probability).
- **Analytics** (monthly heatmap, hour/weekday, MAE/MFE, streaks, durations, rolling Sharpe/PF, underwater, buy & hold), **live signal panel** for saved strategies (no auto-trading) and a downloadable **HTML report**.

## Tes

- `python tests/e2e_apptest.py binance BTCUSDT carry evo combo scan` (juga `dukascopy EURUSD carry`, `dukascopy XAUUSD evo`, `dukascopy US30 vela combo`): klik semua tombol lewat Streamlit AppTest.
- `python tools/pinecheck/make_case.py 2 && cd tools/pinecheck && node run_all.mjs`: kecocokan Python vs Pine (PineTS), 63 kasus: semua jenis strategi, filter, dan exit.

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
