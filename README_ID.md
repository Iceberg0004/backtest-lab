# Backtest Lab (pribadi, non-komersial)

Aplikasi ini meniru **alur kerja** backtest seperti "Quant" di Vela: tulis target → atur pengaturan → komputer mencoba ribuan variasi strategi → hanya yang lolos ujian yang ditampilkan → lihat di chart → export ke TradingView.
Ini bukan produk LuxAlgo dan tidak memakai merek atau kode mereka.

## Datanya dari mana? (otomatis, tidak perlu download manual)
| Pasar | Sumber | Catatan |
|---|---|---|
| Crypto (±80 pair USDT paling ramai di Binance, daftar diperbarui tiap 6 jam) | Binance (gratis, realtime) | Riwayat sejak koin listing (BTC sejak Agustus 2017). Token saham (NVDAB dll) dan stablecoin tidak ikut. |
| Forex mayor & cross (EURUSD, GBPJPY, EURAUD, …), emas, perak | **Dukascopy** (default) | Riwayat sejak 2003, dipakai untuk backtest dan data unseen. |
| Indeks (US30, NAS100, SPX500, GER40, UK100, JPN225) dan minyak (WTI, BRENT) | Dukascopy | Indeks sejak ±2012, minyak sejak ±2011. Spread diisi dalam **poin** (SPX500 0,1 poin, minyak sen). Chart live memakai Dukascopy saja. |
| Candle yang sedang berjalan untuk forex & emas | biquote.io | Harga live dari broker MT5, jadi bisa beda sedikit dengan Dukascopy. Itu normal. |
| Cadangan | Yahoo Finance | 1h hanya ±2 tahun, 1m hanya 7 hari. Emas memakai GC=F (futures). |
| Opsional (pakai API key gratis) | OANDA practice, Twelve Data | Nonaktif sampai kamu mengisi key. Belum diuji karena belum ada key. |

Data pertama kali diunduh lalu disimpan di cache (`/tmp/vela_cache` secara default, atau folder di `VELA_CACHE`). Setelah itu hanya **candle baru** yang diambil, dan chart diperbarui otomatis setiap beberapa detik.
Backtest hanya memakai candle yang **sudah tutup**.
Timeframe kecil dibatasi supaya server gratis tidak kehabisan memori: 1m 120 hari, 5m 3 tahun, 15m 6 tahun (crypto 3m 1 tahun).

## Cara pakai singkat
1. **Sidebar:** pilih sumber, simbol, timeframe, dan tanggal mulai.
2. **Tab "Cari Strategi":**
   - Tulis target, misalnya "BTC 1h trend, profit factor 1.5+, drawdown di bawah 20%", lalu klik **Terapkan**. Atau atur sendiri.
   - Atur persen data unseen, walk-forward, biaya (fee/slippage untuk crypto, **spread dalam pip** untuk forex), dan jenis strategi.
   - Pilih **mode penilaian**:
     - **Ketat (disarankan):** harus lolos data unseen, walk-forward, stabilitas parameter, dan tidak overfit.
     - **Mode Vela (longgar):** cukup lolos target di data lama dan data unseen. Hasilnya lebih banyak, tapi **lebih mungkin cuma kebetulan**. Rapor (grade) tetap memakai cek versi ketat.
   - Pilih **cara mencari**: *Acak*, atau *Evolusi* (variasi yang bagus di data lama diubah sedikit-sedikit; data unseen tidak dipakai untuk memilih). Semua percobaan tetap dihitung di DSR.
   - Centang **kombinasi filter** kalau mau sinyal masuk dikonfirmasi filter lain (sesi, hari, tren timeframe besar, volatilitas, RSI, volume).
   - Klik **Mulai cari strategi**.
3. **Tab "Scan banyak pasar":** pilih beberapa pasar dan timeframe, lalu klik **Mulai scan**. Pencarian yang sama dijalankan di semuanya (dengan batas waktu), lalu diurutkan dalam satu tabel. Hasil satu pasar bisa dibuka di "Hasil & Detail".
4. **Tab "Hasil & Detail":** klik satu baris. Di situ ada:
   - Chart dengan garis indikator dan label Long/Short/Exit beserta P&L.
   - Performance: equity curve dan P&L harian/mingguan/bulanan.
   - Trades: daftar transaksi.
   - Monte Carlo (bootstrap, acak urutan, atau lewatkan 10% trade).
   - **🧪 Uji ketahanan**, **🌍 Multi-aset**, **🎯 Realisme** (lihat di bawah).
   - Export Pine Script dan simpan strategi (JSON).
   - Di atasnya ada **Rapor ketahanan** (A sampai E, skor 0–100).
5. **Tab "Walk-forward":** optimasi ulang sungguhan dari waktu ke waktu.
6. **Tab "Portofolio":** gabungkan beberapa strategi yang tidak bergerak bersamaan.
7. **Tab "Simpan & Bandingkan":** upload file JSON strategi lama, uji ulang di data terbaru, bandingkan.
8. **Tab "Chart Live":** chart realtime. Kalau strategi sudah dipilih, garis dan sinyalnya ikut tampil, dan kamu bisa lihat apakah strategi sedang dalam posisi.

## Cara mencegah "overfitting" (strategi yang cuma cocok dengan data lama)
- **Data unseen.** Bagian data terbaru (misalnya 30%) tidak dipakai saat mencari. Strategi harus tetap bagus di bagian ini.
- **Walk-forward.** Data pencarian dibagi beberapa periode, dan strategi harus untung di sebagian besar periode.
- **Stabilitas parameter.** Angka di sekitar parameternya (misalnya EMA 18/36 di sekitar EMA 20/40) juga harus untung.
- **Jumlah trade minimal**, serta fee, slippage, dan spread selalu dihitung.
- **Monte Carlo.** Urutan trade diacak 1000 kali untuk melihat drawdown terburuk yang mungkin terjadi.
- Urutan hasil hanya berdasarkan data pencarian (in-sample). Data unseen hanya dipakai sebagai ujian, bukan untuk memilih.
- Status **⚠️ Overfit** berarti hasil di data unseen jauh lebih buruk, jadi strategi itu jangan dipakai.

## Tes tambahan (lebih ketat dari sekadar data unseen)
Semua tes berat hanya jalan kalau tombolnya diklik.
- **Deflated Sharpe Ratio (DSR).** Makin banyak variasi dicoba, makin gampang dapat hasil bagus karena kebetulan. DSR mengoreksi itu (memakai jumlah variasi yang benar-benar berbeda; versi paling ketat juga ditampilkan). Di atas 95% bagus. Otomatis dihitung.
- **Peluang overfit (PBO).** Data pencarian dipotong 12 bagian dan dicoba 924 cara bagi-dua. Dihitung seberapa sering "juara" di satu separuh jadi di bawah rata-rata di separuh lain. Di bawah 20% bagus. Otomatis dihitung.
- **Tes harga acak.** Urutan candle diacak ratusan kali (pola pasarnya hilang). Strategi yang bagus harus jauh lebih jelek di harga palsu. Hasilnya p-value: di bawah 0,05 bagus.
- **Tes pencarian acak.** Seluruh pencarian diulang di harga acak. Kalau juara di harga acak sama hebatnya, berarti pencarian cuma menemukan kebetulan.
- **Geser parameter.** Semua angka pengaturan digeser acak ±10–30%. Strategi yang kuat tetap untung.
- **Biaya 2x dan 3x.** Apakah masih untung kalau biaya lebih mahal.
- **Walk-forward sungguhan.** Di tiap jendela waktu, strategi dicari ulang hanya dari data sebelumnya, lalu dipakai di jendela berikutnya. Hasilnya disambung jadi satu kurva, plus *walk-forward efficiency* (≥ 0,5 sehat).
- **Multi-aset.** Strategi yang sama persis dijalankan di aset lain (misalnya BTC, ETH, SOL, BNB, XRP, atau EURUSD, GBPUSD, USDJPY, AUDUSD, XAUUSD).
- **Rapor ketahanan.** Semua tes di atas digabung jadi satu skor. Hanya tes yang sudah dijalankan yang dihitung.

## Lebih realistis
- **Urutan SL/TP asli.** Kalau satu candle menyentuh stop loss dan take profit sekaligus, aplikasi bisa membuka candle 1 menit untuk tahu mana yang duluan (default tetap menganggap SL duluan).
- **Funding rate** (crypto, seperti perpetual futures) dari riwayat funding Binance, dan **swap** forex/emas dengan angka per tahun yang kamu isi.
- **Filter kondisi pasar (ADX)** ikut dicari: hanya trading saat trending, hanya saat sideways, atau tanpa filter.

## Fitur lanjutan (v4, semua opsional)
- **🤖 ML (tab sendiri):** model belajar dari indikator untuk menebak apakah harga lebih dulu menyentuh target atau stop loss (triple-barrier). Validasi silang berurutan dengan jeda (purged/embargo), lalu diuji di data unseen dan diberi rapor (lolos unseen, DSR, biaya 2x). Ada grafik fitur yang paling berpengaruh. Model ML tidak bisa diekspor ke Pine.
- **🤖 Meta-label (di detail strategi):** model kecil yang memilih sinyal mana dari strategi aturan yang diambil atau dilewati.
- **🌦️ Regime:** pasar dibagi jadi 2-4 "cuaca" (tenang/ramai, naik/turun) dari volatilitas dan tren. Model hanya belajar dari in-sample. Ada warna regime di chart, hasil per regime, dan pilihan strategi berbeda per regime.
- **💰 Ukuran posisi & prop firm:** ukuran tetap, risiko % per trade (dari jarak stop loss), target volatilitas, Kelly pecahan (dibatasi), compounding on/off, batas ukuran. Simulasi challenge prop firm dengan aturan CONTOH (target, rugi harian, rugi total) dan peluang lolos dari Monte Carlo.
- **📐 Analitik:** return bulanan, jam & hari terbaik (WIB), MAE/MFE, kalah beruntun, lama trade, Sharpe & PF bergulir, underwater, perbandingan dengan beli & tahan.
- **📡 Sinyal Live:** sinyal terbaru strategi yang disimpan (candle tutup dan candle berjalan), posisi terbuka, dan sinyal terakhir. Hanya informasi, tidak ada order otomatis.
- **📄 Laporan HTML:** di tab Export, satu file berisi chart, metrik, rapor dan kode Pine.

Hasil tes kami sendiri (jujur): model ML murni di emas dan BTC 1 jam hanya sedikit lebih baik dari menebak (AUC ±0,52-0,55) dan rugi di data unseen setelah biaya. Meta-label kadang menaikkan profit factor tapi mengurangi jumlah trade dan total untung.

## Baru di v5
- **📊 Data derivatif crypto (Binance, opsional di sidebar):** funding rate, open interest (OI), rasio long/short (akun dan top trader), dan order flow (volume beli taker dari candle, jadi riwayatnya penuh). Dipakai sebagai filter pencarian (funding ekstrem, arah order flow, OI naik/turun), 3 jenis strategi baru (fade funding ekstrem, divergensi OI vs harga, order flow/CVD), dan fitur ML. Tiap candle hanya memakai data yang sudah ada **sebelum candle itu tutup** (tanpa melihat masa depan).
  - Sumber: API publik Binance Futures + arsip gratis data.binance.vision. API OI Binance hanya menyimpan ±30 hari, jadi riwayat OI/long-short diambil dari arsip harian (mulai ±Sep 2020). Funding mulai ±Sep 2019.
  - Data likuidasi historis gratis tidak tersedia (arsipnya kosong), jadi tidak dipakai.
  - Pengambilan pertama 2-4 menit, lalu disimpan di cache. Kalau Binance diblokir di jaringan, aplikasi tetap jalan tanpa data ini dan memberi tahu.
  - Strategi yang memakai data derivatif **tidak bisa diekspor ke Pine** (TradingView tidak punya data ini di Pine).
- **🛰️ Matriks scan:** peta nilai rapor pasar × timeframe dan jenis strategi × pasar/timeframe.
- **🔗 Kemiripan strategi:** korelasi return harian (in-sample saja), pengelompokan strategi kembar, dan daftar wakil tiap kelompok. Pemilihan portofolio otomatis sekarang juga hanya memakai korelasi in-sample.
- **💸 Preset biaya broker:** Binance spot / BNB / futures taker / maker, dan akun forex standar / raw-ECN (CONTOH, bukan broker tertentu). Semua angka tetap bisa diubah; spread dan swap disimpan per pasar selama sesi.
- **Rapor dengan DSR ketat (opsional):** versi buku teks yang menganggap semua variasi benar-benar terpisah. Nilai jadi lebih pelit.

Hasil tes kami sendiri (BTC 1 jam, data unseen ±Feb 2024 sampai sekarang, sudah termasuk biaya): data derivatif **tidak otomatis lebih baik**. Rata-rata strategi yang memakai filter derivatif sedikit lebih baik di unseen, tapi strategi terbaik yang lolos tes tetap strategi tanpa data derivatif. Filter OI paling membantu breakout Donchian; ML dengan fitur derivatif rugi lebih kecil dari ML biasa tapi tetap rugi.

## Baru di v6: Grid · Martingale · DCA (tab 🧮)
Tab baru untuk strategi **banyak posisi sekaligus**. Mesinnya terpisah dari backtest biasa: menghitung saldo, margin, eksposur, dan **margin call (stop-out)** di setiap candle (jalur harga dalam candle: open → low → high → close untuk candle hijau, open → high → low → close untuk candle merah; gap mengisi limit order di harga open).
- **Grid:** batas bawah/atas (% dari harga awal), jumlah grid, jarak sama (aritmetika) atau persen sama (geometrik), arah long / short / netral, stop kalau harga keluar range, geser range (recenter), tutup semua di target untung/rugi total.
- **DCA / safety order (gaya 3Commas):** base order, safety order, deviasi, step scale, volume scale, TP dari harga rata-rata, stop loss, cooldown. Sinyal awal: selalu / RSI / tren.
- **Martingale dan anti-martingale:** di atas sinyal strategi mana pun (ukuran ×2 setelah rugi atau setelah untung, batas langkah, dibatasi margin yang tersedia).
- **Pyramiding (gaya Turtle):** tambah posisi tiap N×ATR, stop dari posisi terakhir.
- **Hedging** (long dan short bersamaan) dan **zone recovery** (balik arah dengan ukuran lebih besar sampai total untung).
- **Pairs trading / stat-arb** (tes kointegrasi ADF dan rasio lindung nilai hanya dari in-sample), **rebalancing 2 aset**, dan **DCA rutin vs beli sekali**.
- **Laporan risiko ekor:** eksposur maks, jumlah posisi maks, drawdown terburuk, lama di bawah puncak, **Monte Carlo peluang margin call** (setahun, dari potongan acak riwayat asli), dan peringatan merah.
- **Rapor lebih pelit untuk mode berisiko:** pernah margin call → maks 20; peluang margin call > 1% → maks 60; drawdown > 50% → maks 45; grid/DCA/martingale/hedge/zone **tidak bisa dapat A** kecuali lolos unseen, biaya 2x, setelan tetangga, dan peluang margin call ≤ 1%.
- **Cari setelan** (acak atau evolusi, hanya in-sample, lalu dinilai di unseen) dan **uji di pasar lain**.
- **Pine:** DCA dan martingale diekspor sebagai strategy() (DCA dicek dengan PineTS: semua deal sama di data BTC uji); grid sebagai indikator level + alert; hedging, zone recovery, dan pyramiding belum bisa diekspor.

**Jenis strategi sinyal baru (12):** divergensi MACD, pin bar, doji, pullback Fibonacci, VWAP harian (mean reversion), Fair Value Gap, Break of Structure / Order Block, liquidity sweep, musiman jam, musiman awal/akhir bulan, buy & hold (pembanding), dan carry (crypto, dari funding). Semua kecuali carry punya kembaran Pine yang dicek PineTS.
Sudah ada sebelumnya (tidak dibuat ulang): breakout sesi / opening range, Donchian (Turtle), divergensi RSI, engulfing, inside bar. Swap/carry forex diatur lewat pengaturan swap.
**Tidak dibuat:** pola harmonik (sulit dibuat konsisten tanpa "melihat masa depan"), kalender berita (tidak ada data historis gratis yang andal).

Hasil tes kami sendiri (biaya realistis, leverage 10x, data unseen): grid, martingale, hedging, dan zone recovery **sering kena margin call** atau rugi di unseen. DCA terlihat paling "rapi" (win rate hampir 100%) tapi drawdown-nya besar dan peluang margin call di Monte Carlo tidak nol. Tidak ada satu pun yang mengalahkan strategi terbaik v5 setelah risiko dihitung. Win rate tinggi ≠ aman.

## Baru di v7: SL / TP lengkap + 69 jenis strategi

**Stop loss (14 jenis + "tanpa SL")**: ATR, % harga, pips/poin, swing low/high + buffer, pivot/struktur + buffer, channel Donchian,
band Bollinger, band Keltner, low/high candle sinyal + buffer, chandelier, Parabolic SAR, Supertrend, standar deviasi, maks rugi (uang),
dan "tanpa SL" (ada peringatan). Kalau level SL tidak masuk akal (mis. di sisi yang salah), dipakai SL ATR biasa.

**Take profit (10 jenis)**: sama satuan dengan SL, kelipatan R (1–5R), ATR, %, pips, swing/S&R terdekat, pivot terakhir,
Fibonacci extension 1.272 / 1.618 / 2.0, band seberang, tanpa TP (hanya trailing / sinyal kebalikan).

**Ambil untung bertahap**: TP1 dan TP2 di X R dengan % yang ditutup, sisanya ke TP akhir. Breakeven di X R + offset.

**Trailing (8 jenis)**: ATR, chandelier, %, Parabolic SAR, Supertrend, garis MA, step (naik per X R), Donchian. Bisa "aktif setelah X R".

**Aturan trade**: keluar setelah N candle, time stop (keluar kalau belum untung setelah N candle), tutup akhir hari (00:00 / 21:00 /
16:00 UTC), tutup akhir pekan (Jumat 20:00 UTC), jeda N candle setelah rugi, maks trade per hari, batas rugi / target untung harian,
tidak masuk ulang ke arah yang sama sebelum ada sinyal kebalikan. (Pyramid / tambah posisi ada di tab 📐 Posisi sejak v6.)

Semua jenis ini ikut dicari otomatis (Acak dan Evolusi), ikut tes robust, dan ikut diekspor ke Pine. Di tab cari ada
"🎯 SL / TP lengkap" untuk membatasi jenis yang dicoba. Daftar trade sekarang menampilkan SL awal, SL akhir, TP, TP1/TP2 yang kena,
dan alasan keluar. Di tab Robust ada **⚖️ Bandingkan exit**: sinyal masuk sama, 60–200 kombinasi exit, diurutkan dari data unseen
(baris teratas cenderung terlalu bagus karena dipilih dari unseen; lebih aman pilih yang untung di in-sample dan unseen).

**Pine**: versi indicator sama persis dengan aplikasi di semua kasus tes (112) (PineTS), termasuk semua jenis SL/TP/trailing baru.
Versi strategy sama di ±95% kasus; yang beda semuanya memakai TP1/TP2 (mesin order TradingView kadang menggabung / melewati satu leg).

**Strategi yang ada (69 jenis), per kategori**
- Tren / MA: MA cross (SMA/EMA, termasuk golden/death cross), 2 MA + filter, Hull/TEMA/KAMA, ZLEMA/ALMA/T3/McGinley, MA ribbon (triple MA),
  Supertrend, Parabolic SAR, Ichimoku, Aroon, Vortex, Heikin Ashi, SSL channel, UT Bot, Range filter, Gaussian channel, LRC.
- Momentum / osilator: RSI, RSI-2 (Connors), Stoch RSI, Williams %R, CCI, MFI, MACD, TRIX, KST, ROC, Awesome Oscillator, Elder Ray
  (bull / bear power), Fisher transform, QQE, WaveTrend.
- Breakout / volatilitas: Donchian (+ garis tengah), Bollinger squeeze, Squeeze (LazyBear), Keltner, ATR channel, volatility breakout,
  NR4/NR7/inside bar, Chandelier entry, breakout level (high/low kemarin, open mingguan, range Asia, high/low minggu lalu), ORB,
  opening drive, Larry Williams volatility breakout.
- Mean reversion: RSI, Williams %R, MFI, Z-score, VWAP reversion, linear regression channel, Bollinger %B, gap fill.
- Candle / pola: engulfing, pin bar (hammer / shooting star), doji, morning/evening star, three soldiers/crows, harami, tweezer, Fibonacci pullback, pivot S&R.
- SMC: fair value gap, break of structure / order block, liquidity sweep.
- Musiman: jam & hari tertentu, pergantian bulan. Patokan: beli & tahan.
- Volume: OBV, breakout + lonjakan volume, divergensi OBV / AD / CMF, POC (sederhana).
- Divergensi: RSI, MACD.
- Kekuatan relatif: koin vs BTC (rasio harga, EMA cross / momentum rasio; hanya koin Binance selain BTC, Pine memakai request.security).
- Lainnya: grid, martingale, DCA, pyramid (tab 🧮 / 📐), pairs / spread, ML, data derivatif (funding ekstrem, carry funding, open interest, order flow CVD).

**Belum ada (jujur)**: volume profile lengkap (hanya POC sederhana dari candle), pola harmonik.

## Jenis strategi, exit, dan filter
- **43 jenis strategi** (fade funding, divergensi OI, order flow, dan carry khusus data derivatif crypto): MA cross, RSI, Donchian, Bollinger squeeze, MACD, Supertrend, Keltner, Stoch RSI pullback, Z-score, Hull/TEMA/KAMA cross, Ichimoku, Parabolic SAR, Heikin-Ashi, CCI, Williams %R, MFI, OBV, breakout + volume, ATR channel, pivot support/resistance, inside bar / NR7, engulfing, opening range breakout (hanya timeframe ≤ 1 jam), Aroon, Vortex, Squeeze Momentum, linear regression channel, divergence RSI, fade funding ekstrem, divergensi OI, order flow/CVD.
- **Exit (dicoba otomatis):** stop loss ATR atau %, take profit, ambil untung 50% di TP1, stop pindah ke harga masuk (breakeven) setelah X R, trailing ATR atau chandelier, keluar setelah N candle, keluar hanya saat sinyal kebalikan.
- **Filter konfirmasi (kalau kombinasi dicentang):** sesi Asia / London / New York / London+NY (jam UTC), hari (tanpa Senin / tanpa Jumat / hari kerja), tren timeframe besar (4 jam / harian, hanya candle besar yang sudah tutup, jadi tanpa "melihat masa depan"), volatilitas tinggi / rendah, RSI di sisi yang sama, volume di atas rata-rata, dan (crypto, kalau data derivatif aktif) funding tidak ekstrem, order flow searah, OI naik/turun.

## Export ke TradingView
Ada dua file `.pine` (Pine Script v5):
- **strategy**: menampilkan hasil backtest di TradingView.
- **indicator**: label Long/Short/TP1/SL/TP/Exit, bisa dipasang alert. Posisinya dihitung sendiri, **sama persis** dengan aplikasi.

Keduanya punya **tabel statistik** di pojok chart (hasil uji aplikasi; versi strategy juga menampilkan hasil TradingView sendiri) dan **alert** untuk sinyal Long/Short/Exit, kena stop loss, kena take profit, dan posisi baru. Semua jenis strategi (kecuali yang memakai data derivatif), semua filter, dan semua jenis exit sudah dicek dengan PineTS (emulator Pine open-source) di 85 kasus: versi indicator sama persis dengan aplikasi. Versi strategy juga sama, kecuali trade yang SL dan TP-nya tersentuh di candle yang sama (TradingView menebak urutannya sendiri).

Caranya: buka TradingView → Pine Editor → tempel kodenya → Add to chart, di simbol dan timeframe yang sama.
Hasil di TradingView bisa sedikit berbeda karena:
- data broker-nya beda (misalnya OANDA vs Dukascopy);
- urutan kena SL/TP di dalam satu candle (aplikasi ini selalu menganggap SL kena lebih dulu);
- jam sesi, hari, dan tren timeframe besar dihitung dari waktu **UTC** candle, jadi candle harian broker yang tutup di jam lain bisa memberi hasil sedikit beda;
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
- Data derivatif hanya untuk crypto Binance, tanpa data likuidasi. Coverage OI/long-short mulai ±2020.
- Funding dan swap adalah perkiraan (swap pakai angka yang kamu isi). Likuidasi/margin call dihitung hanya di tab Grid · Martingale · DCA, dengan aturan sederhana (stop-out saat ekuitas ≤ X% margin), bukan aturan broker tertentu.
- Skor rapor dan semua tes adalah hasil uji data lama, bukan saran trading.
- Aturan prop firm di aplikasi hanya contoh, bukan aturan resmi perusahaan mana pun.
- Harga Dukascopy adalah BID. Spread ditambahkan lewat pengaturan spread (pip).
- Data gratis bisa sesekali error atau terlambat. Kalau itu terjadi, klik ulang atau tunggu sebentar.
- Data Dukascopy hanya untuk pemakaian pribadi non-komersial (sesuai ketentuan mereka). Jangan dibagikan ulang.
- Monte Carlo memakai potongan riwayat asli yang diacak. Kejadian yang belum pernah terjadi (misalnya crash lebih parah) tidak tercakup, jadi peluang margin call sebenarnya bisa lebih tinggi.

## v8 / v9: pencarian jalan di server + tampilan baru

- **Pencarian & scan jalan di server.** Minimize / kunci HP / muat ulang halaman tidak mengulang pencarian: alamat halaman membawa kode sesi (`?s=...`), jadi halaman tersambung lagi ke pekerjaan yang sedang jalan atau membuka hasil yang sudah selesai. Hasil disimpan 24 jam. Kalau server gratis dimulai ulang, hasil ikut hilang.
- **Lebih simpel:** pilih pasar & timeframe di halaman utama, satu tombol besar, pengaturan lanjutan dilipat, 6 tab, kartu ringkasan hasil, dan tab Riwayat.
- **Komponen open-source:** streamlit-extras (Apache-2.0), streamlit-shadcn-ui (MIT), streamlit-lightweight-charts (MIT, chart TradingView). Kalau komponen gagal dimuat, aplikasi otomatis memakai tampilan Streamlit biasa.
