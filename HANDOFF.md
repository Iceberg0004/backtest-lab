# Handoff (v9.2, Nov 2026)

How to continue development on a new machine/assistant.

- Live: https://backtest-lab-iceberg.streamlit.app/ (Streamlit Community Cloud, Python 3.12, auto-deploys from `main`).
- Backup branches: upgrade-v3 ... upgrade-v9, upgrade-v9.2. Work branches: feat-memory (memory/speed), feat-ux (UI/mobile).
- Setup: `python3.12 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && streamlit run app.py`
- Layout: `app.py` + `ui_*.py` (UI), `core/` (data, engine, families, exits, indicators, pine, realism, robust, scan, jobs, isolate).
- Background jobs: `core/jobs.py` (session token `?s=` in URL, results in /tmp/vela_jobs, TTL 24h). Heavy tools run in a forked child via `core/isolate.py` so memory is released.
- Archive (v9): `core/archive.py` keeps finished search/scan results in SQLite (~/.backtest_lab/archive.sqlite, 150 MB / 200 items) and rehydrates them into the job folder on demand; Riwayat has download/upload of a `.btlab` file (zip of compressed pickles, loaded with a strict allow-list unpickler).
- Memory budget: keep the app under ~1 GB (Community Cloud limit). Indicator cache 32 MB global.
- Tests: `tests/e2e_apptest.py <source> <symbol> [flags]`, `tests/test_reconnect.py`, `tests/test_stop.py`, `tests/test_tools_bg.py`, `tests/test_archive.py`, `tests/test_ux_fixes.py`, Pine parity `tests/test_pine_parity.py` (PineTS, 117 cases incl. TP1/TP2 legs).
- v9.2 merged: feat-memory@0e8f55e (numba rolling max/min, gc.freeze around the isolate fork, float32 PBO blocks; outputs bit-identical) and feat-ux@e726a6a (retry failed search, correct tab after a job, phone layout/touch targets, red Stop). Multi-asset test now builds one asset cache at a time.
- Open items: feat-memory round 2 (after 0e8f55e) not merged yet; transient memory peaks while an isolate child runs; real forward/paper test. Pine SL+TP in the same candle stays a TradingView-side guess.
- Rules: honest results only, no lookahead, free/legal data sources only.
