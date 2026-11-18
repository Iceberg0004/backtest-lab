# Handoff (v8.9, Nov 2026)

How to continue development on a new machine/assistant.

- Live: https://backtest-lab-iceberg.streamlit.app/ (Streamlit Community Cloud, Python 3.12, auto-deploys from `main`).
- Backup branches: upgrade-v3 ... upgrade-v8.
- Setup: `python3.12 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && streamlit run app.py`
- Layout: `app.py` + `ui_*.py` (UI), `core/` (data, engine, families, exits, indicators, pine, realism, robust, scan, jobs, isolate).
- Background jobs: `core/jobs.py` (session token `?s=` in URL, results in /tmp/vela_jobs, TTL 24h). Heavy tools run in a forked child via `core/isolate.py` so memory is released.
- Memory budget: keep the app under ~1 GB (Community Cloud limit). Indicator cache 32 MB global.
- Tests: `tests/e2e_apptest.py <source> <symbol> [flags]`, `tests/test_reconnect.py`, `tests/test_stop.py`, `tests/test_tools_bg.py`, Pine parity via PineTS.
- Open items: Pine partial TP1/TP2 parity; persistent result storage (free DB) instead of /tmp; transient memory peaks on EURUSD/XAU heavy runs.
- Rules: honest results only, no lookahead, free/legal data sources only.
