Parity check: Python engine vs exported Pine, run by PineTS (LuxAlgo's open-source Pine runtime, an emulator, not TradingView itself).

    npm install            # once, inside tools/pinecheck
    cd ../.. && python tools/pinecheck/make_case.py && cd tools/pinecheck && node run_all.mjs

Covers all 9 families (incl. the ADX regime filter) on BTCUSDT, EURUSD and XAUUSD 1h, 4000 candles each,
for both the indicator() and the strategy() export.

Partial take-profits (TP1/TP2): the strategy() export opens one entry per portion (TP1 leg, TP2 leg, main leg),
each with its own stop + target, so TradingView's closed-trade list matches the app leg by leg (no qty_percent guessing).
Wrapper: `python tests/test_pine_parity.py`. v9 result: 117 of 117 cases pass (indicator and strategy), 30 cases with partial legs.
