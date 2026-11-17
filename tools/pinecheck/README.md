Parity check: Python engine vs exported Pine, run by PineTS (LuxAlgo's open-source Pine runtime, an emulator, not TradingView itself).

    npm install            # once, inside tools/pinecheck
    cd ../.. && python tools/pinecheck/make_case.py && cd tools/pinecheck && node run_all.mjs

Covers all 9 families (incl. the ADX regime filter) on BTCUSDT, EURUSD and XAUUSD 1h, 4000 candles each,
for both the indicator() and the strategy() export.
