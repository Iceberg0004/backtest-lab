"""Very small parser: pulls targets out of a sentence like
'BTC 1h trend strategy, profit factor 1.5+, drawdown under 20%, winrate 40%'."""
import re

SYM = [(r"\b(btc|bitcoin)\b", "binance", "BTCUSDT"), (r"\b(eth|ethereum)\b", "binance", "ETHUSDT"),
       (r"\b(sol|solana)\b", "binance", "SOLUSDT"), (r"\bbnb\b", "binance", "BNBUSDT"), (r"\bxrp\b", "binance", "XRPUSDT"),
       (r"\b(doge)\b", "binance", "DOGEUSDT"),
       (r"\b(xau|xauusd|gold|emas)\b", "dukascopy", "XAUUSD")]
FX = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]


def _num(s):
    return float(s.replace(",", "."))


def parse(text: str) -> dict:
    t = text.lower()
    out = {}
    for pat, src, sym in SYM:
        if re.search(pat, t):
            out["source"], out["symbol"] = src, sym
            break
    for fx in FX:
        if fx.lower() in t.replace("/", "").replace(" ", ""):
            out["source"], out["symbol"] = "dukascopy", fx
    m = re.search(r"\b(\d+)\s*(m|min|menit|h|jam|hour|d|hari|day|w)\b", t)
    if m:
        n, u = m.group(1), m.group(2)
        u = {"min": "m", "menit": "m", "jam": "h", "hour": "h", "hari": "d", "day": "d"}.get(u, u)
        out["tf"] = f"{n}{u}"
    m = re.search(r"(profit\s*factor|\bpf)\D{0,12}?(\d+(?:[.,]\d+)?)", t)
    if m:
        out["min_pf"] = _num(m.group(2))
    m = re.search(r"(drawdown|\bdd)\D{0,15}?(\d+(?:[.,]\d+)?)\s*%?", t)
    if m:
        out["max_dd"] = _num(m.group(2))
    m = re.search(r"(win\s*rate|winrate|wr)\D{0,12}?(\d+(?:[.,]\d+)?)\s*%?", t)
    if m:
        out["min_wr"] = _num(m.group(2))
    m = re.search(r"(\d+)\s*(trade|trades|transaksi)", t)
    if m:
        out["min_trades"] = int(m.group(1))
    if re.search(r"long\s*only|hanya\s*long|cuma\s*long|only\s*long", t):
        out["direction"] = "long"
    elif re.search(r"short\s*only|hanya\s*short|cuma\s*short|only\s*short", t):
        out["direction"] = "short"
    fams = []
    if re.search(r"trend|tren|\bema\b|\bsma\b|moving", t): fams.append("ma_cross")
    if re.search(r"mean\s*rev|rsi|reversal|jenuh", t): fams.append("rsi_mr")
    if re.search(r"breakout|donchian|tembus", t): fams.append("donchian")
    if re.search(r"squeeze|bollinger|kompresi|volatil", t): fams.append("bb_squeeze")
    if re.search(r"macd|momentum", t): fams.append("macd")
    if fams:
        out["families"] = fams
    return out
