"""Very small parser: pulls targets out of a sentence like
'BTC 1h trend strategy, profit factor 1.5+, drawdown under 20%, winrate 40%'."""
import re

SYM = [(r"\b(btc|bitcoin)\b", "binance", "BTCUSDT"), (r"\b(eth|ethereum)\b", "binance", "ETHUSDT"),
       (r"\b(sol|solana)\b", "binance", "SOLUSDT"), (r"\bbnb\b", "binance", "BNBUSDT"), (r"\bxrp\b", "binance", "XRPUSDT"),
       (r"\b(doge)\b", "binance", "DOGEUSDT"),
       (r"\b(xau|xauusd|gold|emas)\b", "dukascopy", "XAUUSD"), (r"\b(xag|xagusd|silver|perak)\b", "dukascopy", "XAGUSD"),
       (r"\b(us30|dow|dow\s*jones)\b", "dukascopy", "US30"), (r"\b(nas100|nasdaq|us100|ustec)\b", "dukascopy", "NAS100"),
       (r"\b(spx500|s&p|sp500|us500)\b", "dukascopy", "SPX500"), (r"\b(ger40|dax|de40)\b", "dukascopy", "GER40"),
       (r"\b(uk100|ftse)\b", "dukascopy", "UK100"), (r"\b(jpn225|nikkei|jp225)\b", "dukascopy", "JPN225"),
       (r"\b(wti|oil|minyak|usoil)\b", "dukascopy", "WTI"), (r"\b(brent|ukoil)\b", "dukascopy", "BRENT")]
FX = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURJPY", "GBPJPY", "EURGBP", "EURAUD", "EURCAD",
      "EURCHF", "EURNZD", "GBPAUD", "GBPCAD", "GBPCHF", "GBPNZD", "AUDJPY", "AUDCAD", "AUDCHF", "AUDNZD", "CADJPY", "CHFJPY",
      "NZDJPY", "NZDCAD", "CADCHF"]


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
    if re.search(r"trend|tren|\bema\b|\bsma\b|moving", t): fams += ["ma_cross", "supertrend", "ma2", "ichimoku", "psar", "aroon"]
    if re.search(r"mean\s*rev|rsi|reversal|jenuh", t): fams += ["rsi_mr", "zscore", "willr", "cci", "stochrsi"]
    if re.search(r"breakout|donchian|tembus", t): fams += ["donchian", "keltner", "atr_ch", "vol_bo", "nr7", "orb", "pivot"]
    if re.search(r"squeeze|bollinger|kompresi|volatil", t): fams += ["bb_squeeze", "sqz"]
    if re.search(r"macd|momentum", t): fams += ["macd", "vortex", "obv", "mfi"]
    if re.search(r"ichimoku", t): fams.append("ichimoku")
    if re.search(r"divergen", t): fams.append("rsi_div")
    if re.search(r"engulf|candle\s*pattern|pola\s*candle|heikin", t): fams += ["engulf", "heikin", "nr7"]
    fams = list(dict.fromkeys(fams))
    if fams:
        out["families"] = fams
    return out
