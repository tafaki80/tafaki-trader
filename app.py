"""
TAFAKI TRADER — Web App Backend
Flask + Yahoo Finance + Technical Analysis
"""

import os, math, warnings
from datetime import datetime, timezone, timedelta
from flask import Flask, jsonify, render_template_string
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import ta

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# ── Indikator ────────────────────────────────────────────
def add_indicators(df):
    df = df.copy()
    df["ema9"]     = df["close"].ewm(span=9,  adjust=False).mean()
    df["ema26"]    = df["close"].ewm(span=26, adjust=False).mean()
    df["ma20"]     = df["close"].rolling(20).mean()
    df["rsi"]      = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    hl = (df["high"] - df["low"]).replace(0, 1e-10)
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl
    df["cmf"]  = (mfm * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()
    df["atr"]  = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    return df

def analyze(ticker, interval="5m"):
    try:
        df = yf.download(ticker, period="5d", interval=interval, progress=False, auto_adjust=True)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df = df.dropna()
        if len(df) < 30: return None
        df = add_indicators(df)

        last = df.iloc[-1]; prev = df.iloc[-2]
        close   = float(last["close"])
        open_   = float(last["open"])
        ema9    = float(last["ema9"])
        ema26   = float(last["ema26"])
        rsi     = float(last["rsi"])   if not math.isnan(last["rsi"])   else None
        cmf     = float(last["cmf"])   if not math.isnan(last["cmf"])   else 0.0
        vol     = float(last["volume"])
        vol_ma  = float(last["vol_ma20"]) if not math.isnan(last["vol_ma20"]) else None
        atr     = float(last["atr"])   if not math.isnan(last["atr"])   else close * 0.02
        chg     = (close - float(prev["close"])) / float(prev["close"]) * 100

        prev_ema9  = float(prev["ema9"])
        prev_ema26 = float(prev["ema26"])

        ema_cross_up   = prev_ema9 <= prev_ema26 and ema9 > ema26
        ema_cross_down = prev_ema9 >= prev_ema26 and ema9 < ema26
        ema_bull       = ema9 > ema26
        candle_green   = close >= open_
        vol_above_ma   = vol_ma and vol > vol_ma
        vol_ratio      = vol / vol_ma if vol_ma else 0

        cond_ema = ema_bull
        cond_vol = candle_green and vol_above_ma
        cond_rsi = rsi and rsi > 50
        cond_cmf = cmf > 0
        conf     = sum([cond_ema, cond_vol, cond_rsi, cond_cmf])

        if ema_cross_up and cond_vol and cond_rsi and cond_cmf:
            signal = "CROSS BUY"
        elif conf == 4:
            signal = "STRONG BUY"
        elif conf == 3:
            signal = "WEAK BUY"
        elif ema_cross_down:
            signal = "CROSS SELL"
        elif not ema_bull and not cond_rsi and not cond_cmf:
            signal = "SELL"
        else:
            signal = "WAIT"

        # Entry plan
        sup5 = float(df["low"].tail(5).min())
        res5 = float(df["high"].tail(5).max())
        sup  = float(df["low"].tail(20).min())
        res  = float(df["high"].tail(20).max())

        if "BUY" in signal:
            entry    = close
            target1  = res5 if res5 > close * 1.003 else round(close + atr * 1.5, 0)
            target2  = res  if res  > target1 * 1.003 else round(close + atr * 3.0, 0)
            stoploss = max(sup5 * 0.98, close - atr * 1.5)
            risk     = close - stoploss
            rr1      = round((target1 - close) / risk, 1) if risk > 0 else 0
            rr2      = round((target2 - close) / risk, 1) if risk > 0 else 0
            gain1    = round((target1 - close) / close * 100, 2)
            gain2    = round((target2 - close) / close * 100, 2)
        elif "SELL" in signal:
            entry    = close
            target1  = sup5 if sup5 < close * 0.997 else round(close - atr * 1.5, 0)
            target2  = sup  if sup  < target1 * 0.997 else round(close - atr * 3.0, 0)
            stoploss = min(res5 * 1.02, close + atr * 1.5)
            risk     = stoploss - close
            rr1      = round((close - target1) / risk, 1) if risk > 0 else 0
            rr2      = round((close - target2) / risk, 1) if risk > 0 else 0
            gain1    = round((close - target1) / close * 100, 2)
            gain2    = round((close - target2) / close * 100, 2)
        else:
            entry = target1 = target2 = stoploss = rr1 = rr2 = gain1 = gain2 = 0

        # OHLCV history untuk mini chart (50 candle terakhir)
        hist = []
        for i in range(max(0, len(df)-50), len(df)):
            row = df.iloc[i]
            hist.append({
                "t":  str(df.index[i]),
                "o":  round(float(row["open"]),   0),
                "h":  round(float(row["high"]),   0),
                "l":  round(float(row["low"]),    0),
                "c":  round(float(row["close"]),  0),
                "v":  int(row["volume"]),
                "e9": round(float(row["ema9"]),   2) if not math.isnan(row["ema9"])  else None,
                "e26":round(float(row["ema26"]),  2) if not math.isnan(row["ema26"]) else None,
                "r":  round(float(row["rsi"]),    1) if not math.isnan(row["rsi"])   else None,
            })

        return {
            "ticker":   ticker.replace(".JK",""),
            "close":    round(close, 0),
            "chg":      round(chg, 2),
            "ema9":     round(ema9, 2),
            "ema26":    round(ema26, 2),
            "ema_cross_up":   ema_cross_up,
            "ema_cross_down": ema_cross_down,
            "ema_bull":       ema_bull,
            "rsi":      round(rsi, 1) if rsi else None,
            "cmf":      round(cmf, 3),
            "vol":      int(vol),
            "vol_ma":   int(vol_ma) if vol_ma else 0,
            "vol_ratio":round(vol_ratio, 1),
            "candle_green": candle_green,
            "cond_ema": cond_ema,
            "cond_vol": cond_vol,
            "cond_rsi": cond_rsi,
            "cond_cmf": cond_cmf,
            "conf":     conf,
            "signal":   signal,
            "entry":    round(entry, 0),
            "target1":  round(target1, 0),
            "target2":  round(target2, 0),
            "stoploss": round(stoploss, 0),
            "rr1":      rr1,
            "rr2":      rr2,
            "gain1":    gain1,
            "gain2":    gain2,
            "support":  round(sup, 0),
            "resistance": round(res, 0),
            "atr":      round(atr, 0),
            "history":  hist,
            "updated":  wib_now().strftime("%H:%M:%S"),
        }
    except Exception as e:
        return {"error": str(e)}

# ── API Routes ───────────────────────────────────────────
@app.route("/api/analyze")
def api_analyze():
    from flask import request
    tickers  = request.args.get("tickers", "BNBR,BBCA,TLKM").split(",")
    interval = request.args.get("interval", "5m")
    results  = []
    for t in tickers[:5]:  # max 5 saham
        ticker = t.strip().upper()
        if not ticker.endswith(".JK"): ticker += ".JK"
        data = analyze(ticker, interval)
        if data: results.append(data)
    return jsonify({"data": results, "time": wib_now().strftime("%H:%M:%S")})

@app.route("/api/sesi")
def api_sesi():
    now = datetime.now()
    mins = now.hour * 60 + now.minute
    if 9*60 <= mins <= 11*60+30:
        status = "SESI1"; label = "SESI 1 AKTIF"; color = "green"
    elif 13*60+30 <= mins <= 15*60+50:
        status = "SESI2"; label = "SESI 2 AKTIF"; color = "green"
    elif 11*60+30 < mins < 13*60+30:
        status = "BREAK"; label = "ISTIRAHAT SIANG"; color = "yellow"
    else:
        status = "CLOSED"; label = "BURSA TUTUP"; color = "red"
    return jsonify({"status": status, "label": label, "color": color,
                    "time": now.strftime("%H:%M:%S WIB")})

@app.route("/api/screen")
def api_screen():
    from flask import request as req
    tickers = req.args.get("t", ",".join([
        "BBCA","BBRI","BMRI","BBNI","TLKM","GOTO","UNVR",
        "ADRO","PTBA","ANTM","INDF","ICBP","BRIS","ASII"
    ])).split(",")
    out = []
    for t in tickers[:15]:
        tk = t.strip().upper()
        if not tk.endswith(".JK"): tk += ".JK"
        df = yf.download(tk, period="6mo", interval="1d", progress=False, auto_adjust=True)
        if df is None or df.empty: continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df = df.dropna()
        if len(df) < 60: continue
        r = analyze(tk, "1d")
        if r and "error" not in r: out.append(r)
    out.sort(key=lambda x: x.get("conf",0), reverse=True)
    return jsonify({"data": out, "ts": wib_now().strftime("%H:%M:%S WIB")})

@app.route("/api/analyze_eod")
def api_analyze_eod():
    from flask import request as req
    t  = req.args.get("t","BBCA").strip().upper()
    tk = t if t.endswith(".JK") else t+".JK"
    r  = analyze(tk, "1d")
    if not r: return jsonify({"error": "Data tidak tersedia"})
    return jsonify(r)

@app.route("/")
def index():
    return render_template_string(HTML_APP)

# ── HTML Frontend ────────────────────────────────────────
HTML_APP = """<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>Tafaki Trader</title>
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#06090f;--s1:#0b1018;--s2:#0f1822;--s3:#141f2e;
  --b1:#1c2d42;--b2:#243550;
  --c:#00e5ff;--g:#00ff94;--r:#ff3366;--y:#ffcc00;--p:#bf5fff;
  --t1:#e8f4ff;--t2:#7a9bb5;--t3:#3d5a75;
  --mono:'JetBrains Mono',monospace;
  --head:'Barlow Condensed',sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent}
html,body{height:100%;background:var(--bg);color:var(--t1);font-family:var(--mono);overscroll-behavior:none}
nav{position:fixed;bottom:0;left:0;right:0;z-index:100;display:flex;
  background:rgba(11,16,24,0.97);border-top:1px solid var(--b1);
  backdrop-filter:blur(20px);padding-bottom:env(safe-area-inset-bottom)}
.nb{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;
  padding:10px 4px;gap:4px;color:var(--t3);font-family:var(--head);font-size:10px;
  font-weight:700;letter-spacing:.06em;text-transform:uppercase;
  border:none;background:none;cursor:pointer;transition:color .2s;position:relative}
.nb.on{color:var(--c)}
.nb.on::after{content:'';position:absolute;top:0;left:20%;right:20%;
  height:2px;background:var(--c);border-radius:0 0 3px 3px}
.ni{font-size:18px;line-height:1}
header{position:sticky;top:0;z-index:50;background:rgba(6,9,15,0.95);
  border-bottom:1px solid var(--b1);backdrop-filter:blur(20px);
  padding:12px 16px;display:flex;align-items:center;gap:10px}
.logo{font-family:var(--head);font-size:22px;font-weight:800;letter-spacing:-.5px;
  background:linear-gradient(135deg,var(--c),var(--g));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.sesi{margin-left:auto;font-family:var(--head);font-size:11px;font-weight:700;
  padding:4px 12px;border-radius:20px;border:1px solid currentColor;letter-spacing:.06em}
.wt{font-size:11px;color:var(--t3);white-space:nowrap}
.page{display:none;padding:16px 16px 90px;max-width:500px;margin:0 auto}
.page.on{display:block}
.stitle{font-family:var(--head);font-size:13px;font-weight:700;color:var(--t3);
  letter-spacing:.12em;text-transform:uppercase;margin-bottom:12px}
.card{background:var(--s2);border:1px solid var(--b1);border-radius:16px;padding:16px;margin-bottom:12px}
.ticker-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:12px}
.tinp{background:var(--s3);border:1px solid var(--b1);border-radius:10px;padding:10px 6px;
  color:var(--t1);font-family:var(--mono);font-size:14px;font-weight:700;
  text-transform:uppercase;text-align:center;width:100%;transition:border-color .2s}
.tinp:focus{outline:none;border-color:var(--c)}
.ivrow{display:flex;gap:6px;margin-bottom:12px}
.ivbtn{flex:1;padding:8px 4px;border-radius:8px;border:1px solid var(--b1);
  background:var(--s3);color:var(--t3);font-family:var(--head);font-size:13px;
  font-weight:700;cursor:pointer;transition:all .15s}
.ivbtn.on{border-color:var(--c);color:var(--c);background:rgba(0,229,255,.08)}
.rbtn{width:100%;padding:13px;border-radius:12px;
  background:linear-gradient(135deg,rgba(0,229,255,.15),rgba(0,255,148,.1));
  border:1px solid var(--c);color:var(--c);font-family:var(--head);
  font-size:15px;font-weight:800;letter-spacing:.1em;cursor:pointer;transition:all .2s}
.rbtn.stop{border-color:var(--r);color:var(--r);background:rgba(255,51,102,.08)}
.rbar{height:2px;background:var(--b1);border-radius:2px;margin-bottom:14px;overflow:hidden}
.rbarfill{height:100%;width:0%;background:linear-gradient(90deg,var(--c),var(--g));transition:width linear}
.sc{background:var(--s1);border:1px solid var(--b1);border-radius:16px;
  overflow:hidden;margin-bottom:12px;transition:border-color .3s;
  animation:fu .35s ease both}
.sc.buy{border-color:rgba(0,255,148,.35)}.sc.sell{border-color:rgba(255,51,102,.35)}
@keyframes fu{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.sch{padding:14px 16px 10px;display:flex;justify-content:space-between;align-items:flex-start}
.sct{font-family:var(--head);font-size:24px;font-weight:800;color:var(--c);line-height:1}
.xtag{display:inline-block;margin-left:8px;font-size:11px;font-weight:700;
  padding:2px 8px;background:rgba(0,255,148,.15);color:var(--g);
  border-radius:20px;border:1px solid rgba(0,255,148,.3);vertical-align:middle}
.scp{text-align:right}.pv{font-family:var(--head);font-size:22px;font-weight:700}
.pc{font-size:12px;margin-top:2px;font-weight:600}
.G{color:var(--g)}.R{color:var(--r)}.Y{color:var(--y)}.M{color:var(--t3)}
.sr{padding:0 16px 10px;display:flex;align-items:center;gap:10px}
.sb{font-family:var(--head);font-size:13px;font-weight:800;padding:5px 14px;
  border-radius:20px;letter-spacing:.08em}
.sbuy{background:rgba(0,255,148,.12);color:var(--g);border:1px solid rgba(0,255,148,.3)}
.ssell{background:rgba(255,51,102,.12);color:var(--r);border:1px solid rgba(255,51,102,.3)}
.swait{background:rgba(255,204,0,.08);color:var(--y);border:1px solid rgba(255,204,0,.2)}
.cc{font-size:12px;color:var(--t3)}
.kr{display:flex;gap:6px;padding:0 16px 10px}
.kb{flex:1;text-align:center;padding:6px 2px;border-radius:8px;
  font-family:var(--head);font-size:12px;font-weight:700;letter-spacing:.05em}
.kb.ok{background:rgba(0,255,148,.1);color:var(--g);border:1px solid rgba(0,255,148,.25)}
.kb.no{background:var(--s3);color:var(--t3);border:1px solid var(--b1)}
.cw{padding:0 16px 10px}
.rsib{padding:0 16px 8px}
.rsibg{height:4px;background:var(--s3);border-radius:4px}
.rsibf{height:100%;border-radius:4px;transition:width .5s ease}
.rsill{display:flex;justify-content:space-between;font-size:9px;color:var(--t3);margin-top:3px}
.pb{margin:0 12px 14px;padding:12px 14px;background:var(--s3);
  border:1px solid var(--b1);border-radius:12px}
.pt{font-size:10px;color:var(--t3);letter-spacing:.1em;margin-bottom:10px}
.pg{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.pl{font-size:10px;color:var(--t3);margin-bottom:3px}
.pv2{font-family:var(--head);font-size:16px;font-weight:700}
.ps{font-size:10px;color:var(--t3);margin-top:2px}
.sf{padding:8px 16px;border-top:1px solid var(--b1);font-size:10px;color:var(--t3);text-align:right}
.sinp{display:flex;gap:8px;margin-bottom:14px}
.si{flex:1;background:var(--s2);border:1px solid var(--b1);border-radius:10px;
  padding:11px 14px;color:var(--t1);font-family:var(--mono);font-size:14px;
  font-weight:700;text-transform:uppercase;transition:border-color .2s}
.si:focus{outline:none;border-color:var(--c)}
.sbtn{padding:11px 18px;border-radius:10px;background:rgba(0,229,255,.12);
  border:1px solid var(--c);color:var(--c);font-family:var(--head);
  font-size:14px;font-weight:700;cursor:pointer}
.sg{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:14px}
.sb2{background:var(--s2);border:1px solid var(--b1);border-radius:12px;padding:12px 14px}
.sl{font-size:10px;color:var(--t3);letter-spacing:.08em;margin-bottom:4px}
.sv{font-family:var(--head);font-size:20px;font-weight:700}
.ss{font-size:11px;color:var(--t3);margin-top:2px}
.it{background:var(--s2);border:1px solid var(--b1);border-radius:12px;overflow:hidden;margin-bottom:14px}
.ir{display:flex;justify-content:space-between;padding:10px 14px;
  border-bottom:1px solid var(--b1);font-size:13px}
.ir:last-child{border-bottom:none}
.il{color:var(--t3)}.iv{font-weight:600}
.ff{display:flex;gap:6px;margin-bottom:14px;flex-wrap:wrap}
.fb{padding:6px 14px;border-radius:20px;font-family:var(--head);font-size:12px;
  font-weight:700;letter-spacing:.06em;border:1px solid var(--b1);
  background:var(--s2);color:var(--t3);cursor:pointer;transition:all .15s}
.fb.on{border-color:var(--c);color:var(--c);background:rgba(0,229,255,.08)}
.srow{background:var(--s2);border:1px solid var(--b1);border-radius:12px;
  padding:12px 14px;margin-bottom:8px;display:flex;align-items:center;gap:12px;
  animation:fu .3s ease both;cursor:pointer}
.srow.buy{border-left:3px solid var(--g)}.srow.sell{border-left:3px solid var(--r)}
.stk{font-family:var(--head);font-size:18px;font-weight:800;color:var(--c);min-width:50px}
.sm{flex:1}.spr{font-family:var(--head);font-size:16px;font-weight:700}
.ssig{font-size:11px;font-weight:600;margin-top:2px}
.srt{text-align:right}.scn{font-family:var(--head);font-size:20px;font-weight:800}
.skr{display:flex;gap:3px;margin-top:4px;justify-content:flex-end}
.sk{width:8px;height:8px;border-radius:2px}
.sk.ok{background:var(--g)}.sk.no{background:var(--b2)}
.shimmer{background:linear-gradient(90deg,var(--s2) 25%,var(--s3) 50%,var(--s2) 75%);
  background-size:200% 100%;animation:sh 1.5s infinite;border-radius:12px;margin-bottom:8px}
@keyframes sh{0%{background-position:200% 0}100%{background-position:-200% 0}}
.spin{width:32px;height:32px;border-radius:50%;border:3px solid var(--b1);
  border-top-color:var(--c);animation:sp .7s linear infinite;margin:40px auto}
@keyframes sp{to{transform:rotate(360deg)}}
.em{text-align:center;color:var(--t3);padding:40px 20px;font-size:13px}
.csw{width:100%;height:80px;display:block;border-radius:8px}
</style>
</head>
<body>
<header>
  <div class="logo">TAFAKI TRADER</div>
  <span class="sesi" id="sesiPill" style="color:var(--t3)">--</span>
  <span class="wt" id="wibTime">--:--</span>
</header>

<div class="page on" id="pg-scalp">
  <div class="card">
    <div class="stitle">Watchlist & Interval</div>
    <div class="ticker-grid">
      <input class="tinp" id="s1" value="BNBR" maxlength="6">
      <input class="tinp" id="s2" value="BBCA" maxlength="6">
      <input class="tinp" id="s3" value="TLKM" maxlength="6">
    </div>
    <div class="ivrow">
      <button class="ivbtn on" data-iv="1m"  onclick="setIv('1m')">1m</button>
      <button class="ivbtn on" data-iv="5m"  onclick="setIv('5m')">5m</button>
      <button class="ivbtn"    data-iv="15m" onclick="setIv('15m')">15m</button>
      <button class="ivbtn"    data-iv="1h"  onclick="setIv('1h')">1h</button>
    </div>
    <button class="rbtn" id="runBtn" onclick="toggleRun()">▶ MULAI MONITOR</button>
  </div>
  <div class="rbar"><div class="rbarfill" id="rbf"></div></div>
  <div id="scalp-cards">
    <div class="shimmer" style="height:120px"></div>
    <div class="shimmer" style="height:120px"></div>
    <div class="shimmer" style="height:120px"></div>
  </div>
</div>

<div class="page" id="pg-analisa">
  <div class="sinp">
    <input class="si" id="ainp" placeholder="BBCA" maxlength="6" onkeydown="if(event.key==='Enter')doAnalyze()">
    <button class="sbtn" onclick="doAnalyze()">CARI</button>
  </div>
  <div id="aresult"><div class="em">Ketik kode saham lalu tekan CARI</div></div>
</div>

<div class="page" id="pg-screen">
  <div class="ff">
    <button class="fb on" onclick="setFilt('all',this)">Semua</button>
    <button class="fb"    onclick="setFilt('buy',this)">BUY</button>
    <button class="fb"    onclick="setFilt('sell',this)">SELL</button>
    <button class="fb"    onclick="setFilt('wait',this)">WAIT</button>
    <button class="fb" style="color:var(--c);border-color:var(--c)" onclick="doScreen()">↻ Refresh</button>
  </div>
  <div id="screen-cards"><div class="spin"></div></div>
</div>

<nav>
  <button class="nb on" id="nb-scalp"   onclick="goPage('scalp')"><span class="ni">📡</span>SCALP</button>
  <button class="nb"    id="nb-analisa" onclick="goPage('analisa')"><span class="ni">🔍</span>ANALISA</button>
  <button class="nb"    id="nb-screen"  onclick="goPage('screen')"><span class="ni">📊</span>SCREEN</button>
</nav>

<script>
let IV='5m',run=false,RT=null,RSEC=120,sFilt='all',sData=[]

function goPage(p){
  document.querySelectorAll('.page').forEach(x=>x.classList.remove('on'))
  document.querySelectorAll('.nb').forEach(x=>x.classList.remove('on'))
  document.getElementById('pg-'+p).classList.add('on')
  document.getElementById('nb-'+p).classList.add('on')
  if(p==='screen'&&!sData.length) doScreen()
}

async function fetchSesi(){
  try{
    const d=await(await fetch('/api/sesi')).json()
    const el=document.getElementById('sesiPill')
    el.textContent=d.label; el.style.color=d.color; el.style.borderColor=d.color
    document.getElementById('wibTime').textContent=d.time
  }catch{}
}
setInterval(fetchSesi,30000); fetchSesi()

function setIv(iv){IV=iv;document.querySelectorAll('.ivbtn').forEach(b=>b.classList.toggle('on',b.dataset.iv===iv))}

function toggleRun(){run?stopR():startR()}
function startR(){
  run=true
  const btn=document.getElementById('runBtn')
  btn.textContent='■ STOP'; btn.classList.add('stop')
  fetchScalp(); RT=setInterval(fetchScalp,RSEC*1000); animBar()
}
function stopR(){
  run=false; clearInterval(RT)
  const btn=document.getElementById('runBtn')
  btn.textContent='▶ MULAI MONITOR'; btn.classList.remove('stop')
  document.getElementById('rbf').style.width='0%'
}
function animBar(){
  const f=document.getElementById('rbf')
  f.style.transition='none'; f.style.width='100%'
  setTimeout(()=>{f.style.transition=`width ${RSEC}s linear`;f.style.width='0%'},50)
}
async function fetchScalp(){
  const t=[document.getElementById('s1').value,document.getElementById('s2').value,document.getElementById('s3').value].join(',')
  fetchSesi()
  try{
    const d=await(await fetch(`/api/scalp?t=${t}&iv=${IV}`)).json()
    document.getElementById('scalp-cards').innerHTML=d.data.map((r,i)=>mkCard(r,i*80)).join('')||'<div class="em">Tidak ada data</div>'
    animBar()
  }catch{document.getElementById('scalp-cards').innerHTML='<div class="em">Gagal. Cek koneksi.</div>'}
}

function mkCard(r,delay){
  const iB=r.signal.includes('BUY'),iS=r.signal.includes('SELL')
  const cc=iB?'buy':iS?'sell':'',sc=iB?'sbuy':iS?'ssell':'swait'
  const gc=r.chg>=0?'G':'R'
  const xt=r.cross_up?'<span class="xtag">⚡CROSS</span>':r.cross_down?'<span class="xtag" style="color:var(--r);background:rgba(255,51,102,.1);border-color:rgba(255,51,102,.3)">⚡DOWN</span>':''
  const kf=[{l:'EMA',o:r.ce},{l:'VOL',o:r.cv},{l:'RSI',o:r.cr},{l:'CMF',o:r.cc}].map(k=>`<div class="kb ${k.o?'ok':'no'}">${k.l}</div>`).join('')
  const rv=r.rsi||50,rc=rv>70?'var(--r)':rv<30?'var(--g)':'var(--c)'
  const rs=`<div class="rsib"><div style="display:flex;justify-content:space-between;font-size:10px;color:var(--t3);margin-bottom:4px"><span>RSI</span><span style="color:${rc};font-weight:700">${rv} — ${rv>70?'OVERBOUGHT':rv<30?'OVERSOLD':'NETRAL'}</span></div><div class="rsibg"><div class="rsibf" style="width:${rv}%;background:${rc}"></div></div><div class="rsill"><span>0</span><span>30</span><span>50</span><span>70</span><span>100</span></div></div>`
  let pl=''
  if((iB||iS)&&r.entry>0){
    pl=`<div class="pb"><div class="pt">TRADING PLAN</div><div class="pg">
      <div><div class="pl">ENTRY</div><div class="pv2">${f(r.entry)}</div></div>
      <div><div class="pl">STOP LOSS</div><div class="pv2 R">${f(r.sl)}</div></div>
      <div><div class="pl">TARGET 1</div><div class="pv2 G">${f(r.t1)}<div class="ps">+${r.g1}% · R:R 1:${r.rr1}</div></div></div>
      <div><div class="pl">TARGET 2</div><div class="pv2 G">${f(r.t2)}<div class="ps">+${r.g2}% · R:R 1:${r.rr2}</div></div></div>
    </div><div style="margin-top:8px;font-size:10px;color:var(--t3)">ATR: ${f(r.atr)} | S/R: ${f(r.sup)} / ${f(r.res)}</div></div>`
  }else if(r.conf>=2){
    pl=`<div class="pb"><div class="pt">POTENSI (jika 4/4)</div><div class="pg">
      <div><div class="pl">ENTRY</div><div class="pv2 Y">${f(r.entry)}</div></div>
      <div><div class="pl">STOP LOSS</div><div class="pv2">${f(r.sl)}</div></div>
      <div><div class="pl">TARGET 1</div><div class="pv2">${f(r.t1)}</div></div>
      <div><div class="pl">S/R ZONE</div><div class="pv2 M" style="font-size:12px">${f(r.sup)}/${f(r.res)}</div></div>
    </div></div>`
  }
  return `<div class="sc ${cc}" style="animation-delay:${delay}ms">
    <div class="sch"><div><div class="sct">${r.ticker}${xt}</div><div style="font-size:11px;color:var(--t3);margin-top:3px">CMF: ${r.cmf>0?'+':''}${r.cmf} · Vol: ${r.vol_ratio}x</div></div>
    <div class="scp"><div class="pv">${f(r.close)}</div><div class="pc ${gc}">${r.chg>=0?'+':''}${r.chg}%</div></div></div>
    <div class="sr"><span class="sb ${sc}">${r.signal}</span><span class="cc">${r.conf}/4</span></div>
    <div class="kr">${kf}</div>
    <div class="cw">${svgChart(r.hist)}</div>
    ${rs}${pl}
    <div class="sf">Update: ${r.ts}</div>
  </div>`
}

function svgChart(h){
  if(!h||h.length<2) return ''
  const W=340,H=72,P=4,cs=h.map(x=>x.c),n=cs.length
  const mn=Math.min(...cs),mx=Math.max(...cs),rg=mx-mn||1
  const tx=i=>P+i*(W-P*2)/(n-1),ty=v=>H-P-(v-mn)/rg*(H-P*2)
  let s='',e9='',e26=''
  for(let i=1;i<n;i++){
    const col=cs[i]>=cs[i-1]?'#00ff94':'#ff3366'
    s+=`<line x1="${tx(i-1)}" y1="${ty(cs[i-1])}" x2="${tx(i)}" y2="${ty(cs[i])}" stroke="${col}" stroke-width="1.5" opacity=".75"/>`
  }
  const e9p=h.map((x,i)=>x.e9?`${tx(i)},${ty(x.e9)}`:null).filter(Boolean).join(' ')
  const e26p=h.map((x,i)=>x.e26?`${tx(i)},${ty(x.e26)}`:null).filter(Boolean).join(' ')
  const lx=tx(n-1),ly=ty(cs[n-1]),lc=cs[n-1]>=cs[n-2]?'#00ff94':'#ff3366'
  return `<svg class="csw" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
    <rect width="${W}" height="${H}" fill="#0f1822" rx="8"/>
    ${s}
    ${e9p?`<polyline points="${e9p}" fill="none" stroke="#00e5ff" stroke-width="1" opacity=".5"/>`:''}
    ${e26p?`<polyline points="${e26p}" fill="none" stroke="#bf5fff" stroke-width="1" opacity=".5"/>`:''}
    <circle cx="${lx}" cy="${ly}" r="3.5" fill="${lc}"/>
  </svg>`
}

async function doAnalyze(){
  const t=document.getElementById('ainp').value.trim().toUpperCase()
  if(!t) return
  document.getElementById('aresult').innerHTML='<div class="spin"></div>'
  try{
    const r=await(await fetch(`/api/analyze_eod?t=${t}`)).json()
    if(r.error){document.getElementById('aresult').innerHTML=`<div class="em">${r.error}</div>`;return}
    const iB=r.signal.includes('BUY'),iS=r.signal.includes('SELL')
    const sc=iB?'sbuy':iS?'ssell':'swait',gc=r.chg>=0?'G':'R'
    const kf=[{l:'EMA',o:r.ce},{l:'VOL',o:r.cv},{l:'RSI',o:r.cr},{l:'CMF',o:r.cc}].map(k=>`<div class="kb ${k.o?'ok':'no'}">${k.l}</div>`).join('')
    const rv=r.rsi||50,rc=rv>70?'var(--r)':rv<30?'var(--g)':'var(--c)'
    const pl=(iB||iS)&&r.entry?`<div class="pb"><div class="pt">TRADING PLAN (EOD)</div><div class="pg">
      <div><div class="pl">ENTRY</div><div class="pv2">${f(r.entry)}</div></div>
      <div><div class="pl">STOP LOSS</div><div class="pv2 R">${f(r.sl)}</div></div>
      <div><div class="pl">TARGET 1 (+${r.g1}%)</div><div class="pv2 G">${f(r.t1)}<div class="ps">R:R 1:${r.rr1}</div></div></div>
      <div><div class="pl">TARGET 2 (+${r.g2}%)</div><div class="pv2 G">${f(r.t2)}<div class="ps">R:R 1:${r.rr2}</div></div></div>
    </div></div>`:''
    document.getElementById('aresult').innerHTML=`
      <div class="sc ${iB?'buy':iS?'sell':''}">
        <div class="sch"><div><div class="sct">${r.ticker}</div><div style="font-size:11px;color:var(--t3);margin-top:3px">Data EOD · ${r.ts}</div></div>
        <div class="scp"><div class="pv">${f(r.close)}</div><div class="pc ${gc}">${r.chg>=0?'+':''}${r.chg}%</div></div></div>
        <div class="sr"><span class="sb ${sc}">${r.signal}</span><span class="cc">${r.conf}/4</span></div>
        <div class="kr">${kf}</div>
        ${pl}
      </div>
      <div class="sg">
        <div class="sb2"><div class="sl">RSI 14</div><div class="sv" style="color:${rc}">${rv}</div><div class="ss">${rv>70?'Overbought':rv<30?'Oversold':'Netral'}</div></div>
        <div class="sb2"><div class="sl">CMF 20</div><div class="sv" style="color:${r.cmf>0?'var(--g)':'var(--r)'}">${r.cmf>0?'+':''}${r.cmf}</div><div class="ss">${r.cmf>0.1?'Uang Masuk':r.cmf<-0.1?'Uang Keluar':'Netral'}</div></div>
        <div class="sb2"><div class="sl">1 Minggu</div><div class="sv ${r.chg_w>=0?'G':'R'}">${r.chg_w>=0?'+':''}${r.chg_w||0}%</div></div>
        <div class="sb2"><div class="sl">1 Bulan</div><div class="sv ${r.chg_m>=0?'G':'R'}">${r.chg_m>=0?'+':''}${r.chg_m||0}%</div></div>
      </div>
      <div class="it">
        ${[['EMA 9',f(r.e9)],['EMA 26',f(r.e26)],['MA 20',f(r.ma20||r.e9)],['MA 50',f(r.ma50||r.e26)],['Support',f(r.sup),'G'],['Resistance',f(r.res),'R'],['ATR',f(r.atr)],['Vol Ratio',r.vol_ratio+'x']].map(([l,v,c])=>`<div class="ir"><span class="il">${l}</span><span class="iv ${c||''}">${v}</span></div>`).join('')}
      </div>`
  }catch{document.getElementById('aresult').innerHTML='<div class="em">Gagal.</div>'}
}

async function doScreen(){
  document.getElementById('screen-cards').innerHTML='<div class="spin"></div>'
  try{
    const d=await(await fetch('/api/screen')).json()
    sData=d.data||[]; renderScreen()
  }catch{document.getElementById('screen-cards').innerHTML='<div class="em">Gagal.</div>'}
}

function setFilt(flt,el){
  sFilt=flt
  document.querySelectorAll('.fb').forEach(b=>b.classList.remove('on'))
  el.classList.add('on'); renderScreen()
}

function renderScreen(){
  let data=sData
  if(sFilt==='buy') data=data.filter(r=>r.signal.includes('BUY'))
  if(sFilt==='sell') data=data.filter(r=>r.signal.includes('SELL'))
  if(sFilt==='wait') data=data.filter(r=>r.signal==='WAIT')
  if(!data.length){document.getElementById('screen-cards').innerHTML='<div class="em">Tidak ada data</div>';return}
  document.getElementById('screen-cards').innerHTML=data.map((r,i)=>{
    const iB=r.signal.includes('BUY'),iS=r.signal.includes('SELL')
    const sc=iB?'var(--g)':iS?'var(--r)':'var(--y)'
    const gc=r.chg>=0?'var(--g)':'var(--r)'
    const sks=[r.ce,r.cv,r.cr,r.cc].map(o=>`<div class="sk ${o?'ok':'no'}"></div>`).join('')
    return `<div class="srow ${iB?'buy':iS?'sell':''}" style="animation-delay:${i*40}ms" onclick="analyzeFrom('${r.ticker}')">
      <div class="stk">${r.ticker}</div>
      <div class="sm"><div class="spr">${f(r.close)} <span style="font-size:11px;color:${gc}">${r.chg>=0?'+':''}${r.chg}%</span></div>
      <div class="ssig" style="color:${sc}">${r.signal}</div></div>
      <div class="srt"><div class="scn" style="color:${sc}">${r.conf}/4</div><div class="skr">${sks}</div></div>
    </div>`
  }).join('')
}

function analyzeFrom(t){document.getElementById('ainp').value=t;goPage('analisa');doAnalyze()}

function f(n){try{return 'Rp '+Number(n).toLocaleString('id-ID')}catch{return n}}

window.onload=()=>startR()
</script>
</body>
</html>"""



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
