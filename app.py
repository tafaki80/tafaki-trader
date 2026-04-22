"""
TAFAKI TRADER — Web App Backend
Flask + Yahoo Finance + Technical Analysis
"""

import os, math, warnings
from datetime import datetime
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
            "updated":  datetime.now().strftime("%H:%M:%S"),
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
    return jsonify({"data": results, "time": datetime.now().strftime("%H:%M:%S")})

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

@app.route("/")
def index():
    return render_template_string(HTML_APP)

# ── HTML Frontend ────────────────────────────────────────
HTML_APP = """<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
<title>Tafaki Trader</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Syne:wght@700;800&display=swap" rel="stylesheet">
<style>
:root {
  --bg:    #080d14;
  --bg2:   #0d1520;
  --bg3:   #111d2e;
  --border:#1a2d45;
  --cyan:  #00d4ff;
  --green: #00ff88;
  --red:   #ff3d5a;
  --yellow:#ffd000;
  --white: #e8f0ff;
  --muted: #4a6080;
  --font-mono: 'JetBrains Mono', monospace;
  --font-head: 'Syne', sans-serif;
}
* { box-sizing:border-box; margin:0; padding:0; }
body {
  background: var(--bg);
  color: var(--white);
  font-family: var(--font-mono);
  min-height: 100vh;
}

/* HEADER */
.header {
  background: linear-gradient(135deg, #080d14 0%, #0a1628 100%);
  border-bottom: 1px solid var(--border);
  padding: 14px 16px;
  display: flex; align-items:center; gap:12px;
  position: sticky; top:0; z-index:100;
  backdrop-filter: blur(20px);
}
.logo {
  font-family: var(--font-head);
  font-size: 18px; font-weight:800;
  background: linear-gradient(135deg, var(--cyan), var(--green));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  letter-spacing: -0.5px;
}
.header-right { margin-left:auto; display:flex; align-items:center; gap:10px; }
.sesi-badge {
  font-size:11px; font-weight:700; padding:4px 10px;
  border-radius:20px; letter-spacing:0.06em;
  border: 1px solid currentColor;
}
.sesi-green  { color:var(--green); }
.sesi-yellow { color:var(--yellow); }
.sesi-red    { color:var(--red); }
.time-text { font-size:12px; color:var(--muted); }

/* MAIN */
.main { padding:16px; max-width:480px; margin:0 auto; }

/* SETUP PANEL */
.setup-panel {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 16px;
  margin-bottom: 16px;
}
.setup-title {
  font-size:11px; color:var(--muted);
  letter-spacing:0.12em; margin-bottom:12px;
  text-transform:uppercase;
}
.ticker-inputs {
  display: grid; grid-template-columns:1fr 1fr 1fr;
  gap:8px; margin-bottom:12px;
}
.ticker-input {
  background: var(--bg3); border: 1px solid var(--border);
  border-radius:8px; padding:8px 10px;
  color:var(--white); font-family:var(--font-mono);
  font-size:13px; font-weight:700; text-transform:uppercase;
  text-align:center; width:100%;
  transition: border-color 0.2s;
}
.ticker-input:focus { outline:none; border-color:var(--cyan); }
.interval-row { display:flex; gap:8px; margin-bottom:12px; }
.btn-interval {
  flex:1; padding:7px; border-radius:8px;
  border:1px solid var(--border); background:var(--bg3);
  color:var(--muted); font-family:var(--font-mono);
  font-size:11px; cursor:pointer; transition:all 0.2s;
}
.btn-interval.active {
  border-color:var(--cyan); color:var(--cyan);
  background: rgba(0,212,255,0.08);
}
.btn-monitor {
  width:100%; padding:12px; border-radius:10px;
  background: linear-gradient(135deg, #005566, #003344);
  border: 1px solid var(--cyan); color:var(--cyan);
  font-family:var(--font-mono); font-size:13px; font-weight:700;
  cursor:pointer; letter-spacing:0.08em;
  transition: all 0.2s;
}
.btn-monitor:hover { background: rgba(0,212,255,0.15); }
.btn-monitor.running {
  background: linear-gradient(135deg, #003322, #002211);
  border-color:var(--green); color:var(--green);
}

/* CARD SAHAM */
.card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius:14px; padding:14px 16px;
  margin-bottom:12px;
  transition: border-color 0.3s;
  animation: slideUp 0.4s ease;
}
@keyframes slideUp {
  from { opacity:0; transform:translateY(10px); }
  to   { opacity:1; transform:translateY(0); }
}
.card.buy  { border-color: rgba(0,255,136,0.4); }
.card.sell { border-color: rgba(255,61,90,0.4); }
.card.wait { border-color: var(--border); }

.card-top {
  display:flex; align-items:center;
  justify-content:space-between; margin-bottom:12px;
}
.card-ticker {
  font-family:var(--font-head);
  font-size:20px; font-weight:800; color:var(--cyan);
}
.card-price { text-align:right; }
.price-main { font-size:18px; font-weight:700; }
.price-chg  { font-size:12px; margin-top:2px; }
.text-green { color:var(--green); }
.text-red   { color:var(--red); }
.text-yellow{ color:var(--yellow); }
.text-muted { color:var(--muted); }

/* SIGNAL BADGE */
.signal-badge {
  display:inline-block; padding:5px 14px;
  border-radius:20px; font-size:12px; font-weight:700;
  letter-spacing:0.08em; margin-bottom:12px;
}
.signal-buy   { background:rgba(0,255,136,0.15); color:var(--green); border:1px solid rgba(0,255,136,0.4); }
.signal-sell  { background:rgba(255,61,90,0.15);  color:var(--red);   border:1px solid rgba(255,61,90,0.4); }
.signal-wait  { background:rgba(255,208,0,0.1);   color:var(--yellow);border:1px solid rgba(255,208,0,0.3); }

/* KONFIRMASI DOTS */
.konf-row {
  display:flex; gap:8px; margin-bottom:12px;
}
.konf-item {
  flex:1; padding:6px 4px; border-radius:8px;
  text-align:center; font-size:10px; font-weight:700;
  letter-spacing:0.06em;
}
.konf-ok   { background:rgba(0,255,136,0.12); color:var(--green); border:1px solid rgba(0,255,136,0.3); }
.konf-no   { background:var(--bg3); color:var(--muted); border:1px solid var(--border); }

/* MINI CHART */
.mini-chart { margin-bottom:12px; }
.chart-svg  { width:100%; height:70px; display:block; }

/* TRADING PLAN */
.trading-plan {
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius:10px; padding:12px;
  font-size:12px;
}
.plan-title {
  font-size:10px; color:var(--muted);
  letter-spacing:0.12em; margin-bottom:8px;
}
.plan-grid {
  display:grid; grid-template-columns:1fr 1fr;
  gap:6px;
}
.plan-item { }
.plan-label { font-size:10px; color:var(--muted); margin-bottom:2px; }
.plan-value { font-size:13px; font-weight:700; }

/* FOOTER */
.footer {
  text-align:center; padding:20px 16px;
  font-size:11px; color:var(--muted);
}
.pulse {
  display:inline-block; width:7px; height:7px;
  border-radius:50%; background:var(--green);
  margin-right:5px;
  animation: pulse 2s infinite;
}
@keyframes pulse {
  0%,100% { opacity:1; box-shadow:0 0 0 0 rgba(0,255,136,0.4); }
  50%      { opacity:0.5; box-shadow:0 0 0 5px rgba(0,255,136,0); }
}
.loading-shimmer {
  background: linear-gradient(90deg, var(--bg2) 25%, var(--bg3) 50%, var(--bg2) 75%);
  background-size:200% 100%;
  animation: shimmer 1.5s infinite;
  border-radius:8px; height:120px;
  margin-bottom:12px;
}
@keyframes shimmer {
  0%   { background-position:200% 0; }
  100% { background-position:-200% 0; }
}
.refresh-bar {
  height:3px; background:var(--border);
  border-radius:3px; overflow:hidden; margin-bottom:16px;
}
.refresh-progress {
  height:100%; background:var(--cyan);
  width:0%; transition:width linear;
}
</style>
</head>
<body>

<div class="header">
  <div class="logo">TAFAKI TRADER</div>
  <div class="header-right">
    <span class="sesi-badge sesi-green" id="sesiBadge">--</span>
    <span class="time-text" id="timeText">--:--:--</span>
  </div>
</div>

<div class="main">

  <!-- Setup -->
  <div class="setup-panel">
    <div class="setup-title">Watchlist & Interval</div>
    <div class="ticker-inputs">
      <input class="ticker-input" id="t1" value="BNBR" maxlength="6" placeholder="SAHAM 1">
      <input class="ticker-input" id="t2" value="BBCA" maxlength="6" placeholder="SAHAM 2">
      <input class="ticker-input" id="t3" value="TLKM" maxlength="6" placeholder="SAHAM 3">
    </div>
    <div class="interval-row">
      <button class="btn-interval active" data-iv="1m"  onclick="setInterval_('1m')">1m</button>
      <button class="btn-interval active" data-iv="5m"  onclick="setInterval_('5m')">5m</button>
      <button class="btn-interval"        data-iv="15m" onclick="setInterval_('15m')">15m</button>
      <button class="btn-interval"        data-iv="1h"  onclick="setInterval_('1h')">1h</button>
    </div>
    <button class="btn-monitor" id="btnMonitor" onclick="toggleMonitor()">
      ▶ MULAI MONITOR
    </button>
  </div>

  <!-- Refresh bar -->
  <div class="refresh-bar">
    <div class="refresh-progress" id="refreshBar"></div>
  </div>

  <!-- Cards container -->
  <div id="cards">
    <div class="loading-shimmer"></div>
    <div class="loading-shimmer"></div>
    <div class="loading-shimmer"></div>
  </div>

  <div class="footer">
    <span class="pulse"></span>
    Data delay ~15 menit · Yahoo Finance Free<br>
    Bukan saran investasi resmi
  </div>
</div>

<script>
let selectedInterval = '5m';
let monitorInterval  = null;
let refreshTimer     = null;
let REFRESH_SECS     = 120;
let countdown        = REFRESH_SECS;
let isRunning        = false;

function setInterval_(iv) {
  selectedInterval = iv;
  document.querySelectorAll('.btn-interval').forEach(b => {
    b.classList.toggle('active', b.dataset.iv === iv);
  });
}

function toggleMonitor() {
  if (isRunning) {
    stopMonitor();
  } else {
    startMonitor();
  }
}

function startMonitor() {
  isRunning = true;
  document.getElementById('btnMonitor').textContent = '■ STOP MONITOR';
  document.getElementById('btnMonitor').classList.add('running');
  fetchData();
  monitorInterval = setInterval(fetchData, REFRESH_SECS * 1000);
  startCountdown();
}

function stopMonitor() {
  isRunning = false;
  clearInterval(monitorInterval);
  clearInterval(refreshTimer);
  document.getElementById('btnMonitor').textContent = '▶ MULAI MONITOR';
  document.getElementById('btnMonitor').classList.remove('running');
  document.getElementById('refreshBar').style.width = '0%';
}

function startCountdown() {
  countdown = REFRESH_SECS;
  clearInterval(refreshTimer);
  const bar = document.getElementById('refreshBar');
  bar.style.transition = 'none';
  bar.style.width = '100%';
  setTimeout(() => {
    bar.style.transition = `width ${REFRESH_SECS}s linear`;
    bar.style.width = '0%';
  }, 50);
}

async function fetchSesi() {
  try {
    const r = await fetch('/api/sesi');
    const d = await r.json();
    const badge = document.getElementById('sesiBadge');
    badge.textContent = d.label;
    badge.className = `sesi-badge sesi-${d.color}`;
    document.getElementById('timeText').textContent = d.time;
  } catch(e) {}
}

async function fetchData() {
  const t1 = document.getElementById('t1').value || 'BNBR';
  const t2 = document.getElementById('t2').value || 'BBCA';
  const t3 = document.getElementById('t3').value || 'TLKM';
  const tickers = [t1,t2,t3].join(',');

  fetchSesi();

  try {
    const r = await fetch(`/api/analyze?tickers=${tickers}&interval=${selectedInterval}`);
    const d = await r.json();
    renderCards(d.data);
    startCountdown();
  } catch(e) {
    document.getElementById('cards').innerHTML =
      `<div style="text-align:center;color:var(--muted);padding:40px">
        Gagal mengambil data. Cek koneksi internet.
      </div>`;
  }
}

function fmt(n) {
  return 'Rp ' + Number(n).toLocaleString('id-ID');
}

function miniChart(history) {
  if (!history || history.length < 2) return '';
  const closes = history.map(h => h.c);
  const ema9s  = history.map(h => h.e9).filter(v => v);
  const min = Math.min(...closes);
  const max = Math.max(...closes);
  const range = max - min || 1;
  const W = 340, H = 70, PAD = 4;

  const toX = i => PAD + i * (W - PAD*2) / (closes.length - 1);
  const toY = v => H - PAD - (v - min) / range * (H - PAD*2);

  // Close line
  const closePts = closes.map((c,i) => `${toX(i)},${toY(c)}`).join(' ');

  // Color segments
  let segs = '';
  for (let i=1; i<closes.length; i++) {
    const col = closes[i] >= closes[i-1] ? '#00ff88' : '#ff3d5a';
    segs += `<line x1="${toX(i-1)}" y1="${toY(closes[i-1])}" x2="${toX(i)}" y2="${toY(closes[i])}" stroke="${col}" stroke-width="1.5" opacity="0.8"/>`;
  }

  // EMA9
  const ema9Pts = history.map((h,i) => h.e9 ? `${toX(i)},${toY(h.e9)}` : null).filter(Boolean).join(' ');
  const emaLine = ema9Pts ? `<polyline points="${ema9Pts}" fill="none" stroke="#00d4ff" stroke-width="1" opacity="0.6"/>` : '';

  // Last price dot
  const lx = toX(closes.length-1), ly = toY(closes[closes.length-1]);
  const lastCol = closes[closes.length-1] >= closes[closes.length-2] ? '#00ff88' : '#ff3d5a';

  return `<svg class="chart-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
    <rect width="${W}" height="${H}" fill="#0d1520" rx="8"/>
    ${segs}
    ${emaLine}
    <circle cx="${lx}" cy="${ly}" r="3" fill="${lastCol}"/>
  </svg>`;
}

function renderCards(data) {
  if (!data || !data.length) {
    document.getElementById('cards').innerHTML =
      `<div style="text-align:center;color:var(--muted);padding:40px">
        Tidak ada data. Cek kode saham.
      </div>`;
    return;
  }

  let html = '';
  for (const r of data) {
    const isBuy  = r.signal.includes('BUY');
    const isSell = r.signal.includes('SELL');
    const cardClass = isBuy ? 'buy' : isSell ? 'sell' : 'wait';
    const sigClass  = isBuy ? 'signal-buy' : isSell ? 'signal-sell' : 'signal-wait';
    const chgColor  = r.chg >= 0 ? 'text-green' : 'text-red';
    const chgSign   = r.chg >= 0 ? '+' : '';

    const konf = [
      {label:'EMA', ok: r.cond_ema},
      {label:'VOL', ok: r.cond_vol},
      {label:'RSI', ok: r.cond_rsi},
      {label:'CMF', ok: r.cond_cmf},
    ];
    const konfHtml = konf.map(k =>
      `<div class="konf-item ${k.ok ? 'konf-ok' : 'konf-no'}">${k.label}</div>`
    ).join('');

    // Trading plan (only for BUY/SELL)
    let planHtml = '';
    if ((isBuy || isSell) && r.entry > 0) {
      planHtml = `
        <div class="trading-plan">
          <div class="plan-title">TRADING PLAN</div>
          <div class="plan-grid">
            <div class="plan-item">
              <div class="plan-label">ENTRY</div>
              <div class="plan-value">${fmt(r.entry)}</div>
            </div>
            <div class="plan-item">
              <div class="plan-label">STOP LOSS</div>
              <div class="plan-value text-red">${fmt(r.stoploss)}</div>
            </div>
            <div class="plan-item">
              <div class="plan-label">TARGET 1 (+${r.gain1}%)</div>
              <div class="plan-value text-green">${fmt(r.target1)}</div>
            </div>
            <div class="plan-item">
              <div class="plan-label">TARGET 2 (+${r.gain2}%)</div>
              <div class="plan-value text-green">${fmt(r.target2)}</div>
            </div>
          </div>
          <div style="margin-top:8px;font-size:10px;color:var(--muted)">
            R:R → T1: 1:${r.rr1}  |  T2: 1:${r.rr2}  |  ATR: ${fmt(r.atr)}
          </div>
        </div>`;
    } else if (r.conf >= 2) {
      planHtml = `
        <div class="trading-plan">
          <div class="plan-title">POTENSI ENTRY (jika 4/4)</div>
          <div class="plan-grid">
            <div class="plan-item">
              <div class="plan-label">ENTRY</div>
              <div class="plan-value text-yellow">${fmt(r.close)}</div>
            </div>
            <div class="plan-item">
              <div class="plan-label">STOP LOSS</div>
              <div class="plan-value">${fmt(r.stoploss > 0 ? r.stoploss : r.support)}</div>
            </div>
            <div class="plan-item">
              <div class="plan-label">TARGET 1</div>
              <div class="plan-value">${fmt(r.target1 > 0 ? r.target1 : r.resistance)}</div>
            </div>
            <div class="plan-item">
              <div class="plan-label">R/S ZONE</div>
              <div class="plan-value text-muted">${fmt(r.support)} / ${fmt(r.resistance)}</div>
            </div>
          </div>
        </div>`;
    }

    // EMA cross tag
    const crossTag = r.ema_cross_up
      ? `<span style="font-size:10px;color:var(--green);margin-left:8px">⚡ EMA CROSS!</span>`
      : r.ema_cross_down
      ? `<span style="font-size:10px;color:var(--red);margin-left:8px">⚡ CROSS DOWN!</span>`
      : '';

    html += `
      <div class="card ${cardClass}">
        <div class="card-top">
          <div>
            <div class="card-ticker">${r.ticker}${crossTag}</div>
            <div style="font-size:11px;color:var(--muted);margin-top:2px">
              RSI:${r.rsi || '--'} · CMF:${r.cmf > 0 ? '+' : ''}${r.cmf} · Vol:${r.vol_ratio}x
            </div>
          </div>
          <div class="card-price">
            <div class="price-main">${fmt(r.close)}</div>
            <div class="price-chg ${chgColor}">${chgSign}${r.chg}%</div>
          </div>
        </div>

        <span class="signal-badge ${sigClass}">${r.signal} &nbsp; ${r.conf}/4</span>

        <div class="konf-row">${konfHtml}</div>

        <div class="mini-chart">${miniChart(r.history)}</div>

        ${planHtml}

        <div style="font-size:10px;color:var(--muted);margin-top:8px;text-align:right">
          Update: ${r.updated}
        </div>
      </div>`;
  }

  document.getElementById('cards').innerHTML = html;
}

// Update sesi setiap menit
setInterval(fetchSesi, 60000);
fetchSesi();

// Auto-start saat load
window.onload = () => startMonitor();
</script>
</body>
</html>"""

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
