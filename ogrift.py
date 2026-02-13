import atexit
import os
import sys
import json
import time
import signal
import asyncio
import logging
import argparse
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import ccxt
from dotenv import load_dotenv

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)
# TEST_SYNC = 1

# ------------------------------------------------------------
# OUTPUT / LOGGING (backend visibility)
# ------------------------------------------------------------
sys.stdout.reconfigure(line_buffering=True)

LOG = logging.getLogger("rift")
LOG.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
LOG.addHandler(_handler)

# ------------------------------------------------------------
# ENV
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(str(BASE_DIR / "info.env"))

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
if not BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN in info.env")

# ------------------------------------------------------------
# FILES
# ------------------------------------------------------------
STATE_FILE = str(BASE_DIR / "state.json")
CONTROLS_FILE = str(BASE_DIR / "controls.json")
UNIVERSE_FILE = str(BASE_DIR / "universe.json")
TRADE_LOG_FILE = os.getenv('RIFT_TRADE_LOG_FILE', str(BASE_DIR / 'trades.jsonl'))  # append-only forensic log (json per line)
# ----------------------------
# Exit Intelligence Enhancements
# ----------------------------
ATR_PERIOD = 14
ATR_MULT_NORMAL = 1.5      # normal volatility trail
ATR_MULT_TIGHT = 1.0       # tightened trail on momentum decay

RSI_MOMENTUM_WEAK = 55
VOLUME_DIVERGENCE_LOOKBACK = 6

# ------------------------------------------------------------
# CORE STRATEGY SETTINGS
# ------------------------------------------------------------
TIMEFRAME = "1m"
SCAN_INTERVAL = 30  # seconds
CANDLE_LIMIT = 120

EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 14
RSI_MIN = 55
RSI_MAX = 75
PEG_GUARD_ENABLED = True
PEG_PRICE_LOW = 0.98
PEG_PRICE_HIGH = 1.02
PEG_ATR_PCT_MAX = 0.0015  # 0.15% ATR (very stable)

# ------------------------------------------------------------
# PRICE SANITY / DUST + LIQUIDITY / SPREAD GUARDS (integrity)
# ------------------------------------------------------------
MIN_PRICE_FLOOR = 0.0005           # blocks near-zero "dust" assets slipping into positions (ex: LUNC/USD)
MAX_LEADING_ZEROS = 3              # blocks 0.0000xxxx prices
MIN_NOTIONAL_USD = 5.00            # avoid dust orders / tiny fills
MIN_24H_QUOTE_VOL_USD = 250000.0   # liquidity floor for entries (USD-ish quote volume)
MAX_SPREAD_PCT = 0.004
# Fees (taker, round-trip). If exchange provides market taker fee, we use it; otherwise default.
DEFAULT_TAKER_FEE_RATE = 0.0010     # 0.10% per side (typical spot taker)
INCLUDE_FEES_IN_PNL = True          # net PnL includes fees; exits evaluate net unreal too

# Exchange order constraints (precision/limits)
ENFORCE_MARKET_LIMITS = True
MIN_COST_FALLBACK_USD = 5.00        # if exchange doesn't report min cost
MIN_AMOUNT_FALLBACK = 0.0           # if exchange doesn't report min amount

# Equity curve guardrails (beyond cold stand-down): throttle exposure + raise entry threshold when under water
EQUITY_GUARD_LOOKBACK_TRADES = 50
EQUITY_GUARD_MIN_TRADES = 15
EQUITY_GUARD_MIN_EXPOSURE_MULT = 0.40   # never go below 40% of normal exposure
EQUITY_GUARD_SCORE_BUMP_MAX = 0.08      # add up to +0.08 to ENTER_SCORE when in drawdown
EQUITY_GUARD_DD_SOFT = 0.02             # 2% drawdown -> start throttling
EQUITY_GUARD_DD_HARD = 0.06             # 6% drawdown -> max throttle
             # 0.40% mid spread; blocks thin books / slippage traps

# ------------------------------------------------------------
# GOVERNORS
# ------------------------------------------------------------
# 1) Regime Authority (governs entries)
REGIME_MIN_TREND_SCORE = 0.55
REGIME_MAX_VOL_SPIKE = 2.25        # ATR_now / ATR_baseline
REGIME_COOLDOWN_SECONDS = 120

# 2) Score + Persistence (hysteresis + minimum streak; no flip-flop)
ENTER_SCORE = 0.78
EXIT_SCORE = 0.62
PERSIST_TICKS_REQUIRED = 3
CANDIDATE_TTL_SECONDS = 90

# 3) Performance-aware gating ("cold bot" stands down)
COLD_MIN_TRADES = 12
COLD_LOOKBACK = 25
COLD_MAX_DD_PCT = 0.035
COLD_MIN_EXPECTANCY_USD = -0.25
COLD_STANDDOWN_SECONDS = 20 * 60

# ------------------------------------------------------------
# BE/TRAIL Tight Gates (prevents BE/profit-lock triggering when peak ~ 0)
# ------------------------------------------------------------
BE_MIN_PEAK_USD = 2.00
BE_ARM_AT_PROFIT_USD = 1.00
BE_MIN_HOLD_SECONDS = 45
TRAIL_MIN_PEAK_USD = 3.50
TRAIL_MIN_HOLD_SECONDS = 45


# ----------------------------
# AUTO RISK PROFILE (SMALL <-> STANDARD) with hysteresis
# ----------------------------
SMALL_TO_STANDARD_EQUITY = 600.0
STANDARD_TO_SMALL_EQUITY = 450.0

# STANDARD profile: fixed $ exits (good for mid/large accounts)
STD_TP_DOLLARS = 25.0
STD_SL_DOLLARS = -15.0
STD_TOTAL_EXPOSURE = 0.30  # total deployed across all open positions

# SMALL profile: exits scale with equity (safe for tiny accounts)
SMALL_TP_EQUITY_PCT = 0.004   # +0.4% of equity (ex: $100 -> +$0.40)
SMALL_SL_EQUITY_PCT = -0.003  # -0.3% of equity (ex: $100 -> -$0.30)
SMALL_TOTAL_EXPOSURE = 0.30   # keep total exposure budget consistent

# ----------------------------
# Break-Even (BE) + Trailing (profile-aware via TP dollars)
# ----------------------------
# These are expressed as fractions of the *current* TP target (which changes by profile).
# Example (STANDARD TP=$25):
# - BE arms at +$10.00 and exits at ~$0.00 if it comes back.
# - Trailing activates at +$17.50 and gives back $7.50 from peak.
BE_TRIGGER_TP_FRACTION = 0.40
BE_EXIT_UNREAL_DOLLARS = 0.00  # 0.00 = true break-even (paper). Set to small +$ if you want buffer.
POST_BE_GIVEBACK_FRACTION = 0.25  # 25% of TP
MIN_POST_BE_GIVEBACK = 0.50       # $0.50 floor

TRAIL_TRIGGER_TP_FRACTION = 0.70
TRAIL_GIVEBACK_TP_FRACTION = 0.30

# Guards so SMALL account doesn’t end up with “$0.01 logic”
MIN_BE_TRIGGER_DOLLARS = 0.05
MIN_TRAIL_TRIGGER_DOLLARS = 0.10
MIN_TRAIL_GIVEBACK_DOLLARS = 0.05
# ----------------------------
# Post-BE Profit Lock (prevents BE -> back to $0 donation)
# ----------------------------
POST_BE_LOCK_ENABLED = True
POST_BE_GIVEBACK_TP_FRACTION = 0.25   # 25% of TP (STANDARD: ~6.25 if TP=25)
MIN_POST_BE_GIVEBACK_DOLLARS = 2.00   # floor so it works on smaller TP days

# ----------------------------
# Diversity: one base asset at a time
# ----------------------------
# If True: you can only hold ONE position per base (e.g., PAXG) regardless of quote/exchange.
ONE_BASE_ASSET_AT_A_TIME = True

# Stagnation exits
MAX_HOLD_SECONDS = 90 * 60
RECOVERY_WINDOW_SECONDS = 30 * 60

# Stagnation upgrade: after recovery window ends, do NOT force-close red.
# Arm "exit-on-green" and close the first moment unreal turns > threshold.
EXIT_ON_GREEN_AFTER_STAG_TIMEOUT = True
EXIT_ON_GREEN_MIN_UNREAL = 0.00           # any green: close when unreal > 0.00
EXIT_ON_GREEN_MAX_WAIT_SECONDS = 20 * 60  # safety: if still not green after X seconds, force close (0 disables)

# Portfolio / risk
START_EQUITY_USD = 10_000.0
MAX_OPEN_POSITIONS = 3
COOLDOWN_SECONDS = 45 * 60

# Exchanges (public data only)
EXCHANGE_NAMES = ["coinbase", "kraken"]

# Universe scanning
BATCH_SIZE_DEFAULT = 30
UNIVERSE_REFRESH_SECONDS = 60 * 60

ALLOWED_QUOTES = {"USD", "USDT", "USDC"}
BAD_SYMBOL_SUBSTRINGS = ("UP/", "DOWN/", "BULL/", "BEAR/")
# ------------------------------------------------------------
# STABLECOIN / PEG FILTER (kills unproductive USD-pegged pairs)
# ------------------------------------------------------------
EXCLUDE_STABLE_BASES = True

STABLE_BASES = {
    "USD", "USDT", "USDC", "DAI", "TUSD", "USDP", "GUSD", "PAX", "PAXG",  # (PAXG is gold-pegged; remove if you want it)
    "FDUSD", "USDE", "FRAX", "LUSD", "PYUSD", "USDD", "USDJ", "USDN",
    "EURC", "EURS", "EURO",  # euro stables (optional)
}

# common stable-ish ticker patterns (catches weird ones)
STABLE_BASE_SUBSTRINGS = ("USD", "USDT", "USDC", "DAI", "EUR")

# Heartbeat visibility
HEARTBEAT_SECONDS = 30

_shutdown_signal = False
# ------------------------------------------------------------
# SINGLE-INSTANCE LOCKFILE
# ------------------------------------------------------------
LOCKFILE_PATH = os.path.join(BASE_DIR, "rift.lock")
_lock_fd = None


def acquire_lock():
    global _lock_fd
    try:
        _lock_fd = os.open(LOCKFILE_PATH, os.O_CREAT | os.O_EXCL | os.O_RDWR)
        os.write(_lock_fd, str(os.getpid()).encode())
    except FileExistsError:
        print("[RIFT] ❌ Another instance is already running. Exiting.")
        sys.exit(1)


def release_lock():
    global _lock_fd
    try:
        if _lock_fd is not None:
            os.close(_lock_fd)
        if os.path.exists(LOCKFILE_PATH):
            os.remove(LOCKFILE_PATH)
    except Exception:
        pass


# ============================================================
# SIGNALS
# ============================================================
def _handle_shutdown(sig, frame):
    global _shutdown_signal
    LOG.info("shutdown signal received")
    _shutdown_signal = True


signal.signal(signal.SIGINT, _handle_shutdown)
signal.signal(signal.SIGTERM, _handle_shutdown)
atexit.register(release_lock)


# ============================================================
# JSON HELPERS
# ============================================================
def load_json(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path: str, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


# ============================================================
# FORENSIC TRADE LOG (append-only jsonl)
# ============================================================
def append_trade_log(evt: Dict[str, Any]) -> None:
    try:
        evt = dict(evt)
        evt["ts"] = int(evt.get("ts", time.time()))
        with open(TRADE_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(evt, separators=(",", ":")) + "\n")
    except Exception:
        # never crash engine because logging failed
        pass


def load_recent_trades(n: int = 200) -> List[Dict[str, Any]]:
    if not os.path.exists(TRADE_LOG_FILE):
        return []
    out: List[Dict[str, Any]] = []
    try:
        with open(TRADE_LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        return out[-n:]
    except Exception:
        return out[-n:]


def compute_perf_metrics(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trades:
        return {"n": 0}

    recent = trades[-COLD_LOOKBACK:]
    n = len(recent)

    pnls = [float(t.get("pnl", 0.0)) for t in recent if t.get("evt") == "CLOSE"]
    if not pnls:
        return {"n": 0}

    wins = [p for p in pnls if p > 0]
    losses = [-p for p in pnls if p < 0]

    win_rate = (len(wins) / len(pnls)) if pnls else 0.0
    avg_win = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss = (sum(losses) / len(losses)) if losses else 0.0
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    # simple drawdown estimate from cumulative pnl
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)  # negative
    max_dd_abs = abs(max_dd)

    return {
        "n": len(pnls),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "max_dd_abs": max_dd_abs,
    }


def perf_allows_entries(state: Dict[str, Any], now_ts: int) -> Tuple[bool, str]:
    perf = state.get("perf", {}) or {}
    cold_until = int(perf.get("cold_until", 0) or 0)
    if now_ts < cold_until:
        return False, "COLD_STANDDOWN_ACTIVE"

    trades = load_recent_trades(300)
    metrics = compute_perf_metrics(trades)
    perf["last_metrics"] = metrics
    state["perf"] = perf

    if metrics.get("n", 0) < COLD_MIN_TRADES:
        return True, "WARMING_UP"

    # drawdown% proxy: DD relative to recent profit potential (avoid div by 0)
    denom = max(1.0, abs(metrics.get("avg_win", 1.0)) * max(1, metrics.get("n", 1)))
    dd_pct = float(metrics.get("max_dd_abs", 0.0)) / denom

    if dd_pct >= COLD_MAX_DD_PCT or float(metrics.get("expectancy", 0.0)) <= COLD_MIN_EXPECTANCY_USD:
        perf["cold_until"] = now_ts + int(COLD_STANDDOWN_SECONDS)
        state["perf"] = perf
        return False, "COLD_TRIGGERED"

    return True, "PERF_OK"

# ============================================================
# STATE / CONTROLS
# ============================================================
def default_state() -> Dict[str, Any]:
    return {
        "equity": START_EQUITY_USD,
        "realized_pnl": 0.0,
        "positions": {},      # pid -> {...}
        "cooldowns": {},      # symbol -> unix_ts
        "risk_profile": "STANDARD",  # persisted for hysteresis stability

        # governors (persist for restart integrity)
        "regime": {"ok": True, "label": "UNKNOWN", "score": 0.0, "reason": "", "last_block_ts": 0},
        "candidates": {},
        "perf": {"cold_until": 0, "last_metrics": {}},
        "loss_clusters": {},
    }




def update_equity_guard(state: Dict[str, Any], now_ts: int) -> None:
    """Throttle exposure + raise entry threshold when equity is below its recent peak."""
    eg = state.get("equity_guard", {}) or {}

    # Initialize equity_start if missing
    if "equity_start" not in state:
        state["equity_start"] = float(state.get("equity", 0.0))

    trades = load_recent_trades(max(300, EQUITY_GUARD_LOOKBACK_TRADES + 50))
    closes = [t for t in trades if (t or {}).get("evt") == "CLOSE"]

    if len(closes) < int(EQUITY_GUARD_MIN_TRADES):
        state["exposure_mult"] = 1.0
        state["score_bump"] = 0.0
        eg.update({"dd_pct": 0.0, "peak_equity": float(state.get("equity", 0.0)), "updated_ts": now_ts})
        state["equity_guard"] = eg
        return

    # Reconstruct equity curve from equity_start + net pnl
    eq0 = float(state.get("equity_start", 0.0))
    eq = eq0
    peak = eq0
    dd_pct = 0.0
    for t in closes[-int(EQUITY_GUARD_LOOKBACK_TRADES):]:
        eq += float(t.get("pnl", 0.0))
        peak = max(peak, eq)
        if peak > 0:
            dd_pct = max(dd_pct, (peak - eq) / peak)

    dd_pct = float(dd_pct)
    # Map drawdown to throttles
    if dd_pct <= float(EQUITY_GUARD_DD_SOFT):
        exposure_mult = 1.0
        score_bump = 0.0
    else:
        span = max(1e-9, float(EQUITY_GUARD_DD_HARD) - float(EQUITY_GUARD_DD_SOFT))
        p = min(1.0, (dd_pct - float(EQUITY_GUARD_DD_SOFT)) / span)
        exposure_mult = 1.0 - p * (1.0 - float(EQUITY_GUARD_MIN_EXPOSURE_MULT))
        exposure_mult = max(float(EQUITY_GUARD_MIN_EXPOSURE_MULT), min(1.0, exposure_mult))
        score_bump = p * float(EQUITY_GUARD_SCORE_BUMP_MAX)

    state["exposure_mult"] = float(exposure_mult)
    state["score_bump"] = float(score_bump)

    eg.update({"dd_pct": dd_pct, "peak_equity": peak, "updated_ts": now_ts})
    state["equity_guard"] = eg
def load_state() -> Dict[str, Any]:
    s = load_json(STATE_FILE, default_state())
    base = default_state()
    if isinstance(s, dict):
        base.update(s)

    base["positions"] = base.get("positions", {}) or {}
    base["cooldowns"] = base.get("cooldowns", {}) or {}
    base["candidates"] = base.get("candidates", {}) or {}
    base["perf"] = base.get("perf", {}) or {"cold_until": 0, "last_metrics": {}}
    base["loss_clusters"] = base.get("loss_clusters", {}) or {}

    base["equity"] = float(base.get("equity", START_EQUITY_USD))
    base["realized_pnl"] = float(base.get("realized_pnl", 0.0))

    rp = base.get("risk_profile", "STANDARD")
    base["risk_profile"] = rp if rp in ("SMALL", "STANDARD") else "STANDARD"

    reg = base.get("regime", {}) or {"ok": True, "label": "UNKNOWN", "score": 0.0, "reason": "", "last_block_ts": 0}
    base["regime"] = reg

    # backward-compatible fields used by telegram renderers
    base["regime_ok"] = bool(reg.get("ok", True))
    base["regime_last_msg"] = str(reg.get("reason", "") or "")

    return base



def save_state(state: Dict[str, Any]) -> None:
    save_json(STATE_FILE, state)


def normalize_cooldowns(state: Dict[str, Any]) -> None:
    cds = state.get("cooldowns", {}) or {}
    fixed: Dict[str, int] = {}
    for k, v in cds.items():
        try:
            ts = int(v)
        except Exception:
            continue
        symbol = k.split(":", 1)[1] if ":" in k else k
        prev = fixed.get(symbol, 0)
        fixed[symbol] = ts if ts > prev else prev
    state["cooldowns"] = fixed


def default_controls() -> Dict[str, Any]:
    return {
        "pause_entries": False,
        "shutdown": False,
        "close_all": False,
        "restart": False,  # restart closes all + continues
        "heartbeat_minutes": 1,  # informational only; heartbeat uses HEARTBEAT_SECONDS
        "print_positions_now": False,
        "batch_size": BATCH_SIZE_DEFAULT,
    }


def load_controls() -> Dict[str, Any]:
    c = load_json(CONTROLS_FILE, default_controls())
    base = default_controls()
    if isinstance(c, dict):
        base.update(c)

    if "Batch_size" in base and "batch_size" not in base:
        base["batch_size"] = base.pop("Batch_size")

    try:
        bs = int(base.get("batch_size", BATCH_SIZE_DEFAULT))
    except Exception:
        bs = BATCH_SIZE_DEFAULT
    base["batch_size"] = max(5, min(bs, 200))
    return base


def save_controls(c: Dict[str, Any]) -> None:
    save_json(CONTROLS_FILE, c)


# ============================================================
# AUTO PROFILE + RISK HELPERS
# ============================================================
def compute_profile_with_hysteresis(current: str, equity: float) -> str:
    current = current if current in ("SMALL", "STANDARD") else "STANDARD"
    e = float(equity)

    if current == "SMALL":
        if e >= float(SMALL_TO_STANDARD_EQUITY):
            return "STANDARD"
        return "SMALL"

    # current == STANDARD
    if e <= float(STANDARD_TO_SMALL_EQUITY):
        return "SMALL"
    return "STANDARD"


def ensure_active_profile(state: Dict[str, Any]) -> str:
    cur = state.get("risk_profile", "STANDARD")
    eq = float(state.get("equity", 0.0))
    nxt = compute_profile_with_hysteresis(cur, eq)
    if nxt != cur:
        LOG.info(f"[RIFT] profile switch {cur} -> {nxt} (equity=${eq:,.2f})")
    state["risk_profile"] = nxt
    return nxt


def get_tp_sl_dollars(state: Dict[str, Any]) -> Tuple[float, float, str]:
    prof = ensure_active_profile(state)
    eq = float(state.get("equity", 0.0))

    if prof == "SMALL":
        tp = eq * float(SMALL_TP_EQUITY_PCT)
        sl = eq * float(SMALL_SL_EQUITY_PCT)  # negative
        return tp, sl, prof

    return float(STD_TP_DOLLARS), float(STD_SL_DOLLARS), prof


def get_position_value(state: Dict[str, Any]) -> Tuple[float, str]:
    prof = ensure_active_profile(state)
    eq = float(state.get("equity", 0.0))

    total = float(SMALL_TOTAL_EXPOSURE) if prof == "SMALL" else float(STD_TOTAL_EXPOSURE)
    # Option A: spread total exposure across max slots (so raising slots doesn't multiply exposure)
    ex_mult = float(state.get('exposure_mult', 1.0) or 1.0)
    per_trade = ((eq * total) / float(MAX_OPEN_POSITIONS)) * ex_mult
    return per_trade, prof


def get_be_trail_params(state: Dict[str, Any]) -> Tuple[float, float, float]:
    """
    Returns:
      be_trigger_unreal, trail_trigger_unreal, trail_giveback_unreal
    All in dollars of unrealized PnL (paper).
    """
    tp_dollars, _, _ = get_tp_sl_dollars(state)

    be_trigger = max(float(tp_dollars) * float(BE_TRIGGER_TP_FRACTION), float(MIN_BE_TRIGGER_DOLLARS))
    trail_trigger = max(float(tp_dollars) * float(TRAIL_TRIGGER_TP_FRACTION), float(MIN_TRAIL_TRIGGER_DOLLARS))
    trail_giveback = max(float(tp_dollars) * float(TRAIL_GIVEBACK_TP_FRACTION), float(MIN_TRAIL_GIVEBACK_DOLLARS))

    # Safety: giveback should never exceed trigger
    if trail_giveback >= trail_trigger:
        trail_giveback = max(trail_trigger * 0.5, float(MIN_TRAIL_GIVEBACK_DOLLARS))

    return be_trigger, trail_trigger, trail_giveback


def base_asset(symbol: str) -> str:
    try:
        return symbol.split("/", 1)[0].strip().upper()
    except Exception:
        return ""


# ============================================================
# INDICATORS (NO NUMPY)
# ============================================================
def ema_last(values: List[float], period: int) -> Optional[float]:
    if len(values) < period + 1:
        return None
    alpha = 2.0 / (period + 1.0)
    e = sum(values[:period]) / period
    for v in values[period:]:
        e = alpha * v + (1 - alpha) * e
    return e


def rsi_last(values: List[float], period: int) -> Optional[float]:
    if len(values) < period + 1:
        return None

    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        d = values[i] - values[i - 1]
        if d >= 0:
            gains += d
        else:
            losses -= d

    avg_gain = gains / period
    avg_loss = losses / period

    for i in range(period + 1, len(values)):
        d = values[i] - values[i - 1]
        gain = d if d > 0 else 0.0
        loss = -d if d < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def atr_last(candles: List[list], period: int) -> Optional[float]:
    if len(candles) < period + 1:
        return None

    trs = []
    for i in range(1, len(candles)):
        h = float(candles[i][2])
        l = float(candles[i][3])
        pc = float(candles[i - 1][4])
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))

    window = trs[-period:]
    return sum(window) / period if window else None


def momentum_weak(closes, volumes, ema_fast, ema_slow, rsi) -> bool:
    if rsi is not None and rsi < RSI_MOMENTUM_WEAK:
        return True
    if ema_fast is not None and ema_slow is not None and ema_fast < ema_slow:
        return True

    if len(closes) >= VOLUME_DIVERGENCE_LOOKBACK + 1:
        price_up = closes[-1] > closes[-VOLUME_DIVERGENCE_LOOKBACK]
        vol_down = (
            sum(volumes[-(VOLUME_DIVERGENCE_LOOKBACK//2):])
            < sum(volumes[:(VOLUME_DIVERGENCE_LOOKBACK//2)])
        )
        if price_up and vol_down:
            return True

    return False


# ============================================================
# UNIVERSE + VOLUME
# ============================================================
UNIVERSE_CACHE_VERSION = 2
def _is_good_symbol(symbol: str) -> bool:
    if not isinstance(symbol, str) or "/" not in symbol:
        return False
    if any(x in symbol for x in BAD_SYMBOL_SUBSTRINGS):
        return False

    base, quote = symbol.split("/", 1)
    base = base.strip().upper()
    quote = quote.strip().upper()

    if quote not in ALLOWED_QUOTES:
        return False
    if base in ALLOWED_QUOTES:
        return False

    # ---- stable / pegged filter ----
    if EXCLUDE_STABLE_BASES:
        if base in STABLE_BASES:
            return False

        # catches things like "XUSD", "USDX", "USDY", etc.
        if any(s in base for s in STABLE_BASE_SUBSTRINGS):
            # allow real coins that contain USD in name? usually not needed.
            return False

    return True


def _quote_volume_usdish(t: dict) -> float:
    if not isinstance(t, dict):
        return 0.0
    qv = t.get("quoteVolume")
    if qv is not None:
        try:
            return float(qv)
        except Exception:
            pass
    bv = t.get("baseVolume")
    last = t.get("last")
    try:
        if bv is not None and last is not None:
            return float(bv) * float(last)
    except Exception:
        pass
    return 0.0


# ============================================================
# ENTRY GOVERNOR HELPERS
# ============================================================
def _leading_zeros_after_decimal(px: float) -> int:
    s = f"{px:.12f}"
    if "." not in s:
        return 0
    frac = s.split(".", 1)[1]
    count = 0
    for ch in frac:
        if ch == "0":
            count += 1
        else:
            break
    return count


def pass_price_sanity(symbol: str, last_px: float, quote_vol_usd: Optional[float]) -> Tuple[bool, str]:
    if last_px is None or last_px <= 0:
        return False, "PRICE_INVALID"
    if float(last_px) < float(MIN_PRICE_FLOOR):
        return False, f"PRICE_FLOOR_BLOCK<{MIN_PRICE_FLOOR}"
    if _leading_zeros_after_decimal(float(last_px)) > int(MAX_LEADING_ZEROS):
        return False, "PRICE_DUST_LEADING_ZEROS"
    if quote_vol_usd is not None and float(quote_vol_usd) < float(MIN_24H_QUOTE_VOL_USD):
        return False, f"LIQUIDITY_BLOCK<{MIN_24H_QUOTE_VOL_USD}"
    return True, "OK"


def spread_pct_from_ticker(t: dict) -> Optional[float]:
    try:
        bid = t.get("bid")
        ask = t.get("ask")
        if bid is None or ask is None:
            return None
        bid = float(bid)
        ask = float(ask)
        if bid <= 0 or ask <= 0 or ask < bid:
            return None
        mid = (bid + ask) / 2.0
        return (ask - bid) / mid if mid > 0 else None
    except Exception:
        return None


def get_taker_fee_rate(ex, symbol: str) -> float:
    """Best-effort taker fee rate (per side)."""
    try:
        m = ex.market(symbol) if ex else None
        if isinstance(m, dict):
            r = m.get("taker")
            if r is not None:
                r = float(r)
                if r >= 0:
                    return r
    except Exception:
        pass
    return float(DEFAULT_TAKER_FEE_RATE)


def normalize_order_qty(ex, symbol: str, qty: float, price: float) -> Tuple[Optional[float], str]:
    """Apply exchange precision + limits (amount/cost). Returns (qty, reason)."""
    if qty is None or qty <= 0 or price <= 0:
        return None, "QTY_INVALID"

    q = float(qty)
    try:
        if hasattr(ex, "amount_to_precision"):
            q = float(ex.amount_to_precision(symbol, q))
    except Exception:
        q = float(q)

    if q <= 0:
        return None, "QTY_PRECISION_ZERO"

    if not ENFORCE_MARKET_LIMITS:
        return q, "OK"

    try:
        m = ex.market(symbol)
        limits = (m or {}).get("limits", {}) if isinstance(m, dict) else {}
        amount_lim = limits.get("amount", {}) if isinstance(limits, dict) else {}
        cost_lim = limits.get("cost", {}) if isinstance(limits, dict) else {}

        min_amt = amount_lim.get("min")
        min_cost = cost_lim.get("min")

        if min_amt is None:
            min_amt = float(MIN_AMOUNT_FALLBACK)
        if min_cost is None:
            min_cost = float(MIN_COST_FALLBACK_USD)

        if min_amt is not None and float(min_amt) > 0 and q < float(min_amt):
            return None, f"AMOUNT_MIN<{float(min_amt)}"

        notional = q * float(price)
        if min_cost is not None and float(min_cost) > 0 and notional < float(min_cost):
            return None, f"COST_MIN<{float(min_cost)}"

    except Exception:
        notional = q * float(price)
        if notional < float(MIN_COST_FALLBACK_USD):
            return None, f"COST_MIN_FALLBACK<{float(MIN_COST_FALLBACK_USD)}"

    return q, "OK"


def calc_net_unreal(p: Dict[str, Any], last: float, fee_rate: float) -> float:
    """Net unreal PnL estimate (includes paid open fee + estimated exit fee)."""
    entry = float(p.get("entry", 0.0))
    qty = float(p.get("qty", 0.0))
    gross = (float(last) - entry) * qty

    if not INCLUDE_FEES_IN_PNL:
        return gross

    fee_open = float(p.get("fee_open", 0.0))
    fee_exit_est = float(last) * qty * float(fee_rate)
    return gross - fee_open - fee_exit_est


def compute_features_from_candles(candles: List[List[float]]) -> Optional[Dict[str, Any]]:
        if not candles or len(candles) < max(EMA_SLOW + 5, RSI_PERIOD + 5, ATR_PERIOD + 5):
            return None

        closes = [float(c[4]) for c in candles if c and len(c) >= 6]
        vols = [float(c[5]) for c in candles if c and len(c) >= 6]
        if len(closes) < max(EMA_SLOW + 5, RSI_PERIOD + 5):
            return None

        last = closes[-1]
        ef = ema_last(closes, EMA_FAST)
        es = ema_last(closes, EMA_SLOW)
        r = rsi_last(closes, RSI_PERIOD)
        atr = atr_last(candles, ATR_PERIOD)

        if ef is None or es is None or r is None or atr is None or last <= 0:
            return None

        # ATR baseline: average ATR over last ~3x period
        # (no numpy; simple slice)
        baseline_window = max(ATR_PERIOD * 3, ATR_PERIOD + 1)
        atrs = []
        for i in range(len(candles) - baseline_window, len(candles)):
            if i <= 0:
                continue
            window = candles[: i + 1]
            a = atr_last(window, ATR_PERIOD)
            if a is not None:
                atrs.append(float(a))
        atr_base = sum(atrs) / len(atrs) if atrs else float(atr)

        atr_ratio = float(atr) / float(atr_base) if atr_base > 0 else 1.0

        # trend_score in [0..1] from EMA separation + RSI location
        ema_sep = (float(ef) - float(es)) / float(last)
        ema_component = max(0.0, min(1.0, (ema_sep * 500.0)))  # scaled
        rsi_component = 0.0
        if float(r) >= RSI_MIN and float(r) <= RSI_MAX:
            # closer to middle of band => higher
            mid = (RSI_MIN + RSI_MAX) / 2.0
            dist = abs(float(r) - mid) / (RSI_MAX - RSI_MIN)
            rsi_component = max(0.0, 1.0 - dist)
        trend_score = max(0.0, min(1.0, 0.6 * ema_component + 0.4 * rsi_component))

        # regime label
        if atr_ratio > float(REGIME_MAX_VOL_SPIKE):
            regime = {"label": "BLOCK", "reason": "VOL_SPIKE", "score": 0.0}
        elif trend_score < float(REGIME_MIN_TREND_SCORE):
            regime = {"label": "BLOCK", "reason": "CHOP", "score": float(trend_score)}
        else:
            regime = {"label": "ALLOW", "reason": "TREND", "score": float(trend_score)}

        # entry score in [0..1]
        # requires ef>es, price above es, RSI in band
        hard_ok = (ef > es) and (last > es) and (RSI_MIN <= r <= RSI_MAX)
        if not hard_ok:
            score = 0.0
        else:
            # score boosts with trend_score but penalize high atr_ratio (instability)
            vol_penalty = max(0.0, min(0.5, (atr_ratio - 1.0) * 0.25))
            score = max(0.0, min(1.0, float(trend_score) - vol_penalty))

        # PEG GUARD
        peg_block = False
        if PEG_GUARD_ENABLED and atr is not None and last > 0:
            atr_pct = float(atr) / float(last)
            if PEG_PRICE_LOW <= float(last) <= PEG_PRICE_HIGH and atr_pct <= float(PEG_ATR_PCT_MAX):
                peg_block = True

        return {
            "last": float(last),
            "ef": float(ef),
            "es": float(es),
            "rsi": float(r),
            "atr": float(atr),
            "atr_ratio": float(atr_ratio),
            "trend_score": float(trend_score),
            "score": float(score),
            "regime": regime,
            "peg_block": peg_block,
            "vol_lookback": float(sum(vols[-10:])) if vols else 0.0,
        }


def compute_features(ex, symbol: str) -> Optional[Dict[str, Any]]:
    candles = ex.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=CANDLE_LIMIT)
    return compute_features_from_candles(candles)


def regime_allows_entry(state: Dict[str, Any], now_ts: int, regime: Dict[str, Any]) -> Tuple[bool, str]:
    reg_state = state.get("regime", {}) or {"ok": True, "label": "UNKNOWN", "score": 0.0, "reason": "", "last_block_ts": 0}
    label = str(regime.get("label", "BLOCK"))
    reason = str(regime.get("reason", ""))
    score = float(regime.get("score", 0.0))

    if label != "ALLOW":
        reg_state["ok"] = False
        reg_state["label"] = label
        reg_state["reason"] = reason
        reg_state["score"] = score
        reg_state["last_block_ts"] = now_ts
        state["regime"] = reg_state
        state["regime_ok"] = False
        state["regime_last_msg"] = reason
        return False, f"REGIME_BLOCK:{reason}"

    # cooldown after a block to avoid flip-flop regime
    last_block = int(reg_state.get("last_block_ts", 0) or 0)
    if last_block and (now_ts - last_block) < int(REGIME_COOLDOWN_SECONDS):
        reg_state["ok"] = True
        reg_state["label"] = "ALLOW"
        reg_state["reason"] = f"COOLDOWN({now_ts - last_block}s)"
        reg_state["score"] = score
        state["regime"] = reg_state
        state["regime_ok"] = True
        state["regime_last_msg"] = reg_state["reason"]
        return False, "REGIME_COOLDOWN"

    reg_state["ok"] = True
    reg_state["label"] = "ALLOW"
    reg_state["reason"] = reason
    reg_state["score"] = score
    state["regime"] = reg_state
    state["regime_ok"] = True
    state["regime_last_msg"] = reason
    return True, "REGIME_OK"


def update_candidate(state: Dict[str, Any], symbol: str, score: float, now_ts: int) -> Tuple[bool, str]:
    cands = state.get("candidates", {}) or {}
    c = cands.get(symbol)
    if not isinstance(c, dict):
        c = {"streak": 0, "best": 0.0, "first_ts": now_ts, "last_ts": now_ts}
        cands[symbol] = c

    if (now_ts - int(c.get("last_ts", now_ts))) > int(CANDIDATE_TTL_SECONDS):
        c["streak"] = 0
        c["best"] = 0.0
        c["first_ts"] = now_ts

    c["last_ts"] = now_ts
    c["best"] = max(float(c.get("best", 0.0)), float(score))

    enter_th = float(ENTER_SCORE) + float(state.get('score_bump', 0.0) or 0.0)
    exit_th = float(EXIT_SCORE)  # keep hysteresis stable

    if float(score) >= enter_th:
        c["streak"] = int(c.get("streak", 0)) + 1
    elif float(score) <= exit_th:
        c["streak"] = 0
        c["best"] = float(score)

    cands[symbol] = c
    state["candidates"] = cands

    if int(c.get("streak", 0)) >= int(PERSIST_TICKS_REQUIRED):
        # consume to prevent repeated triggers
        try:
            del cands[symbol]
        except Exception:
            pass
        state["candidates"] = cands
        return True, "PERSISTENCE_OK"

    return False, "PERSISTING"


def classify_loss_reason(trade: Dict[str, Any]) -> str:
    reason = str(trade.get("reason", ""))
    if "SL_" in reason:
        # attempt to classify
        if trade.get("spread_pct") is not None and float(trade["spread_pct"]) > float(MAX_SPREAD_PCT):
            return "low_liquidity_slippage"
        reg_reason = str(trade.get("regime_reason", ""))
        if "CHOP" in reg_reason or "CHOP" in str(trade.get("regime_label", "")):
            return "chop"
        if "VOL_SPIKE" in reg_reason:
            return "volatility_spike"
        return "stop_loss"
    if "TRAIL" in reason:
        return "trail_giveback"
    if "BE" in reason:
        return "break_even"
    if "STAG" in reason or "EOG" in reason:
        return "stagnation"
    return "other"

def build_symbol_lists(exchanges: Dict[str, Any]) -> Tuple[Dict[str, List[str]], List[str]]:
    by_ex: Dict[str, List[str]] = {}
    union = set()

    for ex_name, ex in exchanges.items():
        try:
            markets = ex.load_markets(False)
        except Exception as e:
            LOG.info(f"[UNIVERSE] {ex_name} load_markets failed: {type(e).__name__} {e}")
            by_ex[ex_name] = []
            continue

        symbols: List[str] = []
        for sym, m in markets.items():
            if m.get("spot") is False:
                continue
            if _is_good_symbol(sym):
                symbols.append(sym)

        symbols = sorted(set(symbols))
        by_ex[ex_name] = symbols
        union.update(symbols)
        LOG.info(f"[UNIVERSE] {ex_name}: {len(symbols)} symbols")

    union_list = sorted(union)
    LOG.info(f"[UNIVERSE] union total: {len(union_list)} symbols")
    return by_ex, union_list


def rank_symbols_by_volume(
    exchanges: Dict[str, Any],
    by_ex: Dict[str, List[str]],
    union_list: List[str],) -> List[str]:
    volumes: Dict[str, float] = {}

    for ex_name, ex in exchanges.items():
        syms = by_ex.get(ex_name, [])
        if not syms:
            continue

        tickers = {}
        try:
            tickers = ex.fetch_tickers(syms)
        except Exception:
            for s in syms:
                try:
                    tickers[s] = ex.fetch_ticker(s)
                except Exception:
                    continue

        for s, t in (tickers or {}).items():
            v = _quote_volume_usdish(t)
            if v <= 0:
                continue
            prev = volumes.get(s, 0.0)
            if v > prev:
                volumes[s] = v

    ranked = sorted(union_list, key=lambda s: volumes.get(s, 0.0), reverse=True)
    return ranked


def build_universe_with_volume(exchanges: Dict[str, Any]) -> Tuple[Dict[str, List[str]], List[str]]:
    cached = load_json(UNIVERSE_FILE, None)
    if isinstance(cached, dict):
        if int(cached.get("version", 0)) == int(UNIVERSE_CACHE_VERSION):
            ts = cached.get("ts", 0)
            if time.time() - ts < UNIVERSE_REFRESH_SECONDS:
                by_ex = cached.get("by_exchange", {})
                ranked = cached.get("ranked_union", [])
                if isinstance(by_ex, dict) and isinstance(ranked, list) and ranked:
                    LOG.info(f"[UNIVERSE] loaded cache v{UNIVERSE_CACHE_VERSION} ({len(ranked)} ranked symbols)")
                    return by_ex, ranked

    by_ex, union_list = build_symbol_lists(exchanges)
    ranked = rank_symbols_by_volume(exchanges, by_ex, union_list)

    save_json(
        UNIVERSE_FILE,
        {
            "version": UNIVERSE_CACHE_VERSION,
            "ts": time.time(),
            "by_exchange": by_ex,
            "ranked_union": ranked,
        },
    )
    LOG.info(f"[UNIVERSE] ranked ready: {len(ranked)} symbols")
    return by_ex, ranked


def batched_symbols(symbols: List[str], size: int, batch_index: int) -> List[str]:
    if not symbols:
        return []
    n = len(symbols)
    start = (batch_index * size) % n
    end = start + size
    if end <= n:
        return symbols[start:end]
    return symbols[start:] + symbols[: (end - n)]


# ============================================================
# HEARTBEAT (REALIZED/UNREAL + OPEN TRADES)
# ============================================================
def build_heartbeat_lines(
    state: Dict[str, Any],
    controls: Dict[str, Any],
    batch_size: int,
    now: int,) -> List[str]:
    pos = state.get("positions", {}) or {}

    unreal_total = 0.0
    for p in pos.values():
        try:
            unreal_total += float(p.get("unreal", 0.0))
        except Exception:
            pass

    prof = ensure_active_profile(state)

    lines: List[str] = []
    lines.append(
        f"[RIFT] tick={now} | profile={prof} | positions={len(pos)}/{MAX_OPEN_POSITIONS} | "
        f"pause={controls.get('pause_entries', False)} | batch_size={batch_size} | "
        f"equity=${state.get('equity', 0.0):,.2f} | "
        f"realized=${state.get('realized_pnl', 0.0):,.2f} | "
        f"unreal=${unreal_total:+,.2f}"
    )

    if pos:
        be_trigger, trail_trigger, trail_giveback = get_be_trail_params(state)
        lines.append(
            f"[RIFT] BE/TRAIL: be@{be_trigger:+.2f} exit@{BE_EXIT_UNREAL_DOLLARS:+.2f} | "
            f"trail@{trail_trigger:+.2f} giveback={trail_giveback:.2f} | one_base={ONE_BASE_ASSET_AT_A_TIME}"
        )
        lines.append("[RIFT] open trades:")
        items = sorted(pos.items(), key=lambda kv: int(kv[1].get("opened_ts", 0)))
        for _, p in items:
            entry = float(p.get("entry", 0.0))
            last = float(p.get("last", entry))
            qty = float(p.get("qty", 0.0))
            unreal = float(p.get("unreal", 0.0))
            eog = "EOG" if p.get("eog_armed", False) else "-"
            be = "BE" if p.get("be_armed", False) else "-"
            tr = "TR" if p.get("trail_active", False) else "-"
            peak = float(p.get("peak_unreal", 0.0)) if p.get("trail_active", False) else 0.0
            stop = float(p.get("trail_stop_unreal", 0.0)) if p.get("trail_active", False) else 0.0

            extra = f" be={be} tr={tr} eog={eog}"
            if p.get("trail_active", False):
                extra += f" peak={peak:+.2f} stop={stop:+.2f}"

            lines.append(
                f"  - {p.get('exchange')} {p.get('symbol')} | "
                f"entry={entry:.4f} last={last:.4f} qty={qty:.6f} unreal={unreal:+.2f}{extra}"
            )
    else:
        lines.append("[RIFT] (no open positions)")

    return lines


# ============================================================
# TRADING (PAPER)
# ============================================================
def fetch_last(ex, symbol: str) -> float:
    t = ex.fetch_ticker(symbol)
    last = t.get("last")
    if last is None:
        raise RuntimeError(f"ticker.last None for {symbol}")
    return float(last)



def close_position(state: Dict[str, Any], pid: str, exit_price: float, reason: str) -> None:
    p = state["positions"][pid]
    entry = float(p["entry"])
    qty = float(p["qty"])
    symbol = p["symbol"]
    now_ts = int(state.get("_now", int(time.time())))

    fee_rate = float(p.get("fee_rate", DEFAULT_TAKER_FEE_RATE))
    gross_pnl = (float(exit_price) - entry) * qty

    fee_open = float(p.get("fee_open", 0.0)) if INCLUDE_FEES_IN_PNL else 0.0
    fee_exit = (float(exit_price) * qty * fee_rate) if INCLUDE_FEES_IN_PNL else 0.0
    net_pnl = gross_pnl - fee_open - fee_exit

    state["equity"] += net_pnl
    state["realized_pnl"] += net_pnl

    state["cooldowns"][symbol] = now_ts + int(COOLDOWN_SECONDS)

    LOG.info(
        f"[CLOSE] {p['exchange']} {symbol} entry={entry:.6f} exit={float(exit_price):.6f} "
        f"qty={qty:.6f} pnl={net_pnl:+.2f} gross={gross_pnl:+.2f} fees={fee_open+fee_exit:+.2f} reason={reason}"
    )

    close_evt = {
        "evt": "CLOSE",
        "ts": now_ts,
        "exchange": p.get("exchange"),
        "symbol": symbol,
        "entry": entry,
        "exit": float(exit_price),
        "qty": qty,
        "pnl": net_pnl,
        "gross_pnl": gross_pnl,
        "fee_open": fee_open,
        "fee_exit": fee_exit,
        "fee_rate": fee_rate,
        "reason": reason,
        "hold_s": now_ts - int(p.get("opened_ts", now_ts)),
        "score": p.get("entry_score"),
        "trend_score": p.get("entry_trend_score"),
        "atr_ratio": p.get("entry_atr_ratio"),
        "rsi": p.get("entry_rsi"),
        "spread_pct": p.get("entry_spread_pct"),
        "qv": p.get("entry_qv"),
        "regime_label": p.get("regime_label"),
        "regime_reason": p.get("regime_reason"),
    }
    append_trade_log(close_evt)

    if net_pnl < 0:
        loss_reason = classify_loss_reason({**close_evt, "reason": reason})
        clusters = state.get("loss_clusters", {}) or {}
        clusters[loss_reason] = int(clusters.get(loss_reason, 0)) + 1
        state["loss_clusters"] = clusters

    del state["positions"][pid]



def close_all_positions(state: Dict[str, Any], exchanges: Dict[str, Any], reason: str) -> None:
    if not state["positions"]:
        LOG.info("[RIFT] close_all: no open positions")
        return
    LOG.info("[RIFT] close_all: closing all open positions...")
    for pid in list(state["positions"].keys()):
        p = state["positions"][pid]
        ex = exchanges.get(p["exchange"])
        if not ex:
            continue
        try:
            last = fetch_last(ex, p["symbol"])
            close_position(state, pid, last, reason)
        except Exception as e:
            LOG.info(f"[WARN] close_all failed {p['exchange']} {p['symbol']}: {type(e).__name__} {e}")
    save_state(state)
    LOG.info("[RIFT] close_all: done")


def trend_ok(ex, symbol: str) -> bool:
    candles = ex.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=CANDLE_LIMIT)
    closes = [float(c[4]) for c in candles if c and len(c) >= 5]

    if len(closes) < max(EMA_SLOW + 5, RSI_PERIOD + 5):
        return False

    # ---- PEG GUARD (filters USD-pegged behavior) ----
    if PEG_GUARD_ENABLED:
        atr = atr_last(candles, ATR_PERIOD)
        if atr is not None and closes[-1] > 0:
            atr_pct = atr / closes[-1]
            if PEG_PRICE_LOW <= closes[-1] <= PEG_PRICE_HIGH and atr_pct <= PEG_ATR_PCT_MAX:
                return False

    last = closes[-1]
    ef = ema_last(closes, EMA_FAST)
    es = ema_last(closes, EMA_SLOW)
    r = rsi_last(closes, RSI_PERIOD)

    if ef is None or es is None or r is None:
        return False

    return (ef > es) and (RSI_MIN <= r <= RSI_MAX) and (last > es)


# ============================================================
# ENGINE LOOP (async wrapper around sync logic)
# ============================================================
class RiftEngine:
    def __init__(self):
        self.running = False

    async def run(self):
        global _shutdown_signal
        if self.running:
            LOG.info("[RIFT] engine already running")
            return

        self.running = True
        LOG.info("[RIFT] ENGINE STARTING (master ogrift.py) ✅")
        LOG.info(
            f"[RIFT] tf={TIMEFRAME} tick={SCAN_INTERVAL}s | hold={MAX_HOLD_SECONDS/60:.0f}m rec={RECOVERY_WINDOW_SECONDS/60:.0f}m"
        )
        LOG.info(
            f"[RIFT] max_pos={MAX_OPEN_POSITIONS} | profile auto: SMALL<=${STANDARD_TO_SMALL_EQUITY:.0f} "
            f"STANDARD>=${SMALL_TO_STANDARD_EQUITY:.0f}"
        )

        state = load_state()
        normalize_cooldowns(state)
        ensure_active_profile(state)
        save_state(state)

        # Force SPOT-only mode & avoid heavy market discovery
        exchanges: Dict[str, Any] = {
            n: getattr(ccxt, n)(
                {
                    "enableRateLimit": True,
                    "options": {"defaultType": "spot", "loadAllOptions": False},
                }
            )
            for n in EXCHANGE_NAMES
        }

        try:
            by_ex, ranked_symbols = build_universe_with_volume(exchanges)
            by_ex_sets = {k: set(v) for k, v in by_ex.items()}

            batch_i = 0
            last_universe_refresh = time.time()
            last_heartbeat = 0

            while not _shutdown_signal:
                now = int(time.time())
                state['_now'] = now
                controls = load_controls()

                batch_size = int(controls.get("batch_size", BATCH_SIZE_DEFAULT))
                batch_size = max(5, min(batch_size, 200))

                if now - last_heartbeat >= HEARTBEAT_SECONDS or controls.get("print_positions_now", False):
                    last_heartbeat = now
                    for line in build_heartbeat_lines(state, controls, batch_size, now):
                        LOG.info(line)

                    if controls.get("print_positions_now", False):
                        controls["print_positions_now"] = False
                        save_controls(controls)

                if time.time() - last_universe_refresh >= UNIVERSE_REFRESH_SECONDS:
                    by_ex, ranked_symbols = build_universe_with_volume(exchanges)
                    by_ex_sets = {k: set(v) for k, v in by_ex.items()}
                    last_universe_refresh = time.time()

                if controls.get("shutdown", False) or controls.get("restart", False):
                    if not controls.get("close_all", False):
                        controls["close_all"] = True
                        save_controls(controls)

                if controls.get("close_all", False):
                    close_all_positions(state, exchanges, reason="MANUAL_CLOSE_ALL")
                    controls = load_controls()
                    controls["close_all"] = False
                    save_controls(controls)

                controls = load_controls()
                if controls.get("restart", False):
                    LOG.info("[RIFT] restart flag -> resetting scan cycle (positions already closed)")
                    controls["restart"] = False
                    save_controls(controls)
                    batch_i = 0
                    last_universe_refresh = 0
                    continue

                controls = load_controls()
                if controls.get("shutdown", False):
                    LOG.info("[RIFT] shutdown flag -> stopping engine")
                    controls["shutdown"] = False
                    save_controls(controls)
                    break

                # ----------------------------
                # EXITS (dynamic TP/SL + BE + TRAILING + STAGNATION + EXIT-ON-GREEN)
                # ----------------------------
                tp_dollars, sl_dollars, prof = get_tp_sl_dollars(state)
                be_trigger, trail_trigger, trail_giveback = get_be_trail_params(state)

                for pid in list(state["positions"].keys()):
                    p = state["positions"][pid]
                    ex = exchanges.get(p["exchange"])
                    if not ex:
                        continue

                    try:
                        last = fetch_last(ex, p["symbol"])
                        fee_rate = float(p.get("fee_rate", get_taker_fee_rate(ex, p["symbol"])))
                        unreal_gross = (last - float(p["entry"])) * float(p["qty"])
                        unreal = calc_net_unreal(p, last, fee_rate)

                        # seed/normalize fields
                        p["last"] = last
                        p["unreal_gross"] = unreal_gross
                        p["unreal"] = unreal
                        p["fee_rate"] = fee_rate
                        # --------
                        # PEAK TRACKING (always, even before trailing)
                        # --------
                        prev_peak = float(p.get("peak_unreal", unreal))
                        if unreal > prev_peak:
                             p["peak_unreal"] = unreal
                        else:
                            p["peak_unreal"] = prev_peak

                        if "be_armed" not in p:
                            p["be_armed"] = False
                        if "trail_active" not in p:
                            p["trail_active"] = False
                        if "peak_unreal" not in p:
                            p["peak_unreal"] = unreal
                        if "trail_stop_unreal" not in p:
                            p["trail_stop_unreal"] = 0.0

                        # Hard TP/SL first (profile-aware)
                        if unreal >= tp_dollars:
                            close_position(state, pid, last, f"TP_{prof}")
                            continue
                        if unreal <= sl_dollars:
                            close_position(state, pid, last, f"SL_{prof}")
                            continue

                        # --------
                        # BREAK-EVEN (BE)
                        # --------
                        if (not p.get("be_armed", False)):
                            # Only arm BE after: min hold time AND meaningful peak AND meaningful profit
                            age_s = now - int(p.get("opened_ts", now))
                            peak = float(p.get("peak_unreal", unreal))
                            if age_s >= int(BE_MIN_HOLD_SECONDS) and peak >= float(BE_MIN_PEAK_USD) and unreal >= max(float(be_trigger), float(BE_ARM_AT_PROFIT_USD)):
                                p["be_armed"] = True
                                p["be_armed_ts"] = now
                                LOG.info(
                                    f"[RIFT] BE ARMED {p['exchange']} {p['symbol']} unreal={unreal:+.2f} peak={peak:+.2f} "
                                    f"(exit if <= {float(BE_EXIT_UNREAL_DOLLARS):+.2f})"
                                )
                        # --------
                        # POST-BE PROFIT LOCK (prevents BE -> back to $0 donation)
                        # --------
                        if POST_BE_LOCK_ENABLED and p.get("be_armed", False) and not p.get("trail_active", False):
                            post_be_giveback = max(
                                tp_dollars * POST_BE_GIVEBACK_TP_FRACTION,
                                MIN_POST_BE_GIVEBACK_DOLLARS,
                            )
                            peak = float(p.get("peak_unreal", unreal))
                            # Gate profit lock so it cannot fire on tiny/no peak
                            if peak >= float(BE_MIN_PEAK_USD) and peak >= float(be_trigger):
                                if unreal <= (peak - post_be_giveback) and unreal > 0:
                                    close_position(state, pid, last, "POST_BE_PROFIT_LOCK_EXIT")
                                    continue

                        if p.get("be_armed", False) and unreal <= float(BE_EXIT_UNREAL_DOLLARS):
                            close_position(state, pid, last, "BE_EXIT")
                            continue
                        
                        # --------
                        # TRAILING (ATR + Momentum Aware)
                        # --------
                        if unreal >= trail_trigger:
                            age_s = now - int(p.get('opened_ts', now))
                            peak = float(p.get('peak_unreal', unreal))
                            if age_s < int(TRAIL_MIN_HOLD_SECONDS) or peak < float(TRAIL_MIN_PEAK_USD):
                                # too early / too small to trail
                                pass
                            else:
                                candles = ex.fetch_ohlcv(p["symbol"], timeframe=TIMEFRAME, limit=ATR_PERIOD + 20)
                                closes = [float(c[4]) for c in candles]
                                volumes = [float(c[5]) for c in candles]

                                atr = atr_last(candles, ATR_PERIOD)
                                ef = ema_last(closes, EMA_FAST)
                                es = ema_last(closes, EMA_SLOW)
                                rsi = rsi_last(closes, RSI_PERIOD)

                                weak = momentum_weak(closes, volumes, ef, es, rsi)

                                atr_mult = ATR_MULT_TIGHT if weak else ATR_MULT_NORMAL
                                dynamic_giveback = max(
                                    atr * atr_mult * p["qty"],
                                    trail_giveback
                                )

                                if not p.get("trail_active", False):
                                    p["trail_active"] = True
                                    p["peak_unreal"] = unreal
                                    p["trail_stop_unreal"] = unreal - dynamic_giveback
                                    LOG.info(
                                        f"[RIFT] TRAIL ON {p['exchange']} {p['symbol']} "
                                        f"unreal={unreal:+.2f} stop={p['trail_stop_unreal']:+.2f} weak={weak}"
                                    )
                                else:
                                    if unreal > p["peak_unreal"]:
                                        p["peak_unreal"] = unreal
                                        new_stop = unreal - dynamic_giveback
                                        p["trail_stop_unreal"] = max(p["trail_stop_unreal"], new_stop)

                                if unreal <= p["trail_stop_unreal"]:
                                    close_position(state, pid, last, "TRAIL_EXIT_DYNAMIC")
                                    continue
                                atr_giveback = (atr * atr_mult * p["qty"])

                                if weak:
                                    dynamic_giveback = max(MIN_TRAIL_GIVEBACK_DOLLARS, min(trail_giveback, atr_giveback))
                                else:
                                    dynamic_giveback = max(trail_giveback, atr_giveback)

                            # --------
                        # Exit-on-green armed state (stagnation system)
                        # --------
                        if p.get("eog_armed", False):
                            if unreal > EXIT_ON_GREEN_MIN_UNREAL:
                                close_position(state, pid, last, "EOG_GREEN_EXIT")
                                continue

                            max_wait = int(EXIT_ON_GREEN_MAX_WAIT_SECONDS or 0)
                            if max_wait > 0:
                                armed_ts = int(p.get("eog_armed_ts", now))
                                if now - armed_ts >= max_wait:
                                    close_position(state, pid, last, "EOG_MAX_WAIT_FORCED_EXIT")
                                    continue

                        opened_ts = int(p.get("opened_ts", now))
                        age = now - opened_ts

                        if age >= MAX_HOLD_SECONDS:
                            if unreal > EXIT_ON_GREEN_MIN_UNREAL:
                                close_position(state, pid, last, "STAG_PROFIT_EXIT")
                                continue

                            deadline = int(p.get("recovery_deadline", 0))
                            if deadline <= 0:
                                p["recovery_deadline"] = now + RECOVERY_WINDOW_SECONDS
                                LOG.info(
                                    f"[RIFT] recovery window started {p['exchange']} {p['symbol']} "
                                    f"age_min={age/60:.1f} unreal={unreal:+.2f}"
                                )
                            else:
                                if unreal > EXIT_ON_GREEN_MIN_UNREAL:
                                    close_position(state, pid, last, "RECOVERY_EXIT_ON_GREEN")
                                    continue

                                if now >= deadline:
                                    if EXIT_ON_GREEN_AFTER_STAG_TIMEOUT:
                                        if not p.get("eog_armed", False):
                                            p["eog_armed"] = True
                                            p["eog_armed_ts"] = now
                                            LOG.info(
                                                f"[RIFT] exit-on-green ARMED {p['exchange']} {p['symbol']} "
                                                f"unreal={unreal:+.2f} (waiting for > {EXIT_ON_GREEN_MIN_UNREAL:+.2f})"
                                            )

                                        if unreal > EXIT_ON_GREEN_MIN_UNREAL:
                                            close_position(state, pid, last, "EOG_GREEN_EXIT")
                                            continue

                                        max_wait = int(EXIT_ON_GREEN_MAX_WAIT_SECONDS or 0)
                                        if max_wait > 0:
                                            armed_ts = int(p.get("eog_armed_ts", now))
                                            if now - armed_ts >= max_wait:
                                                close_position(state, pid, last, "EOG_MAX_WAIT_FORCED_EXIT")
                                                continue
                                    else:
                                        close_position(state, pid, last, "STAG_TIMEOUT_EXIT")
                                        continue

                    except Exception as e:
                        LOG.info(f"[WARN] exit-check {p.get('exchange')} {p.get('symbol')}: {type(e).__name__} {e}")

                # ----------------------------
                # ENTRIES (volume-ranked, profile-aware sizing, base-asset diversity)
                # ----------------------------
                controls = load_controls()
                if not controls.get("pause_entries", False):
                    open_symbols = {pos["symbol"] for pos in state["positions"].values()}
                    open_bases = {base_asset(pos["symbol"]) for pos in state["positions"].values()}

                    scan = batched_symbols(ranked_symbols, batch_size, batch_i)
                    batch_i += 1

                    update_equity_guard(state, now)
                    position_value, prof_now = get_position_value(state)

                    # Governor: performance-aware gating (cold bot stands down)
                    perf_ok, perf_msg = perf_allows_entries(state, now)
                    if not perf_ok:
                        # surface state for telegram + logs
                        state['regime_ok'] = False
                        state['regime_last_msg'] = perf_msg
                        LOG.info(f"[RIFT] entry stand-down: {perf_msg}")
                    else:
                        state['regime_last_msg'] = ''

                    for symbol in scan:
                        if len(state["positions"]) >= MAX_OPEN_POSITIONS:
                            break

                        if symbol in open_symbols:
                            continue

                        if int(state["cooldowns"].get(symbol, 0)) > now:
                            continue

                        # Diversity rule: one base asset at a time (prevents PAXG/USD + PAXG/USDC, etc.)
                        if ONE_BASE_ASSET_AT_A_TIME:
                            b = base_asset(symbol)
                            if b and b in open_bases:
                                continue

                        for ex_name, ex in exchanges.items():
                            if symbol not in by_ex_sets.get(ex_name, set()):
                                continue
                            try:
                                # --- ticker (spread/liquidity/price sanity) ---
                                t = ex.fetch_ticker(symbol)
                                last = t.get("last")
                                if last is None:
                                    continue
                                last = float(last)
                                qv = _quote_volume_usdish(t)
                                ok_price, why_price = pass_price_sanity(symbol, last, qv)
                                if not ok_price:
                                    continue

                                sp = spread_pct_from_ticker(t)
                                if sp is not None and float(sp) > float(MAX_SPREAD_PCT):
                                    continue

                                # Notional sanity
                                if float(position_value) < float(MIN_NOTIONAL_USD):
                                    continue

                                # Governor: perf stand-down must pass
                                if not perf_ok:
                                    continue

                                # --- compute features, score, regime ---
                                feat = compute_features(ex, symbol)
                                if not feat:
                                    continue
                                if feat.get("peg_block", False):
                                    continue

                                regime_ok, regime_msg = regime_allows_entry(state, now, feat.get("regime", {}))
                                if not regime_ok:
                                    continue

                                score = float(feat.get("score", 0.0))
                                persist_ok, persist_msg = update_candidate(state, symbol, score, now)
                                if not persist_ok:
                                    continue

                                qty_raw = float(position_value) / last
                                fee_rate = get_taker_fee_rate(ex, symbol)

                                qty, qty_msg = normalize_order_qty(ex, symbol, qty_raw, last)
                                if qty is None or qty <= 0:
                                    continue

                                fee_open = (float(qty) * float(last) * float(fee_rate)) if INCLUDE_FEES_IN_PNL else 0.0

                                pid = f"{ex_name}:{symbol}:{now}"
                                state["positions"][pid] = {
                                    "exchange": ex_name,
                                    "symbol": symbol,
                                    "entry": last,
                                    "qty": qty,
                                    "fee_rate": float(fee_rate),
                                    "fee_open": float(fee_open),
                                    "opened_ts": now,
                                    "recovery_deadline": 0,

                                    # Exit-on-green state
                                    "eog_armed": False,
                                    "eog_armed_ts": 0,

                                    # BE + trailing state
                                    "be_armed": False,
                                    "be_armed_ts": 0,
                                    "trail_active": False,
                                    "peak_unreal": 0.0,
                                    "trail_stop_unreal": 0.0,

                                    # heartbeat fields
                                    "last": last,
                                    "unreal": 0.0,

                                    # forensic snapshot
                                    "entry_score": score,
                                    "entry_trend_score": float(feat.get("trend_score", 0.0)),
                                    "entry_atr_ratio": float(feat.get("atr_ratio", 1.0)),
                                    "entry_rsi": float(feat.get("rsi", 0.0)),
                                    "entry_spread_pct": float(sp) if sp is not None else None,
                                    "entry_qv": float(qv),
                                    "regime_label": str((feat.get("regime") or {}).get("label", "")),
                                    "regime_reason": str((feat.get("regime") or {}).get("reason", "")),
                                }

                                append_trade_log({
                                    "evt": "OPEN",
                                    "exchange": ex_name,
                                    "symbol": symbol,
                                    "entry": last,
                                    "qty": qty,
                                    "fee_rate": float(fee_rate),
                                    "fee_open": float(fee_open),
                                    "score": score,
                                    "trend_score": float(feat.get("trend_score", 0.0)),
                                    "atr_ratio": float(feat.get("atr_ratio", 1.0)),
                                    "rsi": float(feat.get("rsi", 0.0)),
                                    "spread_pct": float(sp) if sp is not None else None,
                                    "qv": float(qv),
                                    "regime_label": str((feat.get("regime") or {}).get("label", "")),
                                    "regime_reason": str((feat.get("regime") or {}).get("reason", "")),
                                })

                                LOG.info(f"[OPEN] {ex_name} {symbol} entry={last:.6f} qty={qty:.6f} score={score:.2f} profile={prof_now}")
                                open_symbols.add(symbol)
                                if ONE_BASE_ASSET_AT_A_TIME:
                                    open_bases.add(base_asset(symbol))
                                break
                            except Exception:
                                continue


                save_state(state)
                await asyncio.sleep(SCAN_INTERVAL)

        finally:
            # Clean shutdown of CCXT exchanges (prevents unclosed connector warnings)
            for ex in (exchanges or {}).values():
                try:
                    await ex.close()
                except Exception:
                    pass

            self.running = False
            LOG.info("[RIFT] ENGINE STOPPED CLEANLY")


ENGINE = RiftEngine()


# ============================================================
# TELEGRAM UI
# ============================================================
def keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("▶️ Start", callback_data="start"),
                InlineKeyboardButton("⏸ Pause", callback_data="pause"),
                InlineKeyboardButton("▶️ Resume", callback_data="resume"),
            ],
            [
                InlineKeyboardButton("📌 Positions", callback_data="positions"),
                InlineKeyboardButton("💰 Equity", callback_data="equity"),
            ],
            [
                InlineKeyboardButton("🧹 Close All", callback_data="closeall"),
                InlineKeyboardButton("🔄 Restart", callback_data="restart"),
            ],
            [
                InlineKeyboardButton("⛔ Shutdown", callback_data="shutdown"),
                InlineKeyboardButton("❓ Help", callback_data="help"),
            ],
        ]
    )


def render_equity_text() -> str:
    """
    Single source of truth for the equity/status line,
    used by both /equity and the Equity button.
    """
    s = load_state()
    pos = s.get("positions", {}) or {}

    unreal_total = 0.0
    for p in pos.values():
        try:
            unreal_total += float(p.get("unreal", 0.0))
        except Exception:
            pass

    prof = ensure_active_profile(s)
    tp_d, sl_d, _ = get_tp_sl_dollars(s)
    be_trigger, trail_trigger, trail_giveback = get_be_trail_params(s)

    # Optional regime fields (safe if never set elsewhere)
    regime_ok = bool(s.get("regime_ok", True))
    regime_msg = str(s.get("regime_last_msg", "") or "")

    # Performance gating visibility
    perf = s.get("perf", {}) or {}
    cold_until = int(perf.get("cold_until", 0) or 0)
    cold_active = int(time.time()) < cold_until
    metrics = perf.get("last_metrics", {}) or {}
    pf_txt = ""
    if metrics.get("n", 0) > 0:
        pf_txt = (
            f" | win={metrics.get('win_rate', 0.0)*100:.0f}%"
            f" avgW={metrics.get('avg_win', 0.0):.2f}"
            f" avgL={metrics.get('avg_loss', 0.0):.2f}"
            f" exp={metrics.get('expectancy', 0.0):+.2f}"
            f" dd={metrics.get('max_dd_abs', 0.0):.2f}"
        )

    clusters = s.get("loss_clusters", {}) or {}
    top_losses = sorted(clusters.items(), key=lambda kv: int(kv[1]), reverse=True)[:3]
    loss_txt = ""
    if top_losses:
        loss_txt = " | loss_clusters=" + ",".join([f"{k}:{v}" for k, v in top_losses])


    txt = (
        f"profile={prof} | equity=${s.get('equity', 0):,.2f} | realized=${s.get('realized_pnl', 0):,.2f} | "
        f"unreal=${unreal_total:+,.2f} | tp=${tp_d:+.2f} sl=${sl_d:+.2f} | "
        f"BE@{be_trigger:+.2f} TRAIL@{trail_trigger:+.2f} giveback={trail_giveback:.2f} | "
        f"one_base={ONE_BASE_ASSET_AT_A_TIME} | regime={'OK' if regime_ok else 'BAD'}"
        f" | perf={'COLD' if cold_active else 'OK'}"
        f"{pf_txt}{loss_txt}"
    )
    if regime_msg:
        txt += f"\nREGIME: {regime_msg}"

    return txt


def fmt_positions() -> str:
    s = load_state()
    pos = s.get("positions", {}) or {}

    unreal_total = 0.0
    for p in pos.values():
        try:
            unreal_total += float(p.get("unreal", 0.0))
        except Exception:
            pass

    prof = ensure_active_profile(s)
    tp_d, sl_d, _ = get_tp_sl_dollars(s)
    be_trigger, trail_trigger, trail_giveback = get_be_trail_params(s)

    lines: List[str] = []
    lines.append(
        f"profile={prof} | equity=${s.get('equity', 0):,.2f} | realized=${s.get('realized_pnl', 0):,.2f} | "
        f"unreal=${unreal_total:+,.2f} | tp=${tp_d:+.2f} sl=${sl_d:+.2f}"
    )
    lines.append(
        f"BE@{be_trigger:+.2f} exit@{BE_EXIT_UNREAL_DOLLARS:+.2f} | "
        f"TRAIL@{trail_trigger:+.2f} giveback={trail_giveback:.2f} | one_base={ONE_BASE_ASSET_AT_A_TIME}"
    )
    lines.append(f"open_positions={len(pos)}/{MAX_OPEN_POSITIONS}")

    for p in pos.values():
        entry = float(p.get("entry", 0.0))
        last = float(p.get("last", entry))
        qty = float(p.get("qty", 0.0))
        unreal = float(p.get("unreal", 0.0))

        eog = "EOG" if p.get("eog_armed", False) else "-"
        be = "BE" if p.get("be_armed", False) else "-"
        tr = "TR" if p.get("trail_active", False) else "-"
        base = base_asset(p.get("symbol", ""))

        extra = f" base={base} be={be} tr={tr} eog={eog}"
        if p.get("trail_active", False):
            peak = float(p.get("peak_unreal", 0.0))
            stop = float(p.get("trail_stop_unreal", 0.0))
            extra += f" peak={peak:+.2f} stop={stop:+.2f}"

        lines.append(
            f"- {p['exchange']} {p['symbol']} entry={entry:.4f} last={last:.4f} qty={qty:.6f} unreal={unreal:+.2f}{extra}"
        )

    return "\n".join(lines)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "/help\n"
        "/startbot  (start engine)\n"
        "/pause\n"
        "/resume\n"
        "/positions\n"
        "/equity\n"
        "/batchsize <n>\n"
        "/closeall\n"
        "/restartbot  (close all + continue)\n"
        "/shutdownbot (close all + stop)\n"
    )
    await update.message.reply_text(msg, reply_markup=keyboard())


async def cmd_startbot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not ENGINE.running:
        asyncio.create_task(ENGINE.run())
    await update.message.reply_text("Started ✅", reply_markup=keyboard())


async def cmd_pause(update: Update, context: ContextTypes.DEFAULT_TYPE):
    c = load_controls()
    c["pause_entries"] = True
    save_controls(c)
    await update.message.reply_text("Entries paused ⏸", reply_markup=keyboard())


async def cmd_resume(update: Update, context: ContextTypes.DEFAULT_TYPE):
    c = load_controls()
    c["pause_entries"] = False
    save_controls(c)
    await update.message.reply_text("Entries resumed ▶️", reply_markup=keyboard())


async def cmd_positions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    c = load_controls()
    c["print_positions_now"] = True
    save_controls(c)
    await update.message.reply_text(fmt_positions(), reply_markup=keyboard())


async def cmd_equity(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(render_equity_text(), reply_markup=keyboard())


async def cmd_batchsize(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            f"batch_size={load_controls().get('batch_size')}", reply_markup=keyboard()
        )
        return
    try:
        n = int(context.args[0])
        n = max(5, min(n, 200))
    except Exception:
        await update.message.reply_text("Usage: /batchsize 30", reply_markup=keyboard())
        return
    c = load_controls()
    c["batch_size"] = n
    save_controls(c)
    await update.message.reply_text(f"batch_size set to {n} ✅", reply_markup=keyboard())


async def cmd_closeall(update: Update, context: ContextTypes.DEFAULT_TYPE):
    c = load_controls()
    c["close_all"] = True
    save_controls(c)
    await update.message.reply_text("Close-all requested 🧹", reply_markup=keyboard())


async def cmd_restartbot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    c = load_controls()
    c["restart"] = True
    save_controls(c)
    await update.message.reply_text("Restart requested 🔄 (close all + continue)", reply_markup=keyboard())


async def cmd_shutdownbot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    c = load_controls()
    c["shutdown"] = True
    save_controls(c)
    await update.message.reply_text("Shutdown requested ⛔ (close all + stop)", reply_markup=keyboard())


async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()

    data = q.data
    if data == "help":
        await q.edit_message_text(
            "Commands:\n/help /startbot /pause /resume /positions /equity /batchsize n "
            "/closeall /restartbot /shutdownbot",
            reply_markup=keyboard(),
        )
        return

    if data == "start":
        if not ENGINE.running:
            asyncio.create_task(ENGINE.run())
        await q.edit_message_text("Started ✅", reply_markup=keyboard())
        return

    if data == "pause":
        c = load_controls()
        c["pause_entries"] = True
        save_controls(c)
        await q.edit_message_text("Entries paused ⏸", reply_markup=keyboard())
        return

    if data == "resume":
        c = load_controls()
        c["pause_entries"] = False
        save_controls(c)
        await q.edit_message_text("Entries resumed ▶️", reply_markup=keyboard())
        return

    if data == "positions":
        c = load_controls()
        c["print_positions_now"] = True
        save_controls(c)
        await q.edit_message_text(fmt_positions(), reply_markup=keyboard())
        return

    if data == "equity":
        await q.message.reply_text(render_equity_text(), reply_markup=keyboard())
        return

    if data == "closeall":
        c = load_controls()
        c["close_all"] = True
        save_controls(c)
        await q.edit_message_text("Close-all requested 🧹", reply_markup=keyboard())
        return

    if data == "restart":
        c = load_controls()
        c["restart"] = True
        save_controls(c)
        await q.edit_message_text("Restart requested 🔄 (close all + continue)", reply_markup=keyboard())
        return

    if data == "shutdown":
        c = load_controls()
        c["shutdown"] = True
        save_controls(c)
        await q.edit_message_text("Shutdown requested ⛔ (close all + stop)", reply_markup=keyboard())
        return


# ============================================================
# PROGRAM ENTRY
# ============================================================
async def post_init(app):
    if not ENGINE.running:
        asyncio.create_task(ENGINE.run())
    LOG.info("[TELEGRAM] controller online (master ogrift.py)")



# ============================================================
# REPLAY / BACKTEST HARNESS (deterministic, single-symbol)
# ============================================================
def run_replay_mode(
    exchange_name: str,
    symbol: str,
    since_ms: Optional[int],
    limit: int,
    starting_equity: float,
    assumed_spread_pct: float,
) -> None:
    ex = getattr(ccxt, exchange_name)(
        {"enableRateLimit": True, "options": {"defaultType": "spot", "loadAllOptions": False}}
    )
    ex.load_markets(False)

    # Isolate replay logs from live
    if not os.getenv("RIFT_TRADE_LOG_FILE"):
        os.environ["RIFT_TRADE_LOG_FILE"] = str(BASE_DIR / "trades_replay.jsonl")

    state = default_state()
    state["equity"] = float(starting_equity)
    state["equity_start"] = float(starting_equity)
    state["positions"] = {}
    state["cooldowns"] = {}
    state["candidates"] = {}
    state["loss_clusters"] = {}
    state["perf"] = {}

    ohlcv = ex.fetch_ohlcv(symbol, timeframe=TIMEFRAME, since=since_ms, limit=int(limit))
    if not ohlcv or len(ohlcv) < (CANDLE_LIMIT + 10):
        raise RuntimeError(f"Not enough candles for replay: got {len(ohlcv) if ohlcv else 0}")

    # Warmup index: need enough candles for indicators
    warmup = max(CANDLE_LIMIT, EMA_SLOW + 10, RSI_PERIOD + 10, ATR_PERIOD + 10)

    def _ticker_from_close(close_px: float) -> dict:
        mid = float(close_px)
        # symmetric spread around mid
        half = float(assumed_spread_pct) / 2.0
        bid = mid * (1.0 - half)
        ask = mid * (1.0 + half)
        return {"bid": bid, "ask": ask, "last": mid}

    for i in range(warmup, len(ohlcv)):
        ts_ms = int(ohlcv[i][0])
        now_ts = int(ts_ms // 1000)
        state["_now"] = now_ts

        # Equity guardrails update (uses replay trade log)
        update_equity_guard(state, now_ts)

        # --- exits ---
        tp_dollars, sl_dollars, _prof = get_tp_sl_dollars(state)
        be_trigger, trail_trigger, trail_giveback = get_be_trail_params(state)

        for pid in list(state["positions"].keys()):
            p = state["positions"][pid]
            if p.get("symbol") != symbol:
                continue

            last = float(ohlcv[i][4])
            fee_rate = float(p.get("fee_rate", get_taker_fee_rate(ex, symbol)))
            unreal_gross = (last - float(p["entry"])) * float(p["qty"])
            unreal = calc_net_unreal(p, last, fee_rate)
            p["last"] = last
            p["unreal_gross"] = unreal_gross
            p["unreal"] = unreal
            p["fee_rate"] = fee_rate

            # peak tracking on net unreal
            prev_peak = float(p.get("peak_unreal", unreal))
            p["peak_unreal"] = max(prev_peak, float(unreal))

            # TP/SL (net)
            if unreal >= float(tp_dollars):
                close_position(state, pid, last, f"TP_{_prof}")
                continue
            if unreal <= float(sl_dollars):
                close_position(state, pid, last, f"SL_{_prof}")
                continue

            # BE arm (tight-gated)
            if not p.get("be_armed", False):
                age_s = now_ts - int(p.get("opened_ts", now_ts))
                peak = float(p.get("peak_unreal", unreal))
                if age_s >= int(BE_MIN_HOLD_SECONDS) and peak >= float(BE_MIN_PEAK_USD) and unreal >= max(float(be_trigger), float(BE_ARM_AT_PROFIT_USD)):
                    p["be_armed"] = True
                    p["be_armed_ts"] = now_ts

            # Trailing activate + update stop
            if not p.get("trail_active", False):
                if float(p.get("peak_unreal", unreal)) >= float(trail_trigger) and float(p.get("peak_unreal", unreal)) >= float(BE_MIN_PEAK_USD):
                    p["trail_active"] = True
                    p["trail_stop_unreal"] = float(p.get("peak_unreal", unreal)) - float(trail_giveback)
            if p.get("trail_active", False):
                peak = float(p.get("peak_unreal", unreal))
                stop = float(p.get("trail_stop_unreal", 0.0))
                new_stop = max(stop, peak - float(trail_giveback))
                p["trail_stop_unreal"] = new_stop
                if unreal <= float(new_stop) and peak >= float(BE_MIN_PEAK_USD):
                    close_position(state, pid, last, "TRAIL_EXIT")
                    continue

            # Post-BE profit-lock (only if peak is meaningful)
            if p.get("be_armed", False):
                peak = float(p.get("peak_unreal", unreal))
                if peak >= max(float(BE_MIN_PEAK_USD), float(be_trigger)) and unreal <= float(BE_EXIT_UNREAL_DOLLARS) and unreal > 0:
                    close_position(state, pid, last, "POST_BE_PROFIT_LOCK")
                    continue

            # Stagnation + exit-on-green
            opened = int(p.get("opened_ts", now_ts))
            if (now_ts - opened) >= int(MAX_HOLD_SECONDS):
                if not p.get("recovery_deadline"):
                    p["recovery_deadline"] = now_ts + int(RECOVERY_WINDOW_SECONDS)
                if now_ts >= int(p.get("recovery_deadline", 0)):
                    if EXIT_ON_GREEN_AFTER_STAG_TIMEOUT:
                        if not p.get("eog_armed", False):
                            p["eog_armed"] = True
                            p["eog_armed_ts"] = now_ts
                        if float(unreal) > float(EXIT_ON_GREEN_MIN_UNREAL):
                            close_position(state, pid, last, "EOG_EXIT")
                            continue
                    else:
                        close_position(state, pid, last, "STAG_TIMEOUT_EXIT")
                        continue

        # --- entries ---
        # Performance gate
        perf_ok, _perf_msg = perf_allows_entries(state, now_ts)
        if not perf_ok:
            continue

        # Cooldown
        if int(state.get("cooldowns", {}).get(symbol, 0)) > now_ts:
            continue
        if len(state["positions"]) >= int(MAX_OPEN_POSITIONS):
            continue
        if ONE_BASE_ASSET_AT_A_TIME:
            open_bases = {base_asset(pos["symbol"]) for pos in state["positions"].values()}
            b = base_asset(symbol)
            if b and b in open_bases:
                continue

        last = float(ohlcv[i][4])
        t = _ticker_from_close(last)
        qv = float(ohlcv[i][5]) * float(last)  # volume * price ~= quote volume
        ok_price, _why = pass_price_sanity(symbol, last, qv)
        if not ok_price:
            continue

        sp = spread_pct_from_ticker(t)
        if sp is not None and float(sp) > float(MAX_SPREAD_PCT):
            continue

        position_value, _ = get_position_value(state)
        if float(position_value) < float(MIN_NOTIONAL_USD):
            continue

        candles_slice = ohlcv[max(0, i - CANDLE_LIMIT + 1) : i + 1]
        feat = compute_features_from_candles(candles_slice)
        if not feat or feat.get("peg_block", False):
            continue

        regime_ok, _ = regime_allows_entry(state, now_ts, feat.get("regime", {}))
        if not regime_ok:
            continue

        score = float(feat.get("score", 0.0))
        persist_ok, _ = update_candidate(state, symbol, score, now_ts)
        if not persist_ok:
            continue

        qty_raw = float(position_value) / last
        fee_rate = get_taker_fee_rate(ex, symbol)
        qty, _ = normalize_order_qty(ex, symbol, qty_raw, last)
        if qty is None:
            continue
        fee_open = (float(qty) * float(last) * float(fee_rate)) if INCLUDE_FEES_IN_PNL else 0.0

        pid = f"REPLAY:{symbol}:{now_ts}"
        state["positions"][pid] = {
            "exchange": "REPLAY",
            "symbol": symbol,
            "entry": last,
            "qty": float(qty),
            "fee_rate": float(fee_rate),
            "fee_open": float(fee_open),
            "opened_ts": now_ts,
            "recovery_deadline": 0,
            "eog_armed": False,
            "eog_armed_ts": 0,
            "be_armed": False,
            "be_armed_ts": 0,
            "trail_active": False,
            "peak_unreal": 0.0,
            "trail_stop_unreal": 0.0,
            "last": last,
            "unreal": 0.0,
            "entry_score": score,
            "entry_trend_score": float(feat.get("trend_score", 0.0)),
            "entry_atr_ratio": float(feat.get("atr_ratio", 1.0)),
            "entry_rsi": float(feat.get("rsi", 0.0)),
            "entry_spread_pct": float(sp) if sp is not None else None,
            "entry_qv": float(qv),
            "regime_label": str((feat.get("regime") or {}).get("label", "")),
            "regime_reason": str((feat.get("regime") or {}).get("reason", "")),
        }
        append_trade_log({"evt": "OPEN", "ts": now_ts, "exchange": "REPLAY", "symbol": symbol, "entry": last, "qty": float(qty), "fee_rate": float(fee_rate), "fee_open": float(fee_open), "score": score})

    LOG.info(f"[REPLAY] done. equity=${state.get('equity', 0.0):,.2f} realized=${state.get('realized_pnl', 0.0):,.2f} open={len(state.get('positions', {}))}")

def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--replay", action="store_true", help="Run deterministic replay/backtest mode (no Telegram).")
    parser.add_argument("--exchange", type=str, default=(EXCHANGE_NAMES[0] if EXCHANGE_NAMES else "bybit"), help="CCXT exchange id")
    parser.add_argument("--symbol", type=str, default="", help="Symbol for replay (e.g., BTC/USDT)")
    parser.add_argument("--since", type=str, default="", help="Since time (YYYY-MM-DD) for replay")
    parser.add_argument("--limit", type=int, default=1500, help="Candles to fetch for replay")
    parser.add_argument("--equity", type=float, default=10000.0, help="Starting equity for replay")
    parser.add_argument("--spread", type=float, default=0.001, help="Assumed spread pct for replay (e.g. 0.001=0.10%)")
    parser.add_argument("--logfile", type=str, default="", help="Replay trade log file (jsonl).")
    args, _ = parser.parse_known_args()

    if args.replay:
        if not args.symbol:
            raise SystemExit("Replay requires --symbol (e.g., BTC/USDT)")
        if args.logfile:
            os.environ["RIFT_TRADE_LOG_FILE"] = args.logfile

        since_ms = None
        if args.since:
            # interpret as UTC date
            dt = datetime.datetime.strptime(args.since, "%Y-%m-%d")
            since_ms = int(dt.replace(tzinfo=datetime.timezone.utc).timestamp() * 1000)

        run_replay_mode(
            exchange_name=args.exchange,
            symbol=args.symbol,
            since_ms=since_ms,
            limit=int(args.limit),
            starting_equity=float(args.equity),
            assumed_spread_pct=float(args.spread),
        )
        return

    # Live mode (Telegram-driven engine)
    acquire_lock()
    if not os.path.exists(CONTROLS_FILE):
        save_controls(default_controls())
    if not os.path.exists(STATE_FILE):
        save_state(default_state())

    # DEV MODE GATE: prevent Telegram polling on non-runner machines
    if os.getenv("RUN_TELEGRAM", "1") != "1":
        LOG.info("[RIFT] RUN_TELEGRAM=0 -> Telegram controller disabled")
        return

    app = ApplicationBuilder().token(BOT_TOKEN).post_init(post_init).build()

    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("startbot", cmd_startbot))
    app.add_handler(CommandHandler("pause", cmd_pause))
    app.add_handler(CommandHandler("resume", cmd_resume))
    app.add_handler(CommandHandler("positions", cmd_positions))
    app.add_handler(CommandHandler("equity", cmd_equity))
    app.add_handler(CommandHandler("batchsize", cmd_batchsize))
    app.add_handler(CommandHandler("closeall", cmd_closeall))
    app.add_handler(CommandHandler("restartbot", cmd_restartbot))
    app.add_handler(CommandHandler("shutdownbot", cmd_shutdownbot))
    app.add_handler(CallbackQueryHandler(on_button))

    LOG.info("[TELEGRAM] starting polling...")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
