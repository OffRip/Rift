import threading
import time
import json
import os
import signal
from dataclasses import dataclass, asdict, fields
from typing import Dict, Any, List, Optional, Tuple

import ccxt

# =========================
# CONFIG
# =========================
import config

TIMEFRAME = config.TIMEFRAME
SCAN_INTERVAL = config.SCAN_INTERVAL
CANDLE_LIMIT = config.CANDLE_LIMIT

EMA_FAST = config.EMA_FAST
EMA_SLOW = config.EMA_SLOW
RSI_PERIOD = config.RSI_PERIOD
RSI_MIN = config.RSI_MIN
RSI_MAX = config.RSI_MAX

COOLDOWN_SECONDS = config.COOLDOWN_SECONDS

DRY_RUN = config.DRY_RUN

START_EQUITY_USD = config.START_EQUITY_USD
RISK_FRACTION_PER_TRADE = config.RISK_FRACTION_PER_TRADE
TP_PCT = config.TP_PCT
SL_PCT = config.SL_PCT

MAX_OPEN_POSITIONS = config.MAX_OPEN_POSITIONS

STATE_FILE = config.STATE_FILE

EXCHANGES = config.EXCHANGES
SYMBOLS = config.SYMBOLS


# =========================
# STATE MODELS
# =========================
@dataclass
class Position:
    exchange: str
    symbol: str
    trigger: str              # "cross" or "trend"
    entry: float
    qty: float
    tp: float
    sl: float
    opened_at: int


# =========================
# UTIL: STATE LOAD/SAVE
# =========================
def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_FILE):
        return {
            "equity": START_EQUITY_USD,
            "realized_pnl": 0.0,
            "cooldowns": {},
            "positions": {},
        }
    try:
        with open(STATE_FILE, "r") as f:
            data = json.load(f)
        # normalize
        data.setdefault("equity", START_EQUITY_USD)
        data.setdefault("realized_pnl", 0.0)
        data.setdefault("cooldowns", {})
        data.setdefault("positions", {})
        if not isinstance(data["cooldowns"], dict):
            data["cooldowns"] = {}
        if not isinstance(data["positions"], dict):
            data["positions"] = {}
        return data
    except Exception:
        # don't brick the bot if file is corrupted
        return {
            "equity": START_EQUITY_USD,
            "realized_pnl": 0.0,
            "cooldowns": {},
            "positions": {},
        }


def save_state(state: Dict[str, Any]) -> None:
    tmp_path = STATE_FILE + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp_path, STATE_FILE)


# =========================
# UTIL: INDICATORS
# =========================
def ema(values: List[float], period: int) -> List[Optional[float]]:
    n = len(values)
    if n < period:
        return [None] * n
    k = 2 / (period + 1)
    out: List[Optional[float]] = [None] * (period - 1)

    sma = sum(values[:period]) / period
    out.append(sma)

    prev = sma
    for v in values[period:]:
        prev = (v - prev) * k + prev
        out.append(prev)

    return out


def rsi(closes: List[float], period: int = RSI_PERIOD) -> List[Optional[float]]:
    n = len(closes)
    if n <= period:
        return [None] * n

    gains: List[float] = []
    losses: List[float] = []
    for i in range(1, n):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))

    out: List[Optional[float]] = [None] * period
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    def calc_rsi(ag: float, al: float) -> float:
        if al == 0:
            return 100.0
        rs = ag / al
        return 100.0 - (100.0 / (1.0 + rs))

    out.append(calc_rsi(avg_gain, avg_loss))

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        out.append(calc_rsi(avg_gain, avg_loss))

    # pad to match closes length
    out = [None] + out
    return out[:n]


# =========================
# EXCHANGE HELPERS (PUBLIC)
# =========================
def make_public_exchange(name: str):
    ex_cls = getattr(ccxt, name)
    return ex_cls({"enableRateLimit": True})


def fetch_ohlcv(ex, symbol: str) -> List[List[float]]:
    return ex.fetch_ohlcv(symbol, timeframe=config.TIMEFRAME, 
limit=CANDLE_LIMIT)


def fetch_last(ex, symbol: str) -> float:
    t = ex.fetch_ticker(symbol)
    last = t.get("last")
    if last is None:
        raise ValueError("ticker last is None")
    return float(last)


# =========================
# COOLDOWNS / GATES
# =========================
def cd_key(exchange: str, symbol: str) -> str:
    return f"{exchange}:{symbol}"


def in_cooldown(state: Dict[str, Any], key: str, now: int) -> bool:
    return now < int(state["cooldowns"].get(key, 0))


def set_cooldown(state: Dict[str, Any], key: str, now: int) -> None:
    state["cooldowns"][key] = now + COOLDOWN_SECONDS


# =========================
# POSITIONS (PAPER)
# =========================
def positions_dict_to_objects(state):
    positions = state.get("positions", {}) or {}
    out = {}
    allowed = {f.name for f in fields(Position)}
    for k, v in positions.items():
        if not isinstance(v, dict):
            continue

        v = dict(v)  # copy so we can safely edit

        # Backward-compat schema fixes
        if "opened_ts" in v and "opened_at" not in v:
            v["opened_at"] = v.pop("opened_ts")
        if "trigger" not in v or v.get("trigger") is None:
            v["trigger"] = "legacy"

        if "tp" not in v or v.get("tp") is None:
            v["tp"] = 0.0

        if "sl" not in v or v.get("sl") is None:
            v["sl"] = 0.0
        # (optional) handle other legacy keys if you ever renamed them
        # if "entry_ts" in v and "entry_at" not in v:
        #     v["entry_at"] = v.pop("entry_ts")
        cleaned = {kk: vv for kk, vv in v.items() if kk in allowed}
        out[k] = Position(**cleaned)

    return out


def positions_objects_to_dict(pos: Dict[str, Position]) -> Dict[str, Any]:
    return {k: asdict(v) for k, v in pos.items()}


def pos_id(exchange: str, symbol: str) -> str:
    return f"{exchange}:{symbol}"


def can_open_more(positions: Dict[str, Position]) -> bool:
    return len(positions) < MAX_OPEN_POSITIONS


def already_open_symbol(positions: Dict[str, Position], exchange: str, symbol: str) -> bool:
    return pos_id(exchange, symbol) in positions


def size_qty(equity: float, entry: float) -> float:
    # Paper sizing: allocate fraction of equity as notional
    notional = equity * RISK_FRACTION_PER_TRADE
    if entry <= 0:
        return 0.0
    return notional / entry


def open_position(
    positions: Dict[str, Position],
    state: Dict[str, Any],
    exchange: str,
    symbol: str,
    trigger: str,
    entry: float,
    now: int,) -> Position:
    qty = size_qty(float(state["equity"]), entry)
    tp = entry * (1.0 + TP_PCT)
    sl = entry * (1.0 - SL_PCT)
    p = Position(
        exchange=exchange,
        symbol=symbol,
        trigger=trigger,
        entry=entry,
        qty=qty,
        tp=tp,
        sl=sl,
        opened_at=now,
    )
    positions[pos_id(exchange, symbol)] = p
    return p


def close_position(
    positions: Dict[str, Position],
    state: Dict[str, Any],
    pid: str,
    exit_price: float,
    reason: str,
    now: int,) -> Tuple[float, str]:
    p = positions[pid]
    pnl = (exit_price - p.entry) * p.qty
    state["realized_pnl"] = float(state.get("realized_pnl", 0.0)) + pnl
    state["equity"] = float(state.get("equity", START_EQUITY_USD)) + pnl
    del positions[pid]
    msg = (
        f"CLOSE {p.exchange} {p.symbol} reason={reason} "
        f"entry={p.entry:.2f} exit={exit_price:.2f} qty={p.qty:.6f} pnl={pnl:+.2f}"
    )
    return pnl, msg


def check_exits(exchanges: Dict[str, Any], positions: Dict[str, Position], 
state: Dict[str, Any]) -> List[str]:
    now = int(time.time())
    logs: List[str] = []
    # We copy keys so we can delete while iterating
    for pid in list(positions.keys()):
        p = positions[pid]
        ex = exchanges.get(p.exchange)
        if ex is None:
            continue
        try:
            last = fetch_last(ex, p.symbol)
            if last >= p.tp:
                _, msg = close_position(positions, state, pid, last, "TP", now)
                logs.append(msg)
            elif last <= p.sl:
                _, msg = close_position(positions, state, pid, last, "SL", now)
                logs.append(msg)
        except Exception as e:
            logs.append(f"[WARN] exit-check {p.exchange} {p.symbol}: {type(e).__name__} {e}")
    return logs


def compute_unrealized(exchanges: Dict[str, Any], positions: Dict[str, Position]) -> Tuple[float, List[str]]:
    total = 0.0
    lines: List[str] = []
    for pid, p in positions.items():
        ex = exchanges.get(p.exchange)
        if ex is None:
            continue
        try:
            last = fetch_last(ex, p.symbol)
            u = (last - p.entry) * p.qty
            total += u
            lines.append(
                f"- {p.exchange} {p.symbol} [{p.trigger}] entry={p.entry:.2f} last={last:.2f} qty={p.qty:.6f}"
            )
        except Exception as e:
            lines.append(f"- {p.exchange} {p.symbol} [WARN ticker] {type(e).__name__} {e}")
    return total, lines


# =========================
# SIGNAL SCAN (CANDLE-SAFE)
# =========================
def evaluate_symbol(ex, symbol: str) -> Tuple[int, Optional[Dict[str, 
Any]]]:
    ohlcv = fetch_ohlcv(ex, symbol)
    last_ts = int(ohlcv[-1][0])
    closes = [c[4] for c in ohlcv]

    e_fast = ema(closes, EMA_FAST)
    e_slow = ema(closes, EMA_SLOW)
    r = rsi(closes, RSI_PERIOD)

    if e_fast[-1] is None or e_slow[-1] is None or r[-1] is None:
        return last_ts, None

    crossed_up = (e_fast[-2] <= e_slow[-2]) and (e_fast[-1] > e_slow[-1])
    trend_up = e_fast[-1] > e_slow[-1]
    r_ok = RSI_MIN <= float(r[-1]) <= RSI_MAX

    if not r_ok:
        return last_ts, None

    trigger = "cross" if crossed_up else ("trend" if trend_up else None)
    if trigger is None:
        return last_ts, None

    return last_ts, {
        "close": closes[-1],
        "ema_fast": float(e_fast[-1]),
        "ema_slow": float(e_slow[-1]),
        "rsi": float(r[-1]),
        "trigger": trigger,
    }


# =========================
# MAIN LOOP
# =========================
_shutdown = False


def _handle_shutdown(sig, frame):
    global _shutdown
    _shutdown = True


def main():
    global _shutdown
    print("[SCANNER] running in background thread")
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, _handle_shutdown)
        signal.signal(signal.SIGTERM, _handle_shutdown)

    exchanges: Dict[str, Any] = {name: make_public_exchange(name) for name in EXCHANGES}

    state = load_state()
    positions = positions_dict_to_objects(state)

    # Candle gate: track last candle timestamp per exchange+symbol
    last_candle_ts: Dict[str, int] = {}

    # One-entry-per-loop lock (prevents multi-fires)
    entry_fired_this_loop = False

    print(f"Scanner started. config.timeframe={config.TIMEFRAME} scan_every={config.SCAN_INTERVAL}s "
          f"cooldown={COOLDOWN_SECONDS}s DRY_RUN={DRY_RUN}")

    while not _shutdown:
        now = int(time.time())

        # Reset per-loop entry lock
        entry_fired_this_loop = False

        # 1) Exit checks (TP/SL)
        exit_logs = check_exits(exchanges, positions, state)
        if exit_logs:
            for line in exit_logs:
                print(line)
            # Persist after exits
            state["positions"] = positions_objects_to_dict(positions)
            save_state(state)

        # 2) Scan for entries (candle-safe + cooldown + caps)
        signals: List[str] = []
        for ex_name, ex in exchanges.items():
            for sym in SYMBOLS:
                k = cd_key(ex_name, sym)

                # cooldown gate
                if in_cooldown(state, k, now):
                    continue

                # no duplicate open
                if already_open_symbol(positions, ex_name, sym):
                    continue

                # position cap
                if not can_open_more(positions):
                    continue

                # one-entry-per-loop
                if entry_fired_this_loop:
                    continue

                try:
                    candle_ts, siginfo = evaluate_symbol(ex, sym)

                    # candle-safe gate
                    if last_candle_ts.get(k) == candle_ts:
                        continue
                    last_candle_ts[k] = candle_ts

                    if siginfo is None:
                        continue

                    trigger = siginfo["trigger"]
                    close_price = float(siginfo["close"])

                    # Open paper position
                    p = open_position(
                        positions=positions,
                        state=state,
                        exchange=ex_name,
                        symbol=sym,
                        trigger=trigger,
                        entry=close_price,
                        now=now,
                    )

                    set_cooldown(state, k, now)  # set cooldown on signal
                    entry_fired_this_loop = True

                    signals.append(
                        f"OPEN {ex_name} {sym} [{trigger}] entry={p.entry:.2f} "
                        f"qty={p.qty:.6f} tp={p.tp:.2f} sl={p.sl:.2f} rsi={siginfo['rsi']:.2f}"
                    )

                except Exception as e:
                    print(f"[WARN] scan {ex_name} {sym}: {type(e).__name__} {e}")

        if signals:
            print("\n==== SIGNALS ====")
            for s in signals:
                print(s)

            # Persist after entries
            state["positions"] = positions_objects_to_dict(positions)
            save_state(state)

        # 3) Print positions + PnL snapshot (like your output)
        unreal, pos_lines = compute_unrealized(exchanges, positions)
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        eq = float(state.get("equity", START_EQUITY_USD))
        rpnl = float(state.get("realized_pnl", 0.0))
        print("\n==== POSITIONS ====")
        print(f"time={ts} | equity=${eq:,.2f} | realized_pnl=${rpnl:,.2f}")
        print(f"open_positions={len(positions)}/{MAX_OPEN_POSITIONS}")
        for line in pos_lines:
            print(line)
        print(f"total_unrealized_pnl=${unreal:,.2f}")
        print("==================")

        # sleep
        time.sleep(config.SCAN_INTERVAL)
    # graceful shutdown save
    state["positions"] = positions_objects_to_dict(positions)
    save_state(state)
    print("Scanner stopped cleanly.")


if __name__ == "__main__":
    main()

