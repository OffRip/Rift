# config.py — single source of truth. Do NOT assign into this from other files.

TIMEFRAME = "1m"
SCAN_INTERVAL = 30
CANDLE_LIMIT = 300

EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 14
RSI_MIN = 55
RSI_MAX = 75

COOLDOWN_SECONDS = 45 * 60  # 45 minutes

DRY_RUN = True
START_EQUITY_USD = 10_000.0
RISK_FRACTION_PER_TRADE = 0.10

TP_PCT = 0.025
SL_PCT = 0.012

MAX_OPEN_POSITIONS = 3

STATE_FILE = "state.json"
CONTROLS_FILE = "controls.json"

EXCHANGES = ["coinbase", "kraken"]  # public data only
SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD"]

