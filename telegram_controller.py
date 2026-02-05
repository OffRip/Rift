import os
import sys
import json
import time
import signal
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import BadRequest
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

# =========================
# ENV
# =========================
load_dotenv("info.env")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN in info.env")

# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent
BOT_SCRIPT = BASE_DIR / "ogrift.py"
PID_FILE = BASE_DIR / "rift.pid"
STATE_FILE = BASE_DIR / "state.json"
CONTROLS_FILE = BASE_DIR / "controls.json"


# =========================
# PROCESS CONTROL
# =========================
def _read_pid():
    if not PID_FILE.exists():
        return None
    try:
        return int(PID_FILE.read_text().strip())
    except Exception:
        return None


def is_running():
    pid = _read_pid()
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def start_bot():
    if is_running():
        return "Bot already running ✅"

    if not BOT_SCRIPT.exists():
        return f"Can't find {BOT_SCRIPT.name} in {BASE_DIR}"

    proc = subprocess.Popen(
        [sys.executable,"-u", str(BOT_SCRIPT)],
        cwd=str(BASE_DIR),
        start_new_session=True,
    )
    PID_FILE.write_text(str(proc.pid))
    print(f"[controller] started {BOT_SCRIPT.name} pid={proc.pid}")
    return f"Started bot ✅ (pid={proc.pid})"


def stop_bot():
    # 1) Request graceful shutdown (engine will close_all automatically)
    c = load_controls()
    c["shutdown"] = True
    save_controls(c)

    # 2) Give the bot a moment to print + close positions
    # (your engine checks controls every loop, so this is enough)
    time.sleep(2)

    # 3) If still running, force kill as a fallback
    if is_running():
        pid = _read_pid()
        try:
            os.killpg(pid, signal.SIGTERM)
        except Exception:
            pass
        PID_FILE.unlink(missing_ok=True)
        return "Shutdown requested (forced kill fallback used) 🛑"

    PID_FILE.unlink(missing_ok=True)
    return "Shutdown requested (graceful) 🛑"


def restart_bot():
    # Close-all on restart is handled inside ogrift.py (paper positions),
    # or you can keep it as just stop+start for now.
    msg1 = stop_bot() if is_running() else "Bot not running."
    time.sleep(1)
    msg2 = start_bot()
    return f"{msg1}\n{msg2}"


# =========================
# STATE HELPERS
# =========================
def load_state():
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def load_controls():
    if not CONTROLS_FILE.exists():
        return {"pause_entries": False}
    try:
        return json.loads(CONTROLS_FILE.read_text())
    except Exception:
        return {"pause_entries": False}


def save_controls(c):
    CONTROLS_FILE.write_text(json.dumps(c, indent=2))


# =========================
# UI + TEXT
# =========================
def keyboard():
    c = load_controls()
    cur_batch = int(c.get("batch_size", 30))

    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("▶️ Start Bot", 
callback_data="startbot"),
            InlineKeyboardButton("🔄 Restart Bot", 
callback_data="restartbot"),
        ],
        [
            InlineKeyboardButton("⏸ Pause", callback_data="pause"),
            InlineKeyboardButton("▶️ Resume", callback_data="resume"),
        ],
        [
            InlineKeyboardButton("🛑 Shutdown Bot", 
callback_data="shutdownbot"),
        ],
        [
            InlineKeyboardButton(f"➖ Batch -10 ({cur_batch})", 
callback_data="batch_-10"),
            InlineKeyboardButton(f"➕ Batch +10 ({cur_batch})", 
callback_data="batch_+10"),
        ],
        [
            InlineKeyboardButton("📍 Status", callback_data="status"),
            InlineKeyboardButton("❓ Help", callback_data="help"),
        ],
    ])


def status_text():
    c = load_controls()
    batch = int(c.get("batch_size", 30))
    paused = bool(c.get("pause_entries", False))

    return (
        f"Bot: {'RUNNING ✅' if is_running() else 'STOPPED ❌'}\n"
        f"Script: {BOT_SCRIPT.name}\n"
        f"batch_size: {batch}\n"
        f"entries: {'PAUSED ⏸' if paused else 'ACTIVE ▶️'}\n"
        f"Python: {sys.executable}\n"
    )


def help_text():
    return (
        "RIFT Commands:\n"
        "/start — show menu\n"
        "/help — show this help\n"
        "/status — bot status\n"
        "/startbot — start ogrift.py\n"
        "/shutdownbot — stop ogrift.py\n"
        "/restartbot — restart ogrift.py\n"
        "/positions — open positions (from state.json)\n"
        "/equity — equity + realized PnL\n"
        "/pause — pause new entries\n"
        "/resume — resume new entries\n"
    )


async def safe_edit(q, text: str):
    try:
        await q.edit_message_text(text, reply_markup=keyboard())
    except BadRequest as e:
        if "Message is not modified" in str(e):
            return
        raise


# =========================
# COMMANDS
# =========================
async def cmd_batch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Usage: /batch 50
    if not context.args:
        c = load_controls()
        cur = c.get("batch_size", 30)
        await update.message.reply_text(f"Current batch_size = {cur}\nUsage: /batch <5-200>", reply_markup=keyboard())
        return

    try:
        n = int(context.args[0])
    except Exception:
        await update.message.reply_text("Usage: /batch <5-200>", reply_markup=keyboard())
        return

    n = max(5, min(n, 200))
    c = load_controls()
    c["batch_size"] = n
    save_controls(c)

    await update.message.reply_text(f"✅ batch_size set to {n}", reply_markup=keyboard())

async def cmd_start(update: 
Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(status_text(), reply_markup=keyboard())


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(help_text(), reply_markup=keyboard())


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(status_text(), reply_markup=keyboard())


async def cmd_startbot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(start_bot(), reply_markup=keyboard())


async def cmd_shutdownbot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    controls = load_controls()
    controls["close_all"] = True
    controls["shutdown"] = True
    save_controls(controls)
    await update.message.reply_text(stop_bot(), reply_markup=keyboard())


async def cmd_restartbot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(restart_bot(), reply_markup=keyboard())


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


async def cmd_equity(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s = load_state()
    eq = float(s.get("equity", 0.0))
    pnl = float(s.get("realized_pnl", 0.0))
    await update.message.reply_text(
        f"Equity: ${eq:,.2f}\nRealized PnL: ${pnl:,.2f}",
        reply_markup=keyboard()
    )


async def cmd_positions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s = load_state()
    positions = s.get("positions", {})
    if not positions:
        await update.message.reply_text("No open positions.", reply_markup=keyboard())
        return

    lines = []
    for p in positions.values():
        lines.append(
            f"{p.get('exchange')} {p.get('symbol')} "
            f"entry={float(p.get('entry', 0.0)):.2f} qty={float(p.get('qty', 0.0)):.6f}"
        )

    await update.message.reply_text("Open Positions:\n" + "\n".join(lines), reply_markup=keyboard())


# =========================
# BUTTON HANDLER
# =========================
def set_batch_size(n: int) -> int:
    n = max(5, min(int(n), 200))  # clamp
    c = load_controls()
    c["batch_size"] = n
    save_controls(c)
    return n

def bump_batch_size(delta: int) -> int:
    c = load_controls()
    cur = int(c.get("batch_size", 30))
    return set_batch_size(cur + delta)

async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()

    if q.data == "startbot":
        msg = start_bot()
    elif q.data == "shutdownbot":
        msg = stop_bot()
    elif q.data == "restartbot":
        msg = restart_bot()
    elif q.data == "pause":
        c = load_controls()
        c["pause_entries"] = True
        save_controls(c)
        msg = "Entries paused ⏸"
    elif q.data == "resume":
        c = load_controls()
        c["pause_entries"] = False
        save_controls(c)
        msg = "Entries resumed ▶️"
    elif q.data == "help":
        msg = help_text()
    elif q.data == "batch_-10":
    	new_n = bump_batch_size(-10)
    	msg = f"✅ batch_size set to {new_n}"
    elif q.data == "batch_+10":
    	new_n = bump_batch_size(+10)
    	msg = f"✅ batch_size set to {new_n}"
    else:
        msg = status_text()

    await safe_edit(q, msg)


# =========================
# MAIN
# =========================
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("batch", cmd_batch))

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("startbot", cmd_startbot))
    app.add_handler(CommandHandler("shutdownbot", cmd_shutdownbot))
    app.add_handler(CommandHandler("restartbot", cmd_restartbot))
    app.add_handler(CommandHandler("pause", cmd_pause))
    app.add_handler(CommandHandler("resume", cmd_resume))
    app.add_handler(CommandHandler("equity", cmd_equity))
    app.add_handler(CommandHandler("positions", cmd_positions))

    app.add_handler(CallbackQueryHandler(on_button))

    print("Telegram controller online")
    app.run_polling()


if __name__ == "__main__":
    main()

