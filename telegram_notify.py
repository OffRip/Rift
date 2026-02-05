import os
import threading
import urllib.parse
import urllib.request
from dotenv import load_dotenv

# Load env (match your project)
load_dotenv("info.env")  # change to load_dotenv() if you use .env

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def _send_blocking(text: str) -> None:
    if not BOT_TOKEN or not CHAT_ID:
        return

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "disable_web_page_preview": True,
    }

    data = urllib.parse.urlencode(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            resp.read()
    except Exception:
        # Never crash the bot for Telegram
        pass


def notify(text: str) -> None:
    # Fire-and-forget (non-blocking)
    t = threading.Thread(target=_send_blocking, args=(text,), daemon=True)
    t.start()

