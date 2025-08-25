import asyncio
import pandas as pd
from telethon import TelegramClient, errors
from zoneinfo import ZoneInfo
import streamlit as st

# === Secrets ===
api_id = int(st.secrets["TELEGRAM_API_ID"])
api_hash = st.secrets["TELEGRAM_API_HASH"]
session_name = "session_utama"
wib = ZoneInfo("Asia/Jakarta")

# === Async Scraper ===
async def scrape_messages_iter(group, start_dt, end_dt, max_messages=None):
    all_messages = []
    sender_cache = {}
    progress_text = st.empty()
    progress = st.progress(0.0)

    try:
        async with TelegramClient(session_name, api_id, api_hash) as client:
            entity = await client.get_entity(group)
            fetched = 0

            async for msg in client.iter_messages(entity):
                if not getattr(msg, "message", None) or not getattr(msg, "date", None):
                    continue

                msg_date_wib = msg.date.astimezone(wib)
                if msg_date_wib < start_dt:
                    break
                if msg_date_wib > end_dt:
                    continue

                sender_id = getattr(msg, "sender_id", None)
                sender_name = sender_cache.get(sender_id)

                if sender_name is None:
                    try:
                        sender = await client.get_entity(sender_id)
                        first_name = getattr(sender, "first_name", "") or ""
                        last_name = getattr(sender, "last_name", "") or ""
                        sender_name = f"{first_name} {last_name}".strip()
                        if not sender_name:
                            sender_name = getattr(sender, "username", f"User ID: {sender_id}")
                    except Exception:
                        sender_name = f"User ID: {sender_id}"
                    sender_cache[sender_id] = sender_name

                all_messages.append({
                    "id": getattr(msg, "id", None),
                    "sender_id": sender_id,
                    "sender_name": sender_name,
                    "text": getattr(msg, "message", ""),
                    "date": msg_date_wib.strftime("%Y-%m-%d %H:%M:%S"),
                    "date_dt": msg_date_wib
                })

                fetched += 1
                if fetched % 50 == 0:
                    progress.progress(min(0.95, fetched / (max_messages or 2000)))
                    progress_text.text(f"Memproses pesan... {fetched}")

            progress.progress(1.0)

    except errors.RPCError as e:
        st.error(f"Error Telethon RPC: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat scraping: {e}")
        return pd.DataFrame()

    if not all_messages:
        return pd.DataFrame()

    return pd.DataFrame(all_messages)

# === Wrapper supaya bisa dipanggil sync di Streamlit ===
def run_scraper(group, start_dt, end_dt, max_messages=1000):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(
        scrape_messages_iter(group, start_dt, end_dt, max_messages=max_messages)
    )
