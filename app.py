import streamlit as st
import asyncio
from datetime import datetime, timedelta
import pandas as pd

from telegram_scraper import scrape_messages_iter
from preprocessing import correct_spelling, is_unimportant_sentence, is_question_like
from topics import analyze_all_topics
from clustering import load_sentence_model
from config import TIMEZONE

# Load dictionary kata baku
try:
    df_kb = pd.read_csv("kata_baku.csv")
    spelling_correction = dict(zip(df_kb["tidak_baku"], df_kb["baku"]))
except Exception:
    spelling_correction = None

# Streamlit UI
st.set_page_config(page_title="Scraper & Analisis Telegram", layout="wide")
st.title("🏦 Analisis Topik Pertanyaan Grup Telegram")

group = st.text_input("Masukkan username atau ID grup Telegram:", "@contohgroup")
today = datetime.now(TIMEZONE).date()
week_ago = today - timedelta(days=7)

col1, col2 = st.columns(2)
with col1:
    start_date_scrape = st.date_input("Scrape dari tanggal", week_ago, format="YYYY-MM-DD")
with col2:
    end_date_scrape = st.date_input("Scrape sampai tanggal", today, format="YYYY-MM-DD")

run_button = st.button("🚀 Mulai Proses dan Analisis", type="primary")

if run_button:
    model_name = 'paraphrase-multilingual-mpnet-base-v2'
    with st.spinner(f"Memuat model sentence-transformers: {model_name} ..."):
        sentence_model = load_sentence_model(model_name)

        if not group or group.strip() == "" or group.strip() == "@contohgroup":
            st.warning("⚠️ Mohon isi nama grup Telegram yang valid terlebih dahulu.")
            st.stop()

        start_dt = datetime.combine(start_date_scrape, datetime.min.time()).replace(tzinfo=TIMEZONE)
        end_dt = datetime.combine(end_date_scrape, datetime.max.time()).replace(tzinfo=TIMEZONE)

        with st.spinner("Mengambil pesan dari Telegram..."):
            df_all = asyncio.run(scrape_messages_iter(group, start_dt, end_dt))

        if df_all is None or df_all.empty:
            st.error("Gagal mengambil data atau tidak ada data yang ditemukan dalam rentang tanggal.")
            st.stop()
        else:
            st.success(f"✅ Berhasil mengambil {len(df_all)} pesan mentah.")

    st.header("📈 Analisis Topik dari Semua Pertanyaan")
    with st.spinner("Membersihkan data dan mencari pertanyaan..."):
        df_all['text'] = df_all['text'].astype(str)
        df_all['text'] = df_all['text'].str.replace(r'http\S+|www\.\S+', '', regex=True)
        df_all['text'] = df_all['text'].str.strip()
        # Koreksi ejaan jika dictionary tersedia
        if spelling_correction:
            df_all['text'] = df_all['text'].apply(lambda x: correct_spelling(x, spelling_correction))
        # Hapus baris dari sender tertentu
        df_all = df_all[~df_all['sender_name'].isin(['CS TokoLadang', 'Eko | TokLa', 'Vava'])]
        # Hapus kalimat tidak penting
        df_all = df_all[~df_all['text'].apply(is_unimportant_sentence)]
        # Deduplicate (sender, text, date)
        dedup_cols = ['sender_id', 'text', 'date'] if 'sender_id' in df_all.columns else ['sender_name', 'text', 'date']
        df_all = df_all.drop_duplicates(subset=dedup_cols, keep='first').reset_index(drop=True)
        # Deteksi pertanyaan
        df_all['is_question'] = df_all['text'].apply(is_question_like)
        df_questions = df_all[df_all['is_question']].copy()

    tab1, tab2 = st.tabs(["❓ Daftar Pertanyaan", "📊 Analisis Topik"])
    with tab1:
        st.subheader(f"❓ Ditemukan {len(df_questions)} Pesan Pertanyaan" if isinstance(df_questions, pd.DataFrame) else "❓ Ditemukan 0 Pesan Pertanyaan")
        if not df_questions.empty:
            display_cols = [c for c in ['date', 'sender_name', 'text'] if c in df_questions.columns]
            st.dataframe(df_questions[display_cols], use_container_width=True)
        else:
            st.info("Tidak ada pesan yang terdeteksi sebagai pertanyaan pada periode ini.")
    with tab2:
        analyze_all_topics(df_questions, sentence_model)

    st.markdown("---")
    st.success("Analisis Selesai!")
