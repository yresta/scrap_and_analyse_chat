import re
import pandas as pd

def clean_text_for_clustering(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)    # URL
    text = re.sub(r'@\w+', '', text)                # mentions
    text = re.sub(r'#\w+', '', text)                # hashtags
    text = re.sub(r'\d+', '', text)                 # numbers
    text = re.sub(r'[^a-z\s]', '', text)            # non-letters
    text = re.sub(r"(.)\1{2,}", r"\1", text)        # repeated chars
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words_id and len(t) > 2]
    tokens = [stemmer.stem(t) for t in tokens]
    cleaned = ' '.join(tokens)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

# Load dictionary kata baku
try:
    dictionary_df = pd.read_csv("kata_baku.csv")
    spelling_correction = dict(zip(dictionary_df['tidak_baku'], dictionary_df['kata_baku']))
except Exception as e:
    st.warning(f"Gagal memuat kata_baku.csv: {e}")
    spelling_correction = {}

def correct_spelling(text, corrections):
    if not isinstance(text, str):
        return text
    words = text.split()
    corrected_words = [corrections.get(word, word) for word in words]
    return ' '.join(corrected_words)


def is_unimportant_sentence(text: str) -> bool:
    if not isinstance(text, str):
        return True
    txt = text.strip().lower()
    unimportant_phrases = [
        "siap", "noted", "oke", "ok", "baik", "sip", "thanks", "makasih", "terima kasih",
        "info apa", "info ni", "info nya", "trus ini", "terus ini", "ini saja", "ini aja",
        "ini min", "ini ya", "ini??", "ini?", "ini.", "ini", "sudah", "udah", "iya", "ya", "oh", "ohh"
    ]
    kata_tanya = ['apa','bagaimana','kenapa','siapa','kapan','dimana','mengapa','gimana','kok']
    if len(txt.split()) <= 2 and not any(q in txt for q in kata_tanya):
        if any(phrase == txt or phrase in txt for phrase in unimportant_phrases):
            return True
    return False


def is_question_like(text: str) -> bool:
    if pd.isna(text) or not isinstance(text, str):
        return False
    txt = text.strip().lower()
    if not txt:
        return False
    if '?' in txt:
        return True
    if len(txt.split()) < 3:
        if any(k in txt for k in ['apa','apakah','kapan','siapa','dimana','kenapa','bagaimana']):
            return True
        return False
    question_phrases = [
        # ==== 1. Permintaan Informasi Umum ====
        "ada yang tahu", "ada yg tau", "ada yg tahu", "ada yang tau ga", "ada yang tau gak",
        "ada yg punya info", "ada yg punya kabar", "ada kabar ga", "ada berita", "ada yg denger",
        "ada yg liat", "ada yg nemu", "ada yg ngalamin", "ada yang pernah", "yg udah tau",
        "udah ada yang tau", "ada info dong", "ada info gak", "info dong", "info donk",
        "kasih info dong", "kasih tau dong", "denger2 katanya", "bener gak sih",
        "tau ga", "tau gak", "kalian ada info?", "siapa yang tau?", "dengar kabar",
        "kabar terbaru apa", "yang tau share dong", "bisa kasih info?", "ada update?",

        # ==== 2. Tanya Langsung / Izin Bertanya ====
        "mau tanya", "pengen tanya", "pingin tanya", "ingin bertanya", "izin bertanya",
        "izin nanya", "boleh tanya", "boleh nanya", "numpang tanya", "tanya dong",
        "tanya donk", "nanya dong", "nanya ya", "aku mau nanya", "saya mau tanya",
        "penasaran nih", "penasaran banget", "penasaran donk", "mau nanya nih",
        "mau nanya ya", "btw mau tanya", "eh mau tanya", "boleh tau nggak",
        "pingin nanya", "penasaran aja", "bisa tanya gak", "lagi cari info nih",

        # ==== 3. Permintaan Bantuan / Solusi ====
        "minta tolong", "tolong dong", "tolongin dong", "tolong bantu", "bisa bantu",
        "butuh bantuan", "mohon bantuan", "mohon bantuannya", "minta bantuannya",
        "bisa tolong", "perlu bantuan nih", "ada solusi ga", "ada solusi gak",
        "apa solusinya", "gimana solusinya", "solusinya gimana", "ada yang bisa bantu",
        "ada yg bisa bantuin", "bisa bantuin gak", "butuh pertolongan", "bantu dong",
        "help dong", "help me", "minta tolong ya", "bantuin ya", "ada yang bisa nolong",

        # ==== 4. Permintaan Saran / Pendapat ====
        "ada saran", "minta sarannya", "butuh saran", "rekomendasi dong", "rekomendasi donk",
        "minta rekomendasi", "saran dong", "saran donk", "menurut kalian", "menurut agan",
        "gimana menurut kalian", "bagusnya gimana", "lebih baik yang mana", "kalian pilih yang mana",
        "kira-kira lebih bagus mana", "lebih enak mana", "mending yg mana", "menurutmu gimana",
        "kira2 pilih yg mana", "enaknya pilih yg mana", "bantu saran dong", "bantu milih dong",

        # ==== 5. Konfirmasi / Cek Status ====
        "sudah diproses belum", "udah masuk belum", "udah diapprove belum", "kok belum masuk",
        "belum cair ya", "pencairannya kapan", "kapan cair", "gimana prosesnya", "statusnya gimana",
        "sudah dicek belum", "cek status dong", "minta dicek", "mohon dicek", "sampai kapan ya",
        "bener ga", "ini valid gak", "ini udah benar?", "masih pending ya", "belum juga nih",
        "harus nunggu berapa lama", "status pending kah", "udah diproses kah", "masih dalam proses?",
        "sudah disetujui belum", "udah dikirim belum", "cek dulu dong", "konfirmasi dong",

        # ==== 6. Tanya Cara / Langkah ====
        "cara pakainya gimana", "cara pakenya gimana", "cara daftar gimana", "cara aksesnya gimana",
        "gimana caranya", "caranya gimana", "apa langkahnya", "apa tahapannya",
        "gimana stepnya", "step by step dong", "bisa kasih tutorial?", "tutorial dong",
        "cara install gimana", "cara setup gimana", "gimana setupnya", "konfigurasinya gimana",
        "gimana mulai", "cara mulainya gimana", "cara ngisi gimana", "cara input gimana",
        "login gimana", "cara reset gimana", "cara klaim gimana",

        # ==== 7. Kata Tanya Baku ====
        "apa", "apakah", "siapa", "kapan", "mengapa", "kenapa", "kenapa ya", "bagaimana",
        "gimana", "gimana ya", "gimana sih", "di mana", "dimana", "di mana ya", "berapa",
        "knp ya", "knp sih", "knp bisa", "apa ya", "yang mana ya", "kenapa begitu",
        "mengapakah", "kok bisa", "apa itu", "kenapa tidak",

        # ==== 8. Gaya Chat / Singkatan Umum ====
        "gmn ya", "gmn caranya", "gmn dong", "gmna sih", "gmna ini", "blh mnt",
        "mnt bantu", "mnt saran", "mnt info", "cek donk", "ini knp ya", "ini bgmn ya",
        "ini harus gimana", "ga ngerti", "bngung nih", "bingung banget", "bingung gw",
        "bisa dijelasin", "minta penjelasan", "bingung jelasin dong",

        # ==== 9. Seputar Pembayaran / Transaksi ====
        "va belum aktif ya", "va nya apa", "va nya belum keluar", "kode pembayaran mana",
        "kenapa pending", "kenapa gagal", "tf nya masuk belum", "rekeningnya mana",
        "sudah bayar belum", "bayar kemana", "no rek nya mana", "status tf nya apa",
        "konfirmasi pembayaran gimana", "bayar pakai apa", "pembayaran berhasil ga", "verifikasi donk",
        "rek belum masuk", "sudah transfer", "sudah tf", "uangnya belum masuk", "status transfer",
        "no pembayaran mana", "kode bayar belum muncul", "tf udah masuk?", "rek sudah benar belum"
    ]
    if any(phrase in txt for phrase in question_phrases):
        return True
    first_word = txt.split()[0]
    if first_word in ['apa','apakah','siapa','kapan','mengapa','kenapa','bagaimana','dimana','berapa','gimana']:
        return True
    return False
