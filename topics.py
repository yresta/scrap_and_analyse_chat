import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

from clustering import semantic_clustering_auto, embed_texts, get_main_keyword, clean_main_keyword, extract_top_keywords_from_texts
from config import CENTROID_SIMILARITY_THRESHOLD, MIN_EXAMPLES_FOR_CENTROID, MIN_CLUSTER_SHOW, stop_words_id

# Aturan Topik Keyword (rule-based) 
topik_keywords = {
    # Topik logika "DAN" 
    "status_bast": [
        ["bast"],
        ["stuck", "bast"]
    ],
    "verifikasi_toko": [
        ["verifikasi", "toko"],
        ["verivikasi", "toko"],
        ["cek", "id", "toko"]
    ],
    "verifikasi_pembayaran": [
        ["verifikasi", "pembayaran"],
        ["verifikasi", "pesanan"],
        ["verivikasi", "pembayaran"],
        ["minta", "verifikasi"],
        ["konfirmasi"],
        ["notif", "error"],
        ["verifikasi"],
        ["verivikasi"]
    ],
    "penerusan_dana": [
        ["penerusan", "dana"],
        ["dana", "diteruskan"],
        ["uang", "diteruskan"],
        ["penerusan"],
        ["diteruskan"],
        ["meneruskan"],
        ["dana", "teruskan"],
        ["uang", "teruskan"],
        ["penyaluran"],
        ["di teruskan"]
    ],
    "dana_belum_masuk": [
        ["dana", "belum", "masuk"],
        ["uang", "belum", "masuk"],
        ["dana", "masuk", "belum"],
        ["uang", "masuk", "belum"],
        ["dana", "tidak", "masuk"],
        ["uang", "tidak", "masuk"],
        ["dana", "gagal", "masuk"],
        ["uang", "gagal", "masuk"],
        ["belum", "masuk", "rekening"],
        ["belum", "transfer", "masuk"],
        ["belum", "masuk"]
    ],
    "jadwal_cair_dana": [
        ["bos", "cair"],
        ["bop", "cair"],
        ["jadwal", "cair"],
        ["kapan", "cair"],
        ["gelombang", "2"],
        ["tahap", "2"],
        ["pencairan"]
    ],
    "modal_talangan": [
        ["modal", "talangan"],
        ["modal", "kerja"],
        ["dana", "talangan"],
        ["dana", "kerja"],
        ["modal", "bantuan"],
        ["modal", "usaha"],
        ["modal", "bantuan", "usaha"]
    ],
    "kendala_akses" : [
        ["kendala", "akses"],
        ["gagal", "akses"],
        ["tidak", "bisa", "akses"],
        ["tidak", "bisa", "login"],
        ["tidak", "bisa", "masuk"],
        ["gagal", "login"],
        ["gagal", "masuk"],
        ["gagal", "akses"],
        ["reset", "akun"],
        ["reset", "password"],
        ["ganti", "password"],
        ["ganti", "akun"],
        ["ganti", "email"],
        ["ganti", "nomor"],
        ["ganti", "no hp"],
        ["ganti", "no telepon"],
        ["ganti", "telepon"],
        ["eror", "akses"],
        ["eror", "login"],
        ["eror"],
        ["error"],
        ["kapan", "normal"],
        ["trouble"],
        ["ganguan"],
        ["web", "dibuka"],
        ["gk", "bisa", "masuk"],
        ["belum", "lancar"],
        ["bisa", "diakses"],
        ["gangguan"],
        ["gangguannya"],
        ["belum", "normal", "webnya"],
        ["trobel"],
        ["trobelnya"],
        ["ga", "bisa", "akses"],
        ["ga", "bisa", "log", "in"],
        ["ga", "bisa", "masuk"],
        ["ga", "bisa", "web"],
        ["g", "masuk2"],
        ["gk", "bisa2"],
        ["web", "troubel"],
        ["jaringan"],
        ["belum", "bisa", "masuk", "situs"],
        ["belum", "normal", "web"],
        ["vpn"],
        ["gabisa", "login"],
        ["gabisa", "akses"],
        ["g", "bisa", "akses"],
        ["g", "bisa", "login"],
        ["tidak", "bisa", "di", "buka"],
        ["bermasalah", "login"],
        ["login", "trouble"],
        ["maintenance"]
    ],
    "kendala_autentikasi": [
        ["kendala", "autentikasi"],
        ["gagal", "autentikasi"],
        ["tidak", "bisa", "autentikasi"],
        ["gagal", "otentikasi"],
        ["tidak", "bisa", "otentikasi"],
        ["authenticator", "reset"], 
        ["autentikasi"],
        ["autentifikasi"],
        ["otentikasi"],
        ["otp", "gagal"],
        ["otp", "tidak", "bisa"],
        ["otp", "tidak", "muncul"],
        ["otp", "tidak", "tampil"],
        ["otp", "tidak", "ada"],
        ["reset", "barcode"],
        ["google", "authenticator"],
        ["gogle", "authenticator"],
        ["aktivasi", "2 langkah"]
    ],
    "kendala_upload": [
        ["kendala", "upload"],
        ["gagal", "upload"],
        ["tidak", "bisa", "upload"],
        ["gagal", "unggah"],
        ["tidak", "bisa", "unggah"],
        ["produk", "tidak", "muncul"],
        ["produk", "tidak", "tampil"],
        ["produk", "tidak", "ada"],
        ["produk", "massal"],
        ["produk", "masal"],
        ["template", "upload"],
        ["template", "unggah"],
        ["unggah", "produk"],
        ["menambahkan"],
        ["menambah", "produk"],
        ["tambah", "produk"],
        ["tambah", "barang"],
        ["unggah", "foto"],
        ["unggah", "gambar"],
        ["unggah", "foto", "produk"],
        ["unggah", "gambar", "produk"]
    ],
    "kendala_pengiriman": [
        ["tidak", "bisa", "pengiriman"],
        ["barang", "rusak"],
        ["barang", "hilang"],
        ["status", "pengiriman"]
    ],
    "tanda_tangan_elektronik": [
        ["tanda", "tangan", "elektronik"],
        ["ttd", "elektronik"],
        ["tte"],
        ["ttd"],
        ["tt elektronik"],
        ["e", "sign"],
        ["elektronik", "dokumen"]
    ],
    "ubah_data_toko": [
        ["ubah", "data", "toko"],
        ["edit", "data", "toko"],
        ["ubah", "nama", "toko"],
        ["edit", "nama", "toko"],
        ["ubah", "rekening"],
        ["edit", "rekening"],
        ["ubah", "status", "toko"],
        ["edit", "status", "toko"],
        ["ubah", "status", "umkm"],
        ["edit", "status", "umkm"],
        ["ubah", "status", "pkp"]
    ],
    "akun_pengguna": [
        ["ganti", "email"],
        ["ubah", "email"],
        ["ganti", "nama", "akun"],
        ["ubah", "nama", "akun"],
        ["ganti", "akun"],
        ["ubah", "akun"],
        ["gagal", "ganti", "akun"],
        ["gagal", "ubah", "akun"]
    ],
    "pengajuan_modal": [
        ["pengajuan", "modal"],
        ["ajukan", "modal"],
        ["modal", "kerja"],
        ["dana", "talangan"],
        ["dibatalkan", "pengajuan"],
        ["tidak", "bisa", "ajukan"]
    ],
    "pajak": [
        ["pajak", "ppn"],
        ["pajak", "invoice"],
        ["pajak", "npwp"],
        ["pajak", "penghasilan"],
        ["e-billing"],
        ["dipotong", "pajak"],
        ["pajak", "keluaran"],
        ["potongan", "pajak"],
        ["coretax"],
        ["pajak"],
        ["ppn"],
        ["npwp"],
        ["e-faktur"],
        ["efaktur"],
        ["e-billing"]
    ],
    "etika_penggunaan": [
        ["bendahara", "dapat", "untung"],
        ["bendahara", "dagang"],
        ["bendahara", "etik"],
        ["distributor", "dilarang"],
        ["etik", "distributor"],
        ["etik", "larangan"],
        ["etik", "juknis"],
        ["larangan"]
    ],
    "waktu_proses": [
        ["kapan"],
        ["estimasi"],
        ["waktu", "proses"],
        ["waktu", "penyelesaian"],
        ["waktu", "selesai"],
    ],
    
    # Topik logika "ATAU" 
    "pembayaran_dana": ["transfer", "dana masuk", "pengembalian", "bayar", "pembayaran", "dana", "dibayar", "notif pembayaran", "transaksi", "expired"],
    "pengiriman_barang": ["pengiriman", "barang rusak", "kapan dikirim", "status pengiriman", "diproses"],
    "penggunaan_siplah": ["pakai siplah", "siplah", "laporan siplah", "pembelanjaan", "tanggal pembelanjaan", "ubah tanggal", "dokumen", "bisa langsung dipakai", "terhubung arkas"],
    "kurir_pengiriman": ["ubah kurir", "ubah jasa kirim", "jasa pengiriman", "jasa kurir"],
    "status": ["cek"],
    "bantuan_umum": ["ijin tanya", "minta tolong", "tidak bisa", "cara", "masalah", "mau tanya", "input", "pkp", "pesanan gantung", "di luar dari arkas", "di bayar dari"],
    "lainnya": []
}

def assign_by_topic_centroid(unclassified_texts, unclassified_indices, topik_per_pertanyaan, sentence_model,
                             sim_threshold=CENTROID_SIMILARITY_THRESHOLD, min_examples=MIN_EXAMPLES_FOR_CENTROID):
    labeled_by_topic = defaultdict(list)
    for item in topik_per_pertanyaan:
        if item['topik'] != 'lainnya':
            labeled_by_topic[item['topik']].append(item['pertanyaan'])
    topics_list, centroids = [], []
    for topic, texts in labeled_by_topic.items():
        if len(texts) >= min_examples:
            emb = embed_texts(texts, sentence_model)
            if emb.shape[0] > 0:
                centroids.append(emb.mean(axis=0))
                topics_list.append(topic)
    if not centroids:
        return {}, list(range(len(unclassified_texts)))
    centroid_matrix = np.vstack(centroids)
    u_emb = embed_texts(unclassified_texts, sentence_model)
    if u_emb.shape[0] == 0:
        return {}, list(range(len(unclassified_texts)))
    sims = np.dot(u_emb, centroid_matrix.T)
    assigned, remaining_local_idxs = {}, []
    for i in range(len(unclassified_texts)):
        best_j = int(np.argmax(sims[i]))
        best_sim = float(sims[i, best_j])
        if best_sim >= sim_threshold:
            assigned[i] = {'topic': topics_list[best_j], 'score': best_sim}
        else:
            remaining_local_idxs.append(i)
    return assigned, remaining_local_idxs

def analyze_all_topics(df_questions: pd.DataFrame, sentence_model):
    if df_questions is None or df_questions.empty:
        st.warning("Tidak ada data pertanyaan yang bisa dianalisis.")
        return

    topik_counter = Counter()
    topik_per_pertanyaan = []
    unclassified_texts, unclassified_indices = [], []

    for idx, row in df_questions.iterrows():
        text = row.get('text', '')
        if pd.isna(text) or not isinstance(text, str):
            continue
        text_lc = text.lower()
        found_topik = []
        for topik, patterns in topik_keywords.items():
            if isinstance(patterns, list) and patterns and isinstance(patterns[0], list):
                if any(all(p in text_lc for p in group) for group in patterns):
                    found_topik.append(topik)
            else:
                if isinstance(patterns, list) and any(p in text_lc for p in patterns):
                    found_topik.append(topik)
        if not found_topik:
            unclassified_texts.append(text)
            unclassified_indices.append(idx)
            selected_topik = 'lainnya'
        else:
            spesifik_topik = [t for t in found_topik if t != 'bantuan_umum']
            selected_topik = spesifik_topik[0] if spesifik_topik else found_topik[0]
        topik_counter[selected_topik] += 1
        topik_per_pertanyaan.append({'topik': selected_topik, 'pertanyaan': text, 'index': idx})

    # Centroid assignment
    assigned_map = {}
    remaining_local_idxs = list(range(len(unclassified_texts)))
    if unclassified_texts:
        try:
            assigned_map, remaining_local_idxs = assign_by_topic_centroid(
                unclassified_texts, unclassified_indices, topik_per_pertanyaan, sentence_model,
                sim_threshold=CENTROID_SIMILARITY_THRESHOLD, min_examples=MIN_EXAMPLES_FOR_CENTROID)
        except Exception:
            assigned_map, remaining_local_idxs = {}, list(range(len(unclassified_texts)))

    if assigned_map:
        for local_i, info in assigned_map.items():
            orig_idx = unclassified_indices[local_i]
            topic = info['topic']
            if topik_counter.get('lainnya', 0) > 0:
                topik_counter['lainnya'] -= 1
            topik_counter[topic] += 1
            for item in topik_per_pertanyaan:
                if item['index'] == orig_idx:
                    item['topik'] = topic
                    break

    # Clustering semantik otomatis
    remaining_texts = [unclassified_texts[i] for i in remaining_local_idxs]
    remaining_indices = [unclassified_indices[i] for i in remaining_local_idxs]

    new_topics_found = {}
    if len(remaining_texts) >= MIN_CLUSTER_SHOW:
        st.subheader("🔍 Mendeteksi Topik Baru")
        with st.spinner("Menghitung embedding & clustering..."):
            labels, cluster_keywords, cluster_example, silhouette, X_umap = semantic_clustering_auto(
                remaining_texts, sentence_model
            )
        if labels is not None and len(labels) == len(remaining_texts):
            cluster_counts = Counter(labels)
            for cluster_id, count in cluster_counts.items():
                if cluster_id == -1:
                    continue

                keywords = cluster_keywords.get(cluster_id, [])
                main_keyword = get_main_keyword(keywords, stop_words_id, top_n=2)
                main_keyword = clean_main_keyword(main_keyword)

                # fallback: coba ambil keyword dari contoh teks cluster
                if not main_keyword:
                    example_text = cluster_example.get(cluster_id, "")
                    kws = extract_top_keywords_from_texts([example_text], top_n=2)
                    main_keyword = get_main_keyword(kws, stop_words_id, top_n=2)
                    main_keyword = clean_main_keyword(main_keyword)

                # kalau masih kosong → balikin ke 'lainnya' saja
                if not main_keyword:
                    for local_i, label in enumerate(labels):
                        if label == cluster_id:
                            orig_idx = remaining_indices[local_i]
                            for item in topik_per_pertanyaan:
                                if item['index'] == orig_idx:
                                    item['topik'] = "lainnya"
                    continue  # skip bikin topik baru utk cluster ini

                # baru bikin nama kalau keyword valid
                new_topic_name = f"(new) {main_keyword}"

                new_topics_found[cluster_id] = {
                    'name': new_topic_name,
                    'keywords': keywords,
                    'count': count,
                    'texts': [],
                    'example': cluster_example.get(cluster_id, "")
                }

            for local_i, label in enumerate(labels):
                if label in new_topics_found:
                    orig_idx = remaining_indices[local_i]
                    orig_text = remaining_texts[local_i]
                    new_topics_found[label]['texts'].append(orig_text)
                    topik_counter[new_topics_found[label]['name']] += 1
                    for item in topik_per_pertanyaan:
                        if item['index'] == orig_idx and item['topik'] == 'lainnya':
                            item['topik'] = new_topics_found[label]['name']
                    if topik_counter.get('lainnya', 0) > 0:
                        topik_counter['lainnya'] -= 1
                        
            if new_topics_found:
                st.success(f"✅ Ditemukan {len(new_topics_found)} potensi topik baru!")
    else:
        st.info(f"Jumlah teks yang tidak terklasifikasi ({len(remaining_texts)}) kurang untuk clustering.")

    # Update ulang counter biar sinkron
    topik_counter = Counter([item['topik'] for item in topik_per_pertanyaan])

    # Ringkasan akhir
    st.subheader("📊 Ringkasan Topik Teratas")
    if not topik_counter:
        st.write("Tidak ada topik yang teridentifikasi.")
        return

    summary_data = pd.DataFrame([{"Topik": t, "Jumlah Pertanyaan": c} for t, c in topik_counter.most_common()])
    st.dataframe(summary_data, use_container_width=True)

    st.subheader("📝 Detail Pertanyaan per Topik")
    mapping = defaultdict(list)
    for item in topik_per_pertanyaan:
        mapping[item['topik']].append(item['pertanyaan'])
    for topik, count in topik_counter.most_common():
        with st.expander(f"Topik: {topik} ({count} pertanyaan)"):
            for q in mapping.get(topik, [])[:200]:
                st.markdown(f"- *{q.strip()}*")
