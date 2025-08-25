import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# === Embedding & helper ===
def extract_top_keywords_from_texts(texts, top_n=2, max_features=1000):
    if not texts:
        return []
    try:
        vectorizer = TfidfVectorizer(
            max_features=max_features, 
            stop_words=list(stop_words_id), 
            ngram_range=(1,2)
            )
        X = vectorizer.fit_transform(texts)
        scores = np.asarray(X.sum(axis=0)).ravel()
        features = vectorizer.get_feature_names_out()
        top_idx = np.argsort(scores)[::-1][:top_n]
        return [features[i] for i in top_idx]
    except Exception:
        return []

def embed_texts(texts, sentence_model):
    if not texts:
        return np.zeros((0, sentence_model.get_sentence_embedding_dimension()))
    preproc = [clean_text_for_clustering(t) for t in texts]
    emb = sentence_model.encode(preproc, show_progress_bar=False)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = emb / norms
    return emb

# === Auto params heuristics ===
def _auto_umap_params(n_texts: int):
    if n_texts <= 50:
        return dict(n_neighbors=10, n_components=5, metric='cosine', random_state=42)
    if n_texts <= 200:
        return dict(n_neighbors=12, n_components=5, metric='cosine', random_state=42)
    return dict(n_neighbors=min(15, max(8, n_texts//20)), n_components=5, metric='cosine', random_state=42)

def _auto_hdbscan_params(n_texts: int):
    mcs = max(3, n_texts // 40)  
    return dict(
        min_cluster_size=mcs,
        min_samples=max(2, mcs // 2),   
        metric="euclidean",
        cluster_selection_method="eom"
    )

def _merge_similar_clusters(embeddings: np.ndarray, labels: np.ndarray, sim_threshold: float = MERGE_CLUSTER_SIM_THRESHOLD) -> np.ndarray:
    uniq = sorted(set(labels) - {-1})
    if len(uniq) <= 1:
        return labels
    centroids = {lab: embeddings[labels == lab].mean(axis=0) for lab in uniq}
    merged_to = {}
    for i, l1 in enumerate(uniq):
        for l2 in uniq[i+1:]:
            sim = float(cosine_similarity([centroids[l1]], [centroids[l2]])[0][0])
            if sim >= sim_threshold:
                rep = min(l1, l2)
                other = max(l1, l2)
                merged_to[other] = rep
    def find_rep(x):
        while x in merged_to:
            x = merged_to[x]
        return x
    new_labels = labels.copy()
    for idx, lab in enumerate(labels):
        if lab == -1:
            continue
        new_labels[idx] = find_rep(lab)
    return new_labels

def _contiguous_labels(labels: np.ndarray) -> np.ndarray:
    mapping, next_id = {}, 0
    new = labels.copy()
    for i, lab in enumerate(labels):
        if lab == -1:
            new[i] = -1
            continue
        if lab not in mapping:
            mapping[lab] = next_id
            next_id += 1
        new[i] = mapping[lab]
    return new

# === Clustering utama (auto-optimal) ===
def semantic_clustering_auto(texts, sentence_model):
    if not texts:
        return None, {}, {}, None, None
    # Stopwords gabungan (ID + EN + custom)
    try:
        stopwords_id = set(stopwords.words('indonesian'))
    except LookupError:
        nltk.download('stopwords')
        stopwords_id = set(stopwords.words('indonesian'))
    stopwords_en = set(stopwords.words('english'))
    stopwords_custom = {'yg','ya','gak','nggak','banget','nih','sih','dong','aja','kayak','dulu','udah','lagi','bisa','akan','sama','ke','di','itu','ini'}
    all_stopwords = list(stopwords_id | stopwords_en | stopwords_custom)
    embeddings = sentence_model.encode(texts, show_progress_bar=False)

    # Dimensionality reduction (UMAP bila ada)
    if UMAP_AVAILABLE and embeddings.shape[0] >= 5:
        params_umap = _auto_umap_params(len(texts))
        reducer = umap.UMAP(**params_umap)
        X_umap = reducer.fit_transform(embeddings)
        X_for_cluster = X_umap
    else:
        X_umap = None
        X_for_cluster = embeddings
    labels = None
  
    if HDBSCAN_AVAILABLE and len(texts) >= 5:
        params_hdb = _auto_hdbscan_params(len(texts))
        clusterer = hdbscan.HDBSCAN(**params_hdb)
        labels = clusterer.fit_predict(X_for_cluster)
        # Merge clusters mirip
        labels = _merge_similar_clusters(embeddings, labels, MERGE_CLUSTER_SIM_THRESHOLD)
        labels = _contiguous_labels(labels)
    else:
        # Fallback: KMeans dengan auto-k pakai silhouette
        max_k = min(20, max(2, len(texts)//5))  # lebih fleksibel
        best_k, best_score, best_labels = 2, -1.0, None
        for k in range(2, max_k+1):
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            lbl = km.fit_predict(X_for_cluster)
            try:
                score = silhouette_score(X_for_cluster, lbl)
            except Exception:
                score = -1.0
            if score > best_score:
                best_k, best_score, best_labels = k, score, lbl
        labels = best_labels if best_labels is not None else np.zeros(len(texts), dtype=int)

    # Keyword & contoh per cluster
    cluster_keywords, cluster_example = {}, {}
    cluster_texts_map = defaultdict(list)

    for idx, label in enumerate(labels):
        if label == -1:
            continue
        cluster_texts_map[label].append(texts[idx])

    for cluster_id, cluster_texts in cluster_texts_map.items():
        if len(cluster_texts) < 2:
            cluster_keywords[cluster_id] = extract_top_keywords_from_texts(cluster_texts, top_n=2)
            cluster_example[cluster_id] = cluster_texts[0]
            continue
        vectorizer = TfidfVectorizer(max_features=2000, stop_words=all_stopwords, ngram_range=(1,2), min_df=1)
        tfidf_matrix = vectorizer.fit_transform(cluster_texts)
        feature_names = vectorizer.get_feature_names_out()
        mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        sorted_idx = np.argsort(mean_tfidf)[::-1]
        top_keywords = [feature_names[i] for i in sorted_idx[:5]]
        cluster_keywords[cluster_id] = top_keywords

        # contoh representatif (paling dekat ke centroid embedding global)
        idxs = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
        cluster_embeddings = embeddings[idxs]
        centroid = np.mean(cluster_embeddings, axis=0)
        sims = np.dot(cluster_embeddings, centroid) / (np.linalg.norm(cluster_embeddings, axis=1) * np.linalg.norm(centroid) + 1e-9)
        best_local = int(np.argmax(sims))
        cluster_example[cluster_id] = texts[idxs[best_local]]

    silhouette = None
    try:
        valid_labels = [lbl for lbl in labels if lbl != -1]
        if len(set(valid_labels)) > 1:
            silhouette = silhouette_score(X_for_cluster, labels)
    except Exception:
        pass

    return labels, cluster_keywords, cluster_example, silhouette, X_umap

def split_stuck_words(word: str) -> str:
    patterns = [
        (r'(dibatalkan)(retur)', r'\1 \2'),
        (r'(retur)(dibatalkan)', r'\1 \2'),
        (r'(bank)(terpotong)', r'\1 \2'),
        (r'(terpotong)(bank)', r'\1 \2'),
        (r'(verifikasi)(pembayaran)', r'\1 \2'),
        (r'(pembayaran)(dana)', r'\1 \2')
    ]
    for pat, repl in patterns:
        word = re.sub(pat, repl, word)
    return word

def get_main_keyword(keywords, stop_words_id, top_n=2):
    if not keywords:
        return None

    generic_words = {
            "baru", "topik", "ready", "izin", "silakan", "hai", "halo", 
            "tolong", "bantu", "tanya", "nanya", "kak", "min", "gan", "om",
            "pak", "bu", "cs", "tokla", "customer", "service", "baru","topik",
            "up","ngalami","proses","info","solusi","kendala","admin","toko",
            "tokla","pakai","bantu","mohon","permohonan","data", "ya","loh",
            "kok","tahi","anjing","brow","wkwk","haha","heh","hmm","mantap",
            "sip","udah","dong","nih"
            }

    tokens = []
    for kw in keywords:
        if not kw:
            continue
        kw_clean = kw.strip().lower()
        kw_clean = split_stuck_words(kw_clean)

        for tok in kw_clean.split():
            if (tok not in stop_words_id and 
                tok not in generic_words and 
                len(tok) > 2):
                tokens.append(tok)

    tokens = list(dict.fromkeys(tokens))
    if not tokens:
        return keywords[0].lower()
    tokens = tokens[:top_n]
    return "_".join(tokens)

def clean_main_keyword(keyword: str) -> str:
    if not keyword:
        return ""
    keyword = keyword.lower().strip()
    keyword = re.sub(r'@\w+', '', keyword)
    keyword = re.sub(r'\b(cs[_-]*\w+)\b', '', keyword)
    keyword = re.sub(r'[^a-z]', ' ', keyword)
    tokens = keyword.split()
    tokens = list(dict.fromkeys(tokens))
    if not tokens:
        return ""
    return "_".join(tokens)
