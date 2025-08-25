from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Konstanta global & konfigurasi
TIMEZONE = "Asia/Jakarta"

CENTROID_SIMILARITY_THRESHOLD = 0.8
MIN_EXAMPLES_FOR_CENTROID = 2
MIN_CLUSTER_SHOW = 2
MERGE_CLUSTER_SIM_THRESHOLD = 0.9

# Stopwords Indonesia + tambahan
stop_words_id = set()
try:
    stop_words_id = set(stopwords.words('indonesian'))
except Exception:
    stop_words_id = set(['yang','dan','di','ke','dari','ini','itu','ada','untuk','dengan','sudah','belum','bisa','tidak'])

tambahan_stop_words = [
    'yg', 'ga', 'gak', 'gk', 'ya', 'dong', 'sih', 'aja', 'kak', 'min', 'gan', 'om', 'pak', 'bu',
    'mohon', 'bantu', 'tolong', 'bantuan', 'solusi', 'tanya', 'nanya', 'bertanya', 'mas',
    'gimana', 'bagaimana', 'kenapa', 'mengapa', 'apa', 'apakah', 'kapan', 'siapa', 'kah',
    'nya', 'nih', 'tuh', 'deh', 'kok', 'kek', 'admin', 'customer', 'service', 'cs', 'halo',
    'terima', 'kasih', 'assalamualaikum', 'waalaikumsalam', 'saya', 'aku', 'anda', 'kami', 'kita',
    'ada', 'ini', 'itu', 'di', 'ke', 'dari', 'dan', 'atau', 'tapi', 'untuk', 'dengan', 'sudah', 'belum',
    'po', 'kode', 'nomor', 'nama', 'toko', 'masuk', 'jam', 'website', 'admin', 'min', 'bapak', 'ibu',
    'pak', 'bu', 'kak', 'halo', 'selamat', 'malam', 'pagi', 'siang', 'sore', 'langsung', 'kenapa',
    'apa', 'siapa', 'berapa', 'dimana', 'bagaimana', 'tu', 'kalo', 'knpa', 'izin', 'promo',
    'produk', 'sekolah', 'input', 'pesanan', 'gagal', 'error', 'trouble', 'maintenance', 'rekening',
    'akun', 'email', 'no', 'hp', 'telepon', 'reset', 'ganti', 'ubah', 'edit', 'status', 'umkm', 'pkp',
    'modal', 'kerja', 'usaha', 'bantuan', 'penyaluran', 'transfer', 'bayar', 'pembayaran', 'dana',
    'uang', 'notif', 'expired', 'pengiriman', 'barang', 'rusak', 'hilang', 'diproses', 'laporan',
    'tanggal', 'dokumen', 'terhubung', 'arkas', 'jasa', 'kurir', 'cek', 'ijin', 'masalah', 'ka','min min'
]
stop_words_id.update(tambahan_stop_words)
stemmer = PorterStemmer()

custom_stopwords = {
    "ready", "izin", "siang", "silakan", "tolong", "kak", "min", 
    "mohon", "terima", "kasih", "minta", "halo", "ya", "oke"
}
