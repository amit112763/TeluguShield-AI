from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ============ MODEL PATHS ============
MBERT_MODEL_ID = "ramesh070/mbert-hatespeechdetection-telugu"

# Hugging Face repo IDs (work on laptop and on Streamlit Cloud).
# If a local folder with the same path exists, app.py uses that instead.
MURIL_LOCAL_PATH      = "akay1127/telugushield-muril"
TELUGUBERT_LOCAL_PATH = "akay1127/telugushield-telugubert"

# ============ DATASET ============
TEST_DATA_PATH = str(BASE_DIR / "telugu_hate_test_final.csv")
TEXT_COLUMN    = "Comments"
LABEL_COLUMN   = "Final_Label"

# ============ LABELS ============
LABEL_MAP = {"non-hate": 0, "hate": 1}
ID2LABEL  = {0: "NON-HATE", 1: "HATE"}

# ============ INFERENCE ============
MAX_LENGTH   = 128
IG_STEPS     = 32
LIME_SAMPLES = 100

# ============ CACHE ============
EVAL_CACHE_PATH = str(BASE_DIR / "prototype_eval_cache.json")