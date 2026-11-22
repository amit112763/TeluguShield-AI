from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ============ MODEL PATHS ============
MBERT_MODEL_ID = "ramesh070/mbert-hatespeechdetection-telugu"

MURIL_LOCAL_PATH = str(BASE_DIR / "MuRIL_Best_Model" / "Best_Model")
# TeluguBERT ka asli folder mil jaye tab yahan path daalna
TELUGUBERT_LOCAL_PATH = str(BASE_DIR / "TeluguBERT_Best_Model")

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