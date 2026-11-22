# ============ MODEL PATHS ============
MBERT_MODEL_ID   = "ramesh070/mbert-hatespeechdetection-telugu"

# Laptop ke local paths (Dono models ke folders)
MURIL_LOCAL_PATH = r"C:\Users\amit\Downloads\TS_extract\TeluguShield_AI\05_Hyperparameter_Tuning\MuRIL\Best_Model"
TELUGUBERT_LOCAL_PATH = r"E:\Telugushield\TeluguBERT\Best_Model"

# ============ DATASET ============
TEST_DATA_PATH = "./telugu_hate_test_final.csv" 
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
EVAL_CACHE_PATH = "./prototype_eval_cache.json"