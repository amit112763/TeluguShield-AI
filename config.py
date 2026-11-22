# ============ MODEL PATHS ============
MBERT_MODEL_ID   = "ramesh070/mbert-hatespeechdetection-telugu"

# Laptop aur Cloud dono ke liye relative paths
MURIL_LOCAL_PATH      = "./Best_Model"
TELUGUBERT_LOCAL_PATH = "./TeluguBERT_Best_Model" # Agar folder ka naam yahi hai

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