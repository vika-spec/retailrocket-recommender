import numpy as np
import torch
from pathlib import Path


# ---------------------------- BASE CONFIG ---------------------------------
BASE_CONFIG = {
    # Model Architecture
    'hidden_size': 224,
    'num_layers': 2,
    'dropout': 0.3,
    'embedding_dim': 64,
    'event_embedding_dim': 8,
    
    # Training
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'batch_size': 1024,
    'gradient_clip': 1.0,
    'label_smoothing': 0.0,
    'use_scheduler': True,
    'patience': 5,
    'epochs': 20,
    
    # Data
    'max_seq_len': 50,
    'test_split_ratio': 0.2,
    'min_train_interactions': 5,
    'num_candidates': None,
    'eval_metrics_k': [1, 3, 5, 10, 20],
    'seed': 42,
    'val_ratio': 0.1,
    'num_events': 4,
    'session_eval_batch_size': 1024,
}

# ---------------------------- PATHS ---------------------------------
# FIXED: Use .resolve() for absolute paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
OUTPUT_DIR = BASE_DIR / 'outputs_official_final'
MODEL_DIR = BASE_DIR / 'models_official_final'

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# RAW DATA FILES
# ============================================================
EVENTS_FILE = RAW_DATA_DIR / 'events.csv'
ITEM_PROPERTIES_1 = RAW_DATA_DIR / 'item_properties_part1.csv'
ITEM_PROPERTIES_2 = RAW_DATA_DIR / 'item_properties_part2.csv'
CATEGORY_TREE_FILE = RAW_DATA_DIR / 'category_tree.csv'

# ============================================================
# PROCESSED DATA FILES
# ============================================================

# --- Training data ---
TRAIN_DIR = PROCESSED_DATA_DIR / 'train'
TRAIN_RAW_EVENTS = TRAIN_DIR / 'raw_events.parquet'
TRAIN_PROCESSED_EVENTS = TRAIN_DIR / 'processed_events.parquet'
TRAIN_USER_FEATURES = TRAIN_DIR / 'user_features.parquet'
TRAIN_ITEM_FEATURES = TRAIN_DIR / 'item_features.parquet'
TRAIN_SESSIONS = TRAIN_DIR / 'sessions.parquet'
TRAIN_INTERACTION_MATRIX = TRAIN_DIR / 'interaction_matrix.npz'
TRAIN_PMI_MATRIX = TRAIN_DIR / 'pmi_matrix.npz'

# --- Metadata ---
TRAIN_ITEM_ENCODER = TRAIN_DIR / 'item_encoder.pkl'
TRAIN_ITEM_IDS = TRAIN_DIR / 'item_ids.pkl'
TRAIN_GAP_MAX = TRAIN_DIR / 'gap_max.pkl'
TRAIN_LE_USER = TRAIN_DIR / 'le_user.pkl'
TRAIN_LE_ITEM = TRAIN_DIR / 'le_item.pkl'

# --- Test data ---
TEST_DIR = PROCESSED_DATA_DIR / 'test'
TEST_SESSIONS = TEST_DIR / 'sessions.parquet'
TEST_RAW_EVENTS = TEST_DIR / 'raw_events.parquet'

# ============================================================
# BACKWARDS COMPATIBILITY
# ============================================================
PROCESSED_EVENTS = TRAIN_PROCESSED_EVENTS
USER_FEATURES = TRAIN_USER_FEATURES
ITEM_FEATURES = TRAIN_ITEM_FEATURES
SESSION_SEQUENCES = TRAIN_SESSIONS
INTERACTION_MATRIX = TRAIN_INTERACTION_MATRIX
RAW_EVENTS = TRAIN_RAW_EVENTS

# ---------------------------- DATA PROCESSING ---------------------------------
EVENT_WEIGHTS = {
    "view": 1,
    "addtocart": 3,
    "transaction": 5,
}
SESSION_GAP_SECONDS = 3600
MIN_SESSION_LENGTH = 1
MAX_SESSION_LENGTH = 50
MIN_EVENTS_PER_USER = 2

# ---------------------------- RANDOM SEED ---------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ---------------------------- ALIASES FOR BACKWARDS COMPATIBILITY ----------
MODEL_CONFIG = BASE_CONFIG
CONFIG = BASE_CONFIG