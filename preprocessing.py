
"""
Preprocessing Pipeline for RetailRocket Dataset
===============================================
- Loads raw events, item properties, category tree
- Splits events temporally (80% train, 20% test) BEFORE any processing
- Builds sessions, item features, user features from training data only
- Saves training artefacts to data/processed/train/
- Saves test sessions to data/processed/test/ (filtered to known items)
- No data leakage: item encoder, gap_max, features computed from training only

Usage:
    python scripts/preprocessing.py
"""

import os
import logging
import warnings
import pickle
import random
from collections import deque
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    EVENTS_FILE, ITEM_PROPERTIES_1, ITEM_PROPERTIES_2, CATEGORY_TREE_FILE,
    EVENT_WEIGHTS, SESSION_GAP_SECONDS, MIN_SESSION_LENGTH, MAX_SESSION_LENGTH,
    OUTPUT_DIR, RANDOM_SEED, TRAIN_DIR, TEST_DIR,
    TRAIN_RAW_EVENTS, TRAIN_PROCESSED_EVENTS, TRAIN_USER_FEATURES,
    TRAIN_ITEM_FEATURES, TRAIN_SESSIONS, TRAIN_INTERACTION_MATRIX,
    TRAIN_PMI_MATRIX, TRAIN_ITEM_ENCODER, TRAIN_ITEM_IDS, TRAIN_GAP_MAX,
    TEST_SESSIONS
)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Raw data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_events(path: str) -> pd.DataFrame:
    log.info("Loading events ...")
    df = pd.read_csv(
        path,
        dtype={"visitorid": "int32", "itemid": "int32", "event": "category"},
        engine="c",
    )
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"timestamp", "visitorid", "event", "itemid"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"events.csv missing columns: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["event"] = df["event"].astype(str).str.strip().str.lower()
    log.info(f"  {len(df):,} events | users: {df['visitorid'].nunique():,} | items: {df['itemid'].nunique():,}")
    return df


def load_item_properties(path1: str, path2: str) -> pd.DataFrame:
    log.info("Loading item properties ...")
    p1 = pd.read_csv(path1, dtype={"itemid": "int32"}, engine="c")
    p2 = pd.read_csv(path2, dtype={"itemid": "int32"}, engine="c")
    df = pd.concat([p1, p2], ignore_index=True)
    df.columns = [c.strip().lower() for c in df.columns]
    log.info(f"  {len(df):,} property rows | {df['itemid'].nunique():,} items")
    return df


def load_category_tree(path: str) -> pd.DataFrame:
    log.info("Loading category tree ...")
    df = pd.read_csv(path, engine="c")
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"categoryid": "category_id", "parentid": "parent_id"})
    df["category_id"] = df["category_id"].fillna(-1).astype(int)
    df["parent_id"] = df["parent_id"].fillna(-1).astype(int)
    log.info(f"  {len(df):,} category nodes")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Event cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_events(df: pd.DataFrame):
    log.info("Cleaning events ...")
    before = len(df)
    df = df[df["event"].isin(EVENT_WEIGHTS)].copy()
    df = df.drop_duplicates(subset=["visitorid", "itemid", "event", "timestamp"])
    df["confidence"] = df["event"].map(EVENT_WEIGHTS)
    raw_clean = df.copy()
    log.info(f"  Raw events after cleaning: {len(raw_clean):,}")
    agg = df.sort_values("timestamp").groupby(["visitorid", "itemid"], as_index=False).agg(
        confidence=("confidence", "sum"),
        first_ts=("timestamp", "min"),
        last_ts=("timestamp", "max"),
        n_events=("event", "count"),
    )
    log.info(f"  {before:,} -> {len(agg):,} user-item pairs")
    return raw_clean, agg


def add_time_decay(agg_df: pd.DataFrame, half_life_days: float = 30.0) -> pd.DataFrame:
    log.info("Applying time-decay ...")
    reference_ts = agg_df["last_ts"].max()
    days = (reference_ts - agg_df["last_ts"]).dt.total_seconds() / 86400.0
    days = days.clip(lower=0)
    decay = np.exp(-np.log(2) * days / half_life_days)
    agg_df["confidence_decayed"] = agg_df["confidence"] * decay
    return agg_df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Item features
# ─────────────────────────────────────────────────────────────────────────────

def _compute_category_depth(tree: pd.DataFrame) -> dict:
    parent_map = dict(zip(tree["category_id"], tree["parent_id"]))
    children_map = {}
    for cid, pid in parent_map.items():
        children_map.setdefault(pid, []).append(cid)
    depth_map = {}
    roots = [cid for cid, pid in parent_map.items() if pid == -1]
    queue = deque((r, 0) for r in roots)
    while queue:
        node, d = queue.popleft()
        depth_map[node] = d
        for child in children_map.get(node, []):
            queue.append((child, d + 1))
    return depth_map


def build_item_features(raw_events: pd.DataFrame, props: pd.DataFrame, tree: pd.DataFrame) -> pd.DataFrame:
    log.info("Building item features ...")
    props = props.copy()
    props.columns = [c.lower() for c in props.columns]
    props["value"] = props["value"].astype(str)

    price_df = props[props["property"] == "790"][["itemid", "value"]].copy()
    price_df["price_raw"] = pd.to_numeric(
        price_df["value"].str.extract(r"([\d.]+)", expand=False),
        errors="coerce"
    )
    price_df = price_df.dropna(subset=["price_raw"])
    if len(price_df) > 0:
        upper_bound = price_df["price_raw"].quantile(0.995)
        price_df = price_df[price_df["price_raw"] <= upper_bound]
        price_log = np.log1p(price_df["price_raw"])
        price_mean = price_log.mean()
        price_std = price_log.std() + 1e-8
        price_df["price_norm"] = (price_log - price_mean) / price_std
    else:
        price_df["price_norm"] = 0.0
    price_df = price_df[["itemid", "price_norm"]].drop_duplicates("itemid")

    cat_df = props[props["property"] == "categoryid"][["itemid", "value"]].drop_duplicates("itemid")
    cat_df["categoryid"] = pd.to_numeric(cat_df["value"], errors="coerce").fillna(-1).astype(int)
    cat_df = cat_df[["itemid", "categoryid"]]

    item_df = pd.DataFrame({"itemid": raw_events["itemid"].unique()})
    item_df = item_df.merge(cat_df, on="itemid", how="left")
    item_df = item_df.merge(price_df, on="itemid", how="left")
    item_df["categoryid"] = item_df["categoryid"].fillna(-1).astype(int)
    item_df["price_norm"] = item_df["price_norm"].fillna(0.0)

    depth_map = _compute_category_depth(tree)
    item_df["category_depth"] = item_df["categoryid"].map(depth_map).fillna(0).astype(int)

    pop = raw_events.groupby(["itemid", "event"]).size().unstack(fill_value=0)
    pop = pop.reindex(columns=["view", "addtocart", "transaction"], fill_value=0)
    pop.columns = ["n_views", "n_carts", "n_purchases"]
    pop["n_total"] = pop.sum(axis=1)
    pop["purchase_rate"] = pop["n_purchases"] / pop["n_views"].clip(lower=1)
    pop["log_views"] = np.log1p(pop["n_views"])
    pop["log_purchases"] = np.log1p(pop["n_purchases"])
    item_df = item_df.merge(pop.reset_index(), on="itemid", how="left").fillna(0)

    item_df["category_pop_rank"] = (
        item_df.groupby("categoryid")["n_total"]
        .rank(ascending=False, method="min")
        .fillna(999)
        .astype(int)
    )

    keep_cols = [
        "itemid", "categoryid", "category_depth", "price_norm",
        "log_views", "log_purchases", "purchase_rate", "category_pop_rank"
    ]
    item_df = item_df[keep_cols]
    log.info(f"  Item features: {len(item_df):,} items, {len(keep_cols)} columns")
    return item_df


# ─────────────────────────────────────────────────────────────────────────────
# 4. User features
# ─────────────────────────────────────────────────────────────────────────────

def build_user_features(raw_events: pd.DataFrame) -> pd.DataFrame:
    log.info("Building user features ...")
    ref_ts = raw_events["timestamp"].max()
    agg = raw_events.groupby("visitorid").agg(
        first_seen=("timestamp", "min"),
        last_seen=("timestamp", "max"),
        n_total_events=("itemid", "count"),
        n_unique_items=("itemid", "nunique"),
    )
    ev = raw_events.groupby(["visitorid", "event"]).size().unstack(fill_value=0)
    ev = ev.reindex(columns=["view", "addtocart", "transaction"], fill_value=0)
    agg["n_carts"] = ev["addtocart"]
    agg["n_purchases"] = ev["transaction"]
    agg["days_since_last"] = (ref_ts - agg["last_seen"]).dt.days
    agg["purchase_rate"] = agg["n_purchases"] / agg["n_total_events"].clip(lower=1)
    agg["cart_rate"] = agg["n_carts"] / agg["n_total_events"].clip(lower=1)
    agg["log_total_events"] = np.log1p(agg["n_total_events"])
    agg["log_unique_items"] = np.log1p(agg["n_unique_items"])
    keep_cols = [
        "visitorid", "days_since_last", "n_total_events", "n_unique_items",
        "n_purchases", "n_carts", "purchase_rate", "cart_rate",
        "log_total_events", "log_unique_items"
    ]
    user_df = agg.reset_index()[keep_cols]
    log.info(f"  User features: {len(user_df):,} users, {len(keep_cols)} columns")
    return user_df


# ─────────────────────────────────────────────────────────────────────────────
# 5. Session building
# ─────────────────────────────────────────────────────────────────────────────

def build_sessions(raw_events: pd.DataFrame) -> pd.DataFrame:
    log.info("Building sessions with time features ...")
    df = raw_events.sort_values(["visitorid", "timestamp"]).copy()

    event_map = {'view': 1, 'addtocart': 1, 'transaction': 2}
    df['event_enc'] = df['event'].map(event_map).fillna(0).astype(int)

    gap = df.groupby("visitorid")["timestamp"].diff().dt.total_seconds()
    new_session = gap.isna() | (gap > SESSION_GAP_SECONDS)
    df["session_id"] = df["visitorid"].astype(str) + "_" + new_session.groupby(df["visitorid"]).cumsum().astype(str)

    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.dayofweek
    df['gap_seconds'] = gap.fillna(0)

    sessions = df.groupby("session_id").agg(
        visitorid=("visitorid", "first"),
        session_start=("timestamp", "min"),
        session_end=("timestamp", "max"),
        item_sequence=("itemid", list),
        event_sequence=("event_enc", list),
        hour_sequence=("hour", list),
        day_sequence=("day", list),
        time_gap_sequence=("gap_seconds", list),
        n_events=("itemid", "count"),
    ).reset_index()

    sessions = sessions[
        (sessions["n_events"] >= MIN_SESSION_LENGTH) &
        (sessions["n_events"] <= MAX_SESSION_LENGTH)
    ]
    for col in ['item_sequence', 'event_sequence', 'hour_sequence', 'day_sequence', 'time_gap_sequence']:
        sessions[col] = sessions[col].apply(lambda x: x[-MAX_SESSION_LENGTH:])

    log.info(f"  {len(sessions):,} sessions (min length ≥ {MIN_SESSION_LENGTH})")
    return sessions


# ─────────────────────────────────────────────────────────────────────────────
# 6. Interaction matrix & PMI
# ─────────────────────────────────────────────────────────────────────────────

def build_interaction_matrix(agg_df: pd.DataFrame):
    log.info("Building sparse interaction matrix ...")
    le_user = LabelEncoder()
    le_item = LabelEncoder()
    user_idx = le_user.fit_transform(agg_df["visitorid"])
    item_idx = le_item.fit_transform(agg_df["itemid"])
    values = np.log1p(agg_df["confidence_decayed"].astype(np.float32))
    matrix = csr_matrix((values, (user_idx, item_idx)), dtype=np.float32)
    log.info(f"  Matrix shape: {matrix.shape}")
    return matrix, le_user, le_item


def build_pmi_cooccurrence(sessions: pd.DataFrame, n_items: int, item_encoder: LabelEncoder):
    from scipy.sparse import coo_matrix, csr_matrix

    log.info("Building PMI co-occurrence (optimised sparse)...")
    enc_lookup = {iid: idx for idx, iid in enumerate(item_encoder.classes_)}

    exploded = sessions[['session_id', 'item_sequence']].explode('item_sequence')
    exploded['item_enc'] = exploded['item_sequence'].map(enc_lookup)
    exploded = exploded.dropna(subset=['item_enc'])
    exploded['item_enc'] = exploded['item_enc'].astype(int)

    session_ids = exploded['session_id'].unique()
    session_map = {sid: i for i, sid in enumerate(session_ids)}
    exploded['session_idx'] = exploded['session_id'].map(session_map)

    n_sessions = len(session_ids)

    F = coo_matrix(
        (np.ones(len(exploded), dtype=np.float32),
         (exploded['session_idx'], exploded['item_enc'])),
        shape=(n_sessions, n_items)
    ).tocsr()

    item_counts = np.array(F.sum(axis=0)).flatten().astype(np.float32)
    C_full = F.T @ F

    diag_orig = (C_full.diagonal() - item_counts) / 2.0
    diag_orig = np.maximum(diag_orig, 0)
    C_full.setdiag(diag_orig)

    cooc = C_full.tocoo()
    total = cooc.data.sum()

    if total == 0:
        log.warning("No co-occurrence pairs found.")
        return csr_matrix((n_items, n_items), dtype=np.float32)

    p_i = item_counts / (total + 1e-9)
    pmi_v = np.log((cooc.data / total) / (p_i[cooc.row] * p_i[cooc.col] + 1e-9) + 1e-9)
    pmi_v = np.clip(pmi_v, 0, None)

    pmi = csr_matrix((pmi_v, (cooc.row, cooc.col)), shape=(n_items, n_items), dtype=np.float32)
    log.info(f"  PMI: {n_items:,} items | {pmi.nnz:,} non-zero pairs")
    return pmi


# ─────────────────────────────────────────────────────────────────────────────
# 7. Temporal split function
# ─────────────────────────────────────────────────────────────────────────────

def split_events_temporally(df: pd.DataFrame, test_ratio: float = 0.2):
    """Split events by timestamp (chronological) - NO LEAKAGE!"""
    df = df.sort_values('timestamp')
    cutoff = int(len(df) * (1 - test_ratio))
    train = df.iloc[:cutoff].copy()
    test = df.iloc[cutoff:].copy()
    log.info(f"Temporal split: {len(train):,} train events, {len(test):,} test events")
    return train, test


# ─────────────────────────────────────────────────────────────────────────────
# 8. Main preprocessing pipeline (SPLITS BEFORE PROCESSING)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_data(train_events: pd.DataFrame,
                    test_events: pd.DataFrame,
                    props: pd.DataFrame,
                    tree: pd.DataFrame):
    """
    Run preprocessing on training events and test events separately.
    Saves train sessions and test sessions, plus all metadata computed from training.
    """
    log.info("=" * 70)
    log.info("Preprocessing pipeline started (NO DATA LEAKAGE)")
    log.info("=" * 70)

    # ---- Build training artefacts ----
    log.info("Processing TRAINING set ...")
    raw_clean_train, events_agg_train = clean_events(train_events)
    events_agg_train = add_time_decay(events_agg_train)

    item_features = build_item_features(raw_clean_train, props, tree)
    user_features = build_user_features(raw_clean_train)
    sessions_train = build_sessions(raw_clean_train)

    # ---- Compute gap_max from training data ----
    log.info("Computing gap_max from training data...")
    all_gaps = sessions_train['time_gap_sequence'].explode()
    all_gaps = all_gaps.dropna()
    all_gaps = all_gaps[all_gaps.apply(lambda x: isinstance(x, (int, float)))]
    
    if len(all_gaps) > 0:
        all_gaps_numeric = pd.to_numeric(all_gaps, errors='coerce').dropna()
        if len(all_gaps_numeric) > 0:
            gap_max = float(np.log1p(all_gaps_numeric.values).max())
        else:
            gap_max = 1.0
            log.warning("No valid numeric gaps found, using gap_max=1.0")
    else:
        gap_max = 1.0
        log.warning("No gaps found, using gap_max=1.0")
    
    log.info(f"Training gap_max (log1p) = {gap_max:.4f}")

    # ---- Build item encoder from TRAINING items only ----
    all_items = sessions_train['item_sequence'].explode().unique()
    item_encoder = {iid: idx + 1 for idx, iid in enumerate(all_items)}
    item_ids = list(all_items)
    log.info(f"Built item encoder with {len(item_ids):,} items (from training only)")

    # ---- Build interaction matrix ----
    matrix, le_user, le_item = build_interaction_matrix(events_agg_train)
    pmi = build_pmi_cooccurrence(sessions_train, matrix.shape[1], le_item)

    # ---- Save training artefacts ----
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    
    raw_clean_train.to_parquet(TRAIN_RAW_EVENTS, index=False)
    events_agg_train.to_parquet(TRAIN_PROCESSED_EVENTS, index=False)
    user_features.to_parquet(TRAIN_USER_FEATURES, index=False)
    item_features.to_parquet(TRAIN_ITEM_FEATURES, index=False)
    sessions_train.to_parquet(TRAIN_SESSIONS, index=False)
    save_npz(TRAIN_INTERACTION_MATRIX, matrix)
    save_npz(TRAIN_PMI_MATRIX, pmi)

    with open(TRAIN_ITEM_ENCODER, 'wb') as f:
        pickle.dump(item_encoder, f)
    with open(TRAIN_ITEM_IDS, 'wb') as f:
        pickle.dump(item_ids, f)
    with open(TRAIN_GAP_MAX, 'wb') as f:
        pickle.dump(gap_max, f)

    log.info(f"Training artefacts saved to: {TRAIN_DIR}")

    # ---- Process test events ----
    if test_events is not None and len(test_events) > 0:
        log.info("Processing TEST set ...")
        raw_clean_test, _ = clean_events(test_events)
        sessions_test = build_sessions(raw_clean_test)

        # Filter test sessions to contain only items seen in training (NO LEAKAGE)
        valid_items = set(item_ids)
        sessions_test['item_sequence'] = sessions_test['item_sequence'].apply(
            lambda seq: [it for it in seq if it in valid_items]
        )
        sessions_test = sessions_test[sessions_test['item_sequence'].map(len) >= MIN_SESSION_LENGTH]
        log.info(f"Test sessions after filtering: {len(sessions_test):,}")

        TEST_DIR.mkdir(parents=True, exist_ok=True)
        sessions_test.to_parquet(TEST_SESSIONS, index=False)
        log.info(f"Test artefacts saved to: {TEST_DIR}")
    else:
        log.warning("No test events provided; skipping test processing.")

    log.info("Preprocessing complete.")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    t0 = time.perf_counter()

    # Load raw data
    raw_events = load_events(EVENTS_FILE)
    props = load_item_properties(ITEM_PROPERTIES_1, ITEM_PROPERTIES_2)
    tree = load_category_tree(CATEGORY_TREE_FILE)

    # TEMPORAL SPLIT BEFORE ANY PROCESSING (CRITICAL FOR NO LEAKAGE)
    train_events, test_events = split_events_temporally(raw_events, test_ratio=0.2)

    # Run preprocessing
    preprocess_data(train_events, test_events, props, tree)

    elapsed = time.perf_counter() - t0
    print(f"\nPreprocessing completed in {elapsed:.1f}s")