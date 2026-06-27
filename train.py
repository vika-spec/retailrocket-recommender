"""
GRU4Rec Training Script
=======================
- Loads training sessions from data/processed/train/sessions.parquet
- Builds item encoder, gap_max from training data only
- Splits training sessions temporally for validation
- Trains GRU4Rec model with early stopping
- Saves model and metadata to models_official_final/

Usage:
    python scripts/train.py
"""

import os
import sys
import json
import pickle
import logging
import warnings
import random
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.models import BaseGRU
from config import BASE_CONFIG, MODEL_DIR, TRAIN_SESSIONS, TRAIN_ITEM_ENCODER, TRAIN_GAP_MAX

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
CONFIG = BASE_CONFIG.copy()
torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
random.seed(CONFIG['seed'])

# Set CPU threads for performance
torch.set_num_threads(24)

# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class GRU4RecDataset(Dataset):
    def __init__(self, items, events, hour, day, time_gap, targets):
        self.items = torch.tensor(items, dtype=torch.long)
        self.events = torch.tensor(events, dtype=torch.long)
        self.hour = torch.tensor(hour, dtype=torch.float)
        self.day = torch.tensor(day, dtype=torch.float)
        self.time_gap = torch.tensor(time_gap, dtype=torch.float)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (self.items[idx], self.events[idx],
                self.hour[idx], self.day[idx],
                self.time_gap[idx], self.targets[idx])

# ----------------------------------------------------------------------
# Sequence preparation
# ----------------------------------------------------------------------
def prepare_sequences(sessions_df, item_encoder, max_seq_len, gap_max):
    X_items, X_events, X_hour, X_day, X_gap, y = [], [], [], [], [], []
    
    for _, row in sessions_df.iterrows():
        items = row['item_sequence']
        events = row['event_sequence']
        hours = row.get('hour_sequence', [0] * len(items))
        days = row.get('day_sequence', [0] * len(items))
        gaps = row.get('time_gap_sequence', [0] * len(items))
        
        enc_items = [item_encoder.get(i, 0) for i in items if i in item_encoder]
        if len(enc_items) < 2:
            continue
        
        min_len = min(len(enc_items), len(events), len(hours), len(days), len(gaps))
        enc_items = enc_items[:min_len]
        events = events[:min_len]
        hours = hours[:min_len]
        days = days[:min_len]
        gaps = gaps[:min_len]
        
        for i in range(1, len(enc_items)):
            hist_items = list(enc_items[:i])
            hist_events = list(events[:i])
            hist_hours = list(hours[:i])
            hist_days = list(days[:i])
            hist_gaps = list(gaps[:i])
            target = enc_items[i]
            
            if len(hist_items) > max_seq_len:
                hist_items = hist_items[-max_seq_len:]
                hist_events = hist_events[-max_seq_len:]
                hist_hours = hist_hours[-max_seq_len:]
                hist_days = hist_days[-max_seq_len:]
                hist_gaps = hist_gaps[-max_seq_len:]
            
            pad_len = max_seq_len - len(hist_items)
            X_items.append([0] * pad_len + hist_items)
            X_events.append([0] * pad_len + hist_events)
            X_hour.append([0] * pad_len + hist_hours)
            X_day.append([0] * pad_len + hist_days)
            X_gap.append([0] * pad_len + hist_gaps)
            y.append(target)
    
    X_items = np.array(X_items, dtype=np.int32)
    X_events = np.array(X_events, dtype=np.int32)
    X_hour = np.array(X_hour, dtype=np.float32) / 23.0
    X_day = np.array(X_day, dtype=np.float32) / 6.0
    X_gap = np.log1p(np.array(X_gap, dtype=np.float32))
    X_gap = X_gap / gap_max
    
    X_hour_last = X_hour[:, -1].reshape(-1, 1)
    X_day_last = X_day[:, -1].reshape(-1, 1)
    X_gap_last = X_gap[:, -1].reshape(-1, 1)
    
    y = np.array(y, dtype=np.int32)
    return X_items, X_events, X_hour_last, X_day_last, X_gap_last, y


def split_sessions_temporally(sessions_df, val_ratio, seed):
    """Split sessions by session_start time (chronological) - NO LEAKAGE."""
    sessions_sorted = sessions_df.sort_values('session_start')
    cutoff = int(len(sessions_sorted) * (1 - val_ratio))
    train = sessions_sorted.iloc[:cutoff]
    val = sessions_sorted.iloc[cutoff:]
    logger.info(f"Temporal split: {len(train):,} train sessions, {len(val):,} val sessions")
    return train, val

# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def main():
    logger.info("=" * 70)
    logger.info("GRU4Rec Training – Corrected Pipeline (NO LEAKAGE)")
    logger.info("=" * 70)
    logger.info(f"Config: {json.dumps(CONFIG, indent=2)}")
    
    # ---- Check if training sessions exist ----
    if not TRAIN_SESSIONS.exists():
        raise FileNotFoundError(
            f"Training sessions not found at {TRAIN_SESSIONS}\n"
            "Run preprocessing.py first with temporal split."
        )
    
    if not TRAIN_ITEM_ENCODER.exists():
        raise FileNotFoundError(
            f"Item encoder not found at {TRAIN_ITEM_ENCODER}\n"
            "Run preprocessing.py first."
        )
    
    if not TRAIN_GAP_MAX.exists():
        raise FileNotFoundError(
            f"gap_max not found at {TRAIN_GAP_MAX}\n"
            "Run preprocessing.py first."
        )
    
    # ---- Load training sessions ----
    logger.info(f"Loading training sessions from: {TRAIN_SESSIONS}")
    sessions = pd.read_parquet(TRAIN_SESSIONS)
    logger.info(f"Loaded {len(sessions):,} training sessions")
    
    # ---- Load item encoder and gap_max ----
    with open(TRAIN_ITEM_ENCODER, 'rb') as f:
        item_encoder = pickle.load(f)
    with open(TRAIN_GAP_MAX, 'rb') as f:
        gap_max = pickle.load(f)
    
    item_ids = list(item_encoder.keys())
    num_items = len(item_ids) + 1  # +1 for padding
    logger.info(f"Unique items: {len(item_ids):,} (encoder size: {num_items})")
    logger.info(f"Gap max (from training): {gap_max:.4f}")
    
    # ---- Split sessions temporally for validation ----
    val_ratio = CONFIG.get('val_ratio', 0.1)
    train_sess, val_sess = split_sessions_temporally(sessions, val_ratio, CONFIG['seed'])
    
    # ---- Prepare sequences ----
    logger.info("Preparing training sequences...")
    X_tr, E_tr, H_tr, D_tr, G_tr, y_tr = prepare_sequences(
        train_sess, item_encoder, CONFIG['max_seq_len'], gap_max
    )
    logger.info(f"Training examples: {len(y_tr):,}")
    
    logger.info("Preparing validation sequences...")
    X_val, E_val, H_val, D_val, G_val, y_val = prepare_sequences(
        val_sess, item_encoder, CONFIG['max_seq_len'], gap_max
    )
    logger.info(f"Validation examples: {len(y_val):,}")
    
    # ---- DataLoaders ----
    train_ds = GRU4RecDataset(X_tr, E_tr, H_tr, D_tr, G_tr, y_tr)
    val_ds = GRU4RecDataset(X_val, E_val, H_val, D_val, G_val, y_val)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2
    )
    
    # ---- Model ----
    num_events = CONFIG.get('num_events', 4)
    model = BaseGRU(
        num_items=num_items,
        num_events=num_events,
        embedding_dim=CONFIG['embedding_dim'],
        event_embedding_dim=CONFIG['event_embedding_dim'],
        hidden_size=CONFIG['hidden_size'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout']
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    logger.info(f"Model on device: {device}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ---- Optimizer, Scheduler, Loss ----
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    ) if CONFIG.get('use_scheduler', True) else None
    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG.get('label_smoothing', 0.0))
    
    # ---- Training loop ----
    best_val_loss = float('inf')
    patience_counter = 0
    history = []
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        for batch in progress:
            items, events, hour, day, gap, target = [b.to(device) for b in batch]
            optimizer.zero_grad()
            logits = model(items, events, hour, day, gap)
            loss = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['gradient_clip'])
            optimizer.step()
            
            batch_size = items.size(0)
            train_loss += loss.item() * batch_size
            _, pred = torch.max(logits, 1)
            train_correct += (pred == target).sum().item()
            train_total += batch_size
            progress.set_postfix({'loss': f"{loss.item():.4f}"})
        
        train_loss /= len(train_ds)
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                items, events, hour, day, gap, target = [b.to(device) for b in batch]
                logits = model(items, events, hour, day, gap)
                loss = criterion(logits, target)
                val_loss += loss.item() * items.size(0)
                _, pred = torch.max(logits, 1)
                val_correct += (pred == target).sum().item()
                val_total += items.size(0)
        val_loss /= len(val_ds)
        val_acc = val_correct / val_total
        
        if scheduler:
            scheduler.step(val_loss)
        
        logger.info(
            f"Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )
        history.append({
            'epoch': epoch+1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_DIR / 'model_best.pt')
            logger.info(f"  ✓ New best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    # ---- Save final artefacts ----
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(torch.load(MODEL_DIR / 'model_best.pt'))
    torch.save(model.state_dict(), MODEL_DIR / 'model_best.pt')
    
    config_to_save = CONFIG.copy()
    config_to_save['num_items'] = num_items
    config_to_save['num_events'] = num_events
    config_to_save['gap_max'] = float(gap_max)
    config_to_save['trained_at'] = datetime.now().isoformat()
    with open(MODEL_DIR / 'config.json', 'w') as f:
        json.dump(config_to_save, f, indent=2)
    
    # Copy metadata from train directory to model directory
    with open(MODEL_DIR / 'item_encoder.pkl', 'wb') as f:
        pickle.dump(item_encoder, f)
    with open(MODEL_DIR / 'item_ids.pkl', 'wb') as f:
        pickle.dump(item_ids, f)
    with open(MODEL_DIR / 'gap_max.pkl', 'wb') as f:
        pickle.dump(float(gap_max), f)
    
    pd.DataFrame(history).to_parquet(MODEL_DIR / 'training_history.parquet')
    
    logger.info("=" * 70)
    logger.info("Training complete.")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Model and metadata saved to {MODEL_DIR}")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()